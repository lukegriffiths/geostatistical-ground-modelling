"""Config schema — the single source of truth for a synthetic dataset.

Everything downstream is a pure function of ``(config, seed)``, so ``truth.yaml``
is just the resolved config written back out alongside realised statistics.

Random number streams are derived by *name* (``cfg.rng("presence", "unit_3")``)
rather than by draw order, so adding a unit or a plot does not perturb the
realisations of everything else.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import numpy as np
import yaml

from ..covariance import GrfConfig

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _unwrap_optional(tp):
    """``Optional[X]`` -> ``X``; anything else unchanged."""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _build(cls, data: Any):
    """Recursively build a (possibly nested) dataclass from plain dicts.

    ``from __future__ import annotations`` turns field types into strings, so the
    concrete types are resolved with ``get_type_hints`` rather than ``f.type``.
    """
    if data is None:
        return None
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config key(s) {sorted(unknown)}")

    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        ftype = _unwrap_optional(hints.get(f.name))
        kwargs[f.name] = _build(ftype, data[f.name]) if is_dataclass(ftype) else data[f.name]
    return cls(**kwargs)


def substream(seed: int, *tags: Any) -> np.random.Generator:
    """A named, order-independent RNG stream derived from the master seed."""
    key = "|".join(str(t) for t in tags).encode("utf-8")
    h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    return np.random.default_rng([int(seed), h])


def gstools_seed(rng: np.random.Generator) -> int:
    """gstools wants a plain int seed."""
    return int(rng.integers(0, 2**31 - 1))


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #


@dataclass
class SiteConfig:
    size_km: float = 15.0
    raster_res_km: float = 0.1
    max_depth_m: float = 50.0


@dataclass
class ClusterConfig:
    """A few tightly spaced CPTs so the variogram has short-lag pairs."""

    n: int = 4
    centre_km: Sequence[float] = (2.0, -1.5)
    spread_km: float = 0.06


@dataclass
class DropCornerConfig:
    """Thin one corner to create a data-sparse region for the sensitivity sweep."""

    enabled: bool = True
    corner: str = "NE"  # NE / NW / SE / SW
    radius_km: float = 4.0
    keep_fraction: float = 0.3


@dataclass
class LayoutConfig:
    grid_n: int = 11  # per side -> grid_n**2 nominal CPTs
    margin_km: float = 0.7
    jitter_km: float = 0.35
    cluster: Optional[ClusterConfig] = None
    drop_corner: Optional[DropCornerConfig] = None


@dataclass
class PresenceConfig:
    """How a unit is (or is not) present laterally.

    mode:
      ``everywhere`` - present at every location.
      ``channel``    - sinuous ribbon; centreline is a sine in a rotated frame.
      ``patch``      - smooth GRF thresholded at a quantile hitting ``coverage``.
    """

    mode: str = "everywhere"
    # channel
    width_km: float = 2.2
    sinuosity: float = 0.35
    wavelength_km: float = 12.0
    azimuth_deg: float = 70.0
    offset_km: float = 0.0
    phase_deg: float = 0.0
    ragged_range_km: float = 1.5
    # patch
    coverage: float = 0.45
    range_km: float = 3.0
    # both
    edge_softness: float = 0.3


@dataclass
class ThicknessConfig:
    mean_m: float = 6.0
    sd_m: float = 2.0
    range_km: float = 4.0
    min_m: float = 1.0
    max_m: float = 10.0


@dataclass
class TrendConfig:
    grad: float = 0.0  # per km, in log Qtn units
    azimuth_deg: float = 0.0


@dataclass
class WithinUnitConfig:
    """Depth-to-depth scatter about the unit value."""

    sd: float = 0.18
    corr_len_m: float = 0.4  # AR(1) correlation length; 0 -> white noise
    drift_per_m: float = 0.0  # slow linear drift in log Qtn across the interval


@dataclass
class PropertyConfig:
    mu: float = 1.6  # mean of log Qtn
    trend: TrendConfig = field(default_factory=TrendConfig)
    grf: GrfConfig = field(default_factory=GrfConfig)
    nugget: float = 0.01  # variance of the location-wise measurement/micro term
    within_unit: WithinUnitConfig = field(default_factory=WithinUnitConfig)


@dataclass
class UnitConfig:
    name: str = ""
    colour: str = "#888888"
    purpose: str = ""
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    thickness: ThicknessConfig = field(default_factory=ThicknessConfig)
    property: PropertyConfig = field(default_factory=PropertyConfig)


@dataclass
class SamplingConfig:
    dz_m: float = 0.02  # real CPT spacing
    min_thickness_m: float = 1.0  # retained-thickness guard


@dataclass
class SectionConfig:
    name: str
    p0_km: Sequence[float]
    p1_km: Sequence[float]
    corridor_km: float = 1.2


@dataclass
class Config:
    seed: int = 20260728
    site: SiteConfig = field(default_factory=SiteConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    units: Dict[str, UnitConfig] = field(default_factory=dict)
    sections: List[SectionConfig] = field(default_factory=list)

    # ---- derived ----------------------------------------------------------
    @property
    def unit_ids(self) -> List[str]:
        """Units in stratigraphic order (shallowest first) — dict insertion order."""
        return list(self.units)

    @property
    def unit_colours(self) -> Dict[str, str]:
        return {u: c.colour for u, c in self.units.items()}

    def rng(self, *tags: Any) -> np.random.Generator:
        return substream(self.seed, *tags)

    # ---- (de)serialisation ------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        data = dict(data)
        units = {k: _build(UnitConfig, v) for k, v in (data.pop("units", {}) or {}).items()}
        sections = [_build(SectionConfig, s) for s in (data.pop("sections", []) or [])]
        cfg = _build(cls, data)
        cfg.units = units
        cfg.sections = sections
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    # ---- checks -----------------------------------------------------------
    def validate(self) -> None:
        if not self.units:
            raise ValueError("config defines no units")
        for uid, u in self.units.items():
            if u.presence.mode not in {"everywhere", "channel", "patch"}:
                raise ValueError(f"{uid}: unknown presence mode {u.presence.mode!r}")
            t = u.thickness
            if not (t.min_m < t.max_m):
                raise ValueError(f"{uid}: thickness min_m must be < max_m")
            if not (t.min_m <= t.mean_m <= t.max_m):
                raise ValueError(f"{uid}: thickness mean_m outside [min_m, max_m]")
            g = u.property.grf
            if g.aniso_ratio < 1.0:
                raise ValueError(
                    f"{uid}: aniso_ratio is major/minor and must be >= 1 "
                    f"(rotate aniso_angle_deg by 90 instead)"
                )
            if g.sill < 0 or u.property.nugget < 0:
                raise ValueError(f"{uid}: sill and nugget must be non-negative")
        if self.sampling.dz_m <= 0:
            raise ValueError("sampling.dz_m must be positive")
