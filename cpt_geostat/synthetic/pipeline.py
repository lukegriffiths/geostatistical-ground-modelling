"""End-to-end synthetic generation (A1-A7)."""

from __future__ import annotations

from ..contract.schema import Dataset, Raster
from ..contract.summarise import summarise_units
from .config import Config
from .fields import property_raster
from .layout import make_layout
from .series import make_depth_series, sample_unit_values
from .strat import assemble, build_strat_rasters
from .truth import realised_truth


def generate(cfg: Config) -> Dataset:
    raster = Raster.from_site(cfg.site.size_km, cfg.site.raster_res_km)
    layout = make_layout(cfg)

    strat = build_strat_rasters(cfg, raster)
    layers = assemble(cfg, raster, layout, strat)

    prop, trend_only, grf_only = {}, {}, {}
    for uid, unit in cfg.units.items():
        prop[uid], trend_only[uid], grf_only[uid] = property_raster(
            raster, unit.property, cfg.rng("property", uid)
        )

    unit_values = sample_unit_values(cfg, raster, layout, prop)
    samples = make_depth_series(cfg, layers, unit_values, layout)
    unit_summary = summarise_units(samples, layers, layout)

    rasters = {
        "presence_prob": strat.presence_prob,
        "thickness": strat.thickness,
        "property": prop,
        "trend": trend_only,
        "grf": grf_only,
    }

    return Dataset(
        layout=layout,
        layers=layers,
        samples=samples,
        unit_summary=unit_summary,
        config=cfg,
        raster=raster,
        rasters=rasters,
        unit_values=unit_values,
        truth=realised_truth(
            cfg, raster, layout, layers, samples, unit_summary, unit_values, rasters
        ),
    )
