"""Pydantic models for tform configuration.

These are the single source of truth for:
  - Default values
  - Type coercion  (Pydantic handles "5.0e15" str → float automatically)
  - Validation     (field constraints replace the manual validate() function)
  - Serialisation  (model_dump() for YAML / runner consumption)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Enums (shared by Click choices and Pydantic fields) ───────────────────────

class Accuracy(str, Enum):
    fast     = "fast"
    accurate = "accurate"


class ExpType(str, Enum):
    sol          = "sol"
    year         = "year"
    multi        = "multi"
    spots        = "spots"
    intervention = "intervention"


# ── Annotated constraint aliases ──────────────────────────────────────────────
# Separating constraints (Annotated) from defaults (= value) lets Pyright
# correctly infer that every __init__ parameter is optional.

_PosFloat   = Annotated[float, Field(gt=0)]
_NNFloat    = Annotated[float, Field(ge=0)]
_Albedo     = Annotated[float, Field(ge=0.0, le=1.0)]
_Greenhouse = Annotated[float, Field(ge=1.0)]
_Latitude   = Annotated[float, Field(ge=-90.0, le=90.0)]


# ── Sub-models ────────────────────────────────────────────────────────────────

class PlanetConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    surface_temperature: _PosFloat                     = 210.0
    surface_pressure:    _PosFloat                     = 610.0
    albedo:              _Albedo                       = 0.25
    greenhouse_factor:   _Greenhouse                   = 1.02
    ice_mass:            _NNFloat                      = 5.0e15
    latitude:            _Latitude                     = 22.0
    longitude:           float                         = 0.0
    elevation_m:         float                         = 0.0
    initial_ls_deg:      float                         = 251.0
    composition:         Optional[dict[str, Any]]      = None


class EngineConfig(BaseModel):
    dt:       _PosFloat = 3600.0
    accuracy: Accuracy  = Accuracy.fast


class ExperimentConfig(BaseModel):
    type: ExpType    = ExpType.sol
    sols: _PosFloat  = 1.0


class InterventionConfig(BaseModel):
    n_years:            Annotated[int, Field(gt=0)]   = 50
    injection:          dict[str, float]              = {}


class OutputConfig(BaseModel):
    save_csv:     bool          = True
    save_plot:    bool          = False
    ground_truth: bool          = False
    out_dir:      Optional[str] = None
    output_path:  Optional[str] = None


class PresetMeta(BaseModel):
    name:        str = "custom"
    description: str = "Custom configuration"


class SimConfig(BaseModel):
    """Complete, validated configuration for a single simulation run."""

    model_config = ConfigDict(populate_by_name=True)

    preset:       PresetMeta         = PresetMeta()
    planet:       PlanetConfig       = PlanetConfig()
    engine:       EngineConfig       = EngineConfig()
    experiment:   ExperimentConfig   = ExperimentConfig()
    intervention: InterventionConfig = InterventionConfig()
    output:       OutputConfig       = OutputConfig()

    @model_validator(mode="before")
    @classmethod
    def _coerce_str_floats(cls, data: Any) -> Any:
        """PyYAML sometimes parses scientific notation as strings; coerce them."""
        if not isinstance(data, dict):
            return data
        planet = data.get("planet", {})
        if isinstance(planet, dict):
            for key in ("surface_temperature", "surface_pressure", "albedo",
                        "greenhouse_factor", "ice_mass", "latitude", "longitude",
                        "elevation_m", "initial_ls_deg"):
                if key in planet and isinstance(planet[key], str):
                    try:
                        planet[key] = float(planet[key])
                    except ValueError:
                        pass
        engine = data.get("engine", {})
        if isinstance(engine, dict) and "dt" in engine and isinstance(engine["dt"], str):
            try:
                engine["dt"] = float(engine["dt"])
            except ValueError:
                pass
        exp = data.get("experiment", {})
        if isinstance(exp, dict) and "sols" in exp and isinstance(exp["sols"], str):
            try:
                exp["sols"] = float(exp["sols"])
            except ValueError:
                pass
        return data


# ── CLI-flag overlay model ─────────────────────────────────────────────────────

class RunFlags(BaseModel):
    """Typed container for all optional CLI flag values from `tform mars run`.

    Fields are Optional so that None means "not provided by the user".
    `apply()` merges only the non-None values onto a SimConfig.
    """

    exp_type:          Optional[ExpType]  = None
    sols:              Optional[float]    = None
    accuracy:          Optional[Accuracy] = None
    dt:                Optional[float]    = None
    lat:               Optional[float]    = None
    lon:               Optional[float]    = None
    elevation:         Optional[float]    = None
    ls:                Optional[float]    = None
    ice_mass:          Optional[float]    = None
    surface_temp:      Optional[float]    = None
    pressure:          Optional[float]    = None
    albedo:            Optional[float]    = None
    greenhouse_factor: Optional[float]    = None
    name:              Optional[str]      = None
    output_dir:        Optional[str]      = None
    no_save:           bool               = False
    plot:              bool               = False
    # intervention flags
    n_years:           Optional[int]      = None
    inject:            Optional[dict[str, float]] = None

    def apply(self, base: SimConfig) -> SimConfig:
        """Return a new SimConfig with non-None flags merged over *base*."""
        p = base.planet.model_dump()
        e = base.engine.model_dump()
        x = base.experiment.model_dump()
        o = base.output.model_dump()

        if self.exp_type is not None:
            x["type"] = self.exp_type
        if self.sols is not None:
            x["sols"] = self.sols
        if self.accuracy is not None:
            e["accuracy"] = self.accuracy
        if self.dt is not None:
            e["dt"] = self.dt
        if self.lat is not None:
            p["latitude"] = self.lat
        if self.lon is not None:
            p["longitude"] = self.lon
        if self.elevation is not None:
            p["elevation_m"] = self.elevation
        if self.ls is not None:
            p["initial_ls_deg"] = self.ls
        if self.ice_mass is not None:
            p["ice_mass"] = self.ice_mass
        if self.surface_temp is not None:
            p["surface_temperature"] = self.surface_temp
        if self.pressure is not None:
            p["surface_pressure"] = self.pressure
        if self.albedo is not None:
            p["albedo"] = self.albedo
        if self.greenhouse_factor is not None:
            p["greenhouse_factor"] = self.greenhouse_factor
        if self.name is not None:
            o["out_dir"] = self.name
        if self.output_dir is not None:
            o["output_path"] = self.output_dir
        if self.no_save:
            o["save_csv"] = False
        if self.plot:
            o["save_plot"] = True

        iv = base.intervention.model_dump()
        if self.n_years is not None:
            iv["n_years"] = self.n_years
        if self.inject is not None:
            iv["injection"] = self.inject

        return SimConfig.model_validate({
            "preset":       base.preset.model_dump(),
            "planet":       p,
            "engine":       e,
            "experiment":   x,
            "intervention": iv,
            "output":       o,
        })
