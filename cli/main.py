"""tform — interactive CLI for Mars climate simulation.

Entry point: ``tform``

Usage
-----
  tform mars run [OPTIONS]
  tform mars config list
  tform mars config show <preset>
  tform mars config validate <file>
  tform man [planet]
  tform --version

Examples
--------
  tform mars run --preset gale-crater
  tform mars run --preset current-mars --sols 5 --accuracy accurate
  tform mars run --config my_run.yaml --lat 45
  tform mars run --type year --preset early-mars
  tform mars run --type multi --sols 3 --accuracy fast
  tform mars config list
  tform mars config show gale-crater
  tform man mars
"""

from __future__ import annotations

import sys

import click
import yaml
from pydantic import ValidationError

from cli import config_loader
from cli import presets as _presets
from cli.models import Accuracy, ExpType, RunFlags, SimConfig
from src.interventions.compounds import list_compounds as _list_compounds

VERSION = "0.1.0"


# ── Space-themed palette helpers ───────────────────────────────────────────────

def _c(text: str, fg: str, bold: bool = False, dim: bool = False) -> str:
    return click.style(text, fg=fg, bold=bold, dim=dim)

def _stars(text: str)  -> str: return _c(text, "yellow", dim=True)
def _planet(text: str) -> str: return _c(text, "bright_red", bold=True)
def _label(text: str)  -> str: return _c(text, "bright_black")
def _value(text: str)  -> str: return _c(text, "bright_white")
def _divider(width: int = 62) -> str: return _c("─" * width, "bright_red", dim=True)


# ── ASCII banner ───────────────────────────────────────────────────────────────

_MARS_ART = [
    ("  ★  ·  *  ·   ·  *  ·  ★  ·   ·  *  ·  ★  ·  *  ", "yellow"     ),
    ("                                                    ", None        ),
    ("           .-~~~~~~~~~~~~~~~~~~~-.                  ", "bright_red" ),
    ("         /  ░▒▓▓████████████▓▒░  \\                ", "bright_red" ),
    ("        |  ░▓▒█████████████████░▓  |               ", "bright_red" ),
    ("        |  ▓░░▒████████████████▒░  |               ", "bright_red" ),
    ("        |  ▓░░▒████████████████▒░  |               ", "bright_red" ),
    ("        |  ▓░░▒████████████████▒░  |               ", "bright_red" ),
    ("        |  ▓░░▒████████████████▒░  |               ", "bright_red" ),
    ("        |  ░▓▒█████████████████░▓  |               ", "bright_red" ),
    ("         \\  ▓░░▒████████████░▒░  /                ", "bright_red" ),
    ("           `~~~~~~~~~~~~~~~~~~~`                    ", "bright_red" ),
    ("                                                    ", None        ),
    ("  ★  ·  *  ·   ·  *  ·  ★  ·   ·  *  ·  ★  ·  *  ", "yellow"     ),
]


def _print_banner() -> None:
    click.echo()
    for line, color in _MARS_ART:
        click.echo(_c(line, color) if color else line)
    click.echo()
    click.echo(
        "  " + _planet("tform")
        + _c("  ·  Terraforming environments, starting with Mars ·  ", "bright_white")
        + _c(f"v{VERSION}", "bright_black")
    )
    click.echo()


# ── Man pages ──────────────────────────────────────────────────────────────────

_MAN_ROOT = """\
\b
tform(1)                 Terraforming Mars CLI

NAME
    tform — Mars climate simulation command-line interface

SYNOPSIS
    tform <planet> <command> [OPTIONS]
    tform man [planet]
    tform --version

DESCRIPTION
    tform drives physics-based simulations of planetary atmospheres and
    climates using the terraforming framework.  The first argument is always
    the target planet object.

    When run without a --preset or --config flag, tform mars run enters an
    interactive setup wizard that guides you through selecting parameters.

PLANETS
    mars      Mars — CO₂ atmosphere, polar caps, 687-day year

COMMANDS
    run       Run a simulation
    config    Manage presets and config files

GLOBAL OPTIONS
    --preset, -p NAME    Use a built-in preset as the base configuration.
    --config, -c FILE    Path to a YAML config file (merged over preset).
    --version            Print version and exit.
    --help               Show this message and exit.

SEE ALSO
    tform mars run --help
    tform mars config --help
    tform man mars
"""

_MAN_MARS = """\
\b
tform-mars(1)            Mars Simulation Manual

NAME
    tform mars — simulate the Martian climate system

SYNOPSIS
    tform [--preset NAME] [--config FILE] mars run [OPTIONS]
    tform mars config list
    tform mars config show <preset>
    tform mars config validate <file>

DESCRIPTION
    Simulates the coupled evolution of Mars surface temperature (T),
    atmospheric pressure (P), and CO₂ ice mass (M_ice) as a system of
    ordinary differential equations integrated forward in time.

    Two integration modes:
      fast      Reduced-order analytical update.  Suitable for year-long runs.
      accurate  4th-order Runge-Kutta.  Recommended for diurnal cycle studies.

    When no --preset or --config is given, tform mars run enters an interactive
    wizard prompting for experiment type, preset selection, and parameter
    overrides before starting the simulation.

INITIAL CONDITIONS
    --surface-temp FLOAT     Surface temperature in K          (default 210)
    --pressure FLOAT         Surface pressure in Pa            (default 610)
    --albedo FLOAT           Bond albedo 0–1                   (default 0.25)
    --greenhouse-factor FLOAT Greenhouse enhancement ≥1        (default 1.02)
    --ice-mass FLOAT         Total CO₂ ice mass in kg          (default 5e15)
    --lat FLOAT              Latitude in degrees N (-90–90)    (default 22)
    --lon FLOAT              Longitude in degrees E (0–360)    (default 0)
    --elevation FLOAT        Elevation in m vs Mars datum       (default 0)
    --ls FLOAT               Initial Solar Longitude in degrees (default 251)

ENGINE OPTIONS
    --accuracy fast|accurate  Integration accuracy              (default fast)
    --dt FLOAT                Timestep in seconds               (default 3600)

EXPERIMENT TYPES (--type)
    sol          Simulate N sols at a fixed location  (use --sols N)
    year         Simulate one full Martian year (668 sols)
    multi        Simulate N sols at 45°N, Equator, -40°S simultaneously
    spots        Simulate N sols at 4 landmark sites simultaneously:
                   1. Olympus Mons   (18.65°N, 226.2°E, +2000 m)
                   2. Elysium Mons   (25.02°N, 147.2°E, +1500 m)
                   3. Hellas Basin   (39.0°S,   61.0°E, -4000 m)
                   4. South Polar Cap(73.0°S,  305.0°E, +1800 m)
    intervention Annual GHG injection loop  (use --years N and --inject)

INTERVENTION OPTIONS
    --years N                   Number of Mars years to simulate.
    --inject COMPOUND:KG_PER_YEAR
                                Inject a super-greenhouse gas at a fixed
                                annual rate.  Repeat for multiple compounds.
                                Omit to run a baseline (no injection) run.

    Available compounds (Marinova 2005 RF efficiencies):

      CF4    Carbon tetrafluoride (CFC-14)        0.0880 W/m²/ppb  lifetime 50 000 yr
             Most stable perfluorocarbon; near-permanent once injected.
             Best choice for long-horizon warming that outlasts the mission.

      C2F6   Hexafluoroethane (CFC-116)           0.2600 W/m²/ppb  lifetime 10 000 yr
             3× stronger forcing than CF4; industrial semiconductor byproduct.
             Good balance of high efficiency and very long persistence.

      C3F8   Octafluoropropane (CFC-218)          0.2400 W/m²/ppb  lifetime  2 600 yr
             Common refrigerant and fire suppressant on Earth.
             Useful for medium-term scenarios where some reversibility is wanted.

      SF6    Sulfur hexafluoride                  0.5700 W/m²/ppb  lifetime  3 200 yr
             Highest RF efficiency of all registered compounds — 6.5× CF4.
             Used as an electrical insulator; most warming per kg injected.

      NF3    Nitrogen trifluoride                 0.2100 W/m²/ppb  lifetime    500 yr
             Shortest lifetime; used in LCD and solar panel manufacturing.
             Best for fast early warming; significant decay within centuries.

      C4F10  Decafluorobutane (CFC-31-10)         0.3600 W/m²/ppb  lifetime  2 600 yr
             Fire suppression agent with strong warming potential.
             Similar lifetime to C3F8 but higher RF efficiency.

      C6F14  Tetradecafluorohexane (CFC-41-12)    0.4900 W/m²/ppb  lifetime  3 200 yr
             Heaviest molecule in the registry; used as a heat-transfer fluid.
             High RF efficiency with multi-millennial persistence.

    Each year the controller:
      1. Injects the scheduled mass (kg → Pa added to atmosphere.composition)
      2. Integrates the coupled ODE for one Mars year with the updated GHF
      3. Decays all GHG partial pressures by their atmospheric lifetime
      4. Records an InterventionSnapshot (T, P, ice, ΔF, GHF, ppb per species)

OUTPUT OPTIONS
    --no-save           Skip CSV output files
    --no-plot           Skip matplotlib plots
    --name TAG          Subfolder name under outputs/ (default: UTC timestamp)
    --output-dir PATH   Full custom output directory path (overrides --name)

BUILT-IN PRESETS
    current-mars        Present-day Mars (baseline)
    gale-crater         Curiosity rover site — Gale Crater (4.6°S, 137.4°E)
    early-mars          Noachian-era Mars (1 atm, 273 K)
    terraforming-phase1 Post-CO₂ release (5 kPa, 240 K)
    equatorial          3-latitude multi-coord survey
    polar               North polar cap (85°N)

EXAMPLES
    # Quick 1-sol at Gale Crater
    tform mars run --preset gale-crater

    # 5 accurate sols at 45°N
    tform mars run --lat 45 --sols 5 --accuracy accurate --dt 300

    # Full Martian year from early-Mars initial conditions
    tform mars run --preset early-mars --type year

    # Interactive wizard (no flags)
    tform mars run

    # 1 sol across all 4 landmark sites
    tform mars run --type spots --sols 1 -y

    # 5 accurate sols across all 4 landmark sites
    tform mars run --type spots --sols 5 --accuracy accurate --preset landmark-spots -y

    # 50-year intervention injecting CF4 and SF6
    tform mars run --type intervention --years 50 --inject CF4:1e9 --inject SF6:5e8

    # Baseline intervention run (no injection — physics only)
    tform mars run --type intervention --years 50

CONFIG FILE FORMAT
    YAML with sections: preset, planet, engine, experiment, output.
    Omitted keys fall back to defaults. CLI flags always override YAML values.

    preset:
      name: my_scenario
    planet:
      surface_temperature: 210.0
      surface_pressure: 610.0
      albedo: 0.25
      greenhouse_factor: 1.02
      ice_mass: 5.0e15
      latitude: 22.0
      longitude: 0.0
      elevation_m: 0.0
      initial_ls_deg: 251.0
    engine:
      dt: 300.0
      accuracy: accurate
    experiment:
      type: sol
      sols: 1.0
    output:
      save_csv: true
      save_plot: false
      out_dir: null
"""

_MAN_PAGES = {"mars": _MAN_MARS}


# ── Interactive wizard ─────────────────────────────────────────────────────────

def _wizard_mars(
    given_preset:   str | None,
    given_config:   str | None,
    given_exp_type: ExpType | None,
    given_flags:    RunFlags,
) -> tuple[str | None, RunFlags]:
    """Interactive setup wizard for a Mars simulation.

    Only prompts for values not already provided via flags/preset/config.
    Returns (resolved_preset_name, RunFlags with wizard-collected values).
    """
    click.echo()
    click.echo(_divider())
    click.echo(
        "  " + _c("✦", "bright_yellow")
        + _c("  Simulation setup", "bright_white", bold=True)
        + _c("  — flags override wizard defaults", "bright_black")
    )
    click.echo(_divider())
    click.echo()

    wizard_updates: dict = {}

    # ── Experiment type ────────────────────────────────────────────────────────
    if given_exp_type is not None:
        exp_type = given_exp_type
        click.echo(
            _label("  type     ")
            + _c(exp_type.value, "bright_white")
            + _c("  (from --type flag)", "bright_black")
        )
    else:
        click.echo(_c("  Experiment type:", "bright_white", bold=True))
        type_choices = [
            (ExpType.sol,   "N sols at a fixed location"),
            (ExpType.year,  "Full Martian year (668 sols)"),
            (ExpType.multi, "3 latitudes simultaneously  (45°N / equator / 40°S)"),
            (ExpType.spots, "4 landmark sites  (Olympus Mons / Elysium / Hellas / South Pole)"),
        ]
        for i, (key, desc) in enumerate(type_choices, 1):
            click.echo(
                f"    {_c(str(i), 'bright_yellow', bold=True)}  "
                + _c(f"{key.value:<6}", "bright_white")
                + _c(f"  {desc}", "bright_black")
            )
        idx      = click.prompt(click.style("\n  Select type", fg="bright_white"),
                                type=click.IntRange(1, 4), default=1, show_default=True)
        exp_type = type_choices[idx - 1][0]
    wizard_updates["exp_type"] = exp_type

    # ── Preset ─────────────────────────────────────────────────────────────────
    preset_name = given_preset
    if given_preset is not None:
        click.echo(
            _label("  preset   ") + _c(given_preset, "bright_white")
            + _c("  (from --preset flag)", "bright_black")
        )
    elif given_config is not None:
        click.echo(
            _label("  config   ") + _c(given_config, "bright_white")
            + _c("  (from --config flag)", "bright_black")
        )
    else:
        click.echo()
        if click.confirm(_c("  Use a built-in preset?", "bright_white"), default=True):
            click.echo()
            click.echo(_c("  Built-in presets:", "bright_white", bold=True))
            for i, name in enumerate(_presets.MARS_PRESET_NAMES, 1):
                desc = _presets.MARS_PRESET_DESCRIPTIONS[name]
                click.echo(
                    f"    {_c(str(i), 'bright_yellow', bold=True)}  "
                    + _c(f"{name:<26}", "bright_white")
                    + _c(desc, "bright_black")
                )
            pidx        = click.prompt(
                click.style("\n  Select preset", fg="bright_white"),
                type=click.IntRange(1, len(_presets.MARS_PRESET_NAMES)),
                default=1, show_default=True,
            )
            preset_name = _presets.MARS_PRESET_NAMES[pidx - 1]
            click.echo("  " + _c("✓", "bright_green") + _c(f"  Preset: {preset_name}", "bright_white"))

    # Load base so prompts show actual preset defaults
    base = config_loader.load(planet="mars", preset=preset_name, config=given_config)
    p, e, x = base.planet, base.engine, base.experiment

    # ── Parameter overrides ────────────────────────────────────────────────────
    click.echo()
    click.echo(
        _c("  Parameters", "bright_white", bold=True)
        + _c("  (Enter = keep default, shown in brackets):", "bright_black")
    )
    click.echo()

    def _ask_float(label: str, field: str, default: float, unit: str = "") -> None:
        existing = getattr(given_flags, field, None)
        if existing is not None:
            click.echo(
                f"    {_label(f'{label:<22}')}"
                + _c(str(existing), "bright_white")
                + _c(f"  (from --{field.replace('_','-')} flag)", "bright_black")
            )
            return
        hint = _c(f"  ({unit})" if unit else "", "bright_black")
        val  = click.prompt(f"    {_c(label, 'bright_white')}{hint}",
                            default=default, type=float, show_default=True)
        if val != default:
            wizard_updates[field] = val

    def _ask_accuracy() -> None:
        if given_flags.accuracy is not None:
            click.echo(
                f"    {_label('accuracy              ')}"
                + _c(given_flags.accuracy.value, "bright_white")
                + _c("  (from --accuracy flag)", "bright_black")
            )
            return
        default = e.accuracy.value
        click.echo(
            f"    {_c('Integration accuracy  ', 'bright_white')}"
            + _c("(fast / accurate)", "bright_black")
            + f"  [{_c(default, 'bright_yellow')}]: ",
            nl=False,
        )
        raw = input().strip()
        if raw in ("fast", "accurate") and raw != default:
            wizard_updates["accuracy"] = Accuracy(raw)

    if exp_type == ExpType.sol:
        _ask_float("Sols to simulate", "sols",      x.sols,           "sols")
    _ask_accuracy()
    if exp_type not in (ExpType.year, ExpType.spots):
        _ask_float("Latitude",         "lat",        p.latitude,       "°N")
        _ask_float("Longitude",        "lon",        p.longitude,      "°E")
        _ask_float("Elevation",        "elevation",  p.elevation_m,    "m")
    if exp_type == ExpType.spots:
        click.echo(
            _label("  location ")
            + _c("fixed — Olympus Mons / Elysium Mons / Hellas Basin / South Polar Cap",
                  "bright_black")
        )
    _ask_float("Solar longitude (Ls)", "ls",         p.initial_ls_deg, "°")

    click.echo()

    # Merge wizard-collected values into a new RunFlags, keeping user-supplied flags intact
    wizard_flags = RunFlags.model_validate({
        **given_flags.model_dump(exclude_none=True),
        **wizard_updates,
    })
    return preset_name, wizard_flags


# ── Root group ─────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.version_option(VERSION, "--version", "-V",
                      message=_c("tform", "bright_red", bold=True) + " %(version)s")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """tform — terraforming simulation CLI.

    \b
    Usage:
      tform mars run [OPTIONS]
      tform mars config list
      tform man [mars]

    Pass --help after any command for detailed flag documentation.
    """
    if ctx.invoked_subcommand is None:
        _print_banner()
        click.echo(ctx.get_help())


# ── man command ────────────────────────────────────────────────────────────────

@cli.command("man")
@click.argument("planet", required=False, default=None,
                type=click.Choice(list(_MAN_PAGES) + ["root"]))
def man_cmd(planet: str | None) -> None:
    """Display the manual page for tform or a specific planet.

    \b
    Examples:
      tform man
      tform man mars
    """
    if planet is None or planet == "root":
        click.echo_via_pager(_MAN_ROOT)
    else:
        click.echo_via_pager(_MAN_MARS)


# ── mars group ─────────────────────────────────────────────────────────────────

@cli.group("mars", invoke_without_command=True)
@click.option("--preset", "-p", type=click.Choice(_presets.MARS_PRESET_NAMES),
              default=None, help="Use a built-in preset as the base configuration.")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Path to a YAML config file (merged over preset).")
@click.pass_context
def mars_group(ctx: click.Context, preset: str | None, config: str | None) -> None:
    """Simulate the Martian climate system.

    \b
    Examples:
      tform mars run --preset gale-crater
      tform mars run --type year --preset early-mars
      tform mars config list
      tform man mars
    """
    ctx.ensure_object(dict)
    ctx.obj["cfg"]    = config_loader.load(planet="mars", preset=preset, config=config)
    ctx.obj["preset"] = preset
    ctx.obj["config"] = config
    if ctx.invoked_subcommand is None:
        _print_banner()
        click.echo(ctx.get_help())


# ── mars run ───────────────────────────────────────────────────────────────────

@mars_group.command("run")
@click.option("--preset", "-p", "run_preset",
              type=click.Choice(_presets.MARS_PRESET_NAMES), default=None,
              help="Built-in preset (may also be set on the mars group before run).")
@click.option("--config", "-c", "run_config",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="YAML config file (may also be set on the mars group before run).")
@click.option("--type", "exp_type",
              type=click.Choice([e.value for e in ExpType]), default=None,
              help="Experiment type (sol, year, multi, spots, or intervention).")
@click.option("--sols",      type=float,  default=None, help="Number of sols to simulate.")
@click.option("--accuracy",  type=click.Choice([a.value for a in Accuracy]), default=None,
              help="Integration mode: fast (reduced-order) or accurate (RK4).")
@click.option("--dt",        type=float,  default=None, help="Timestep in seconds.")
@click.option("--lat",       type=float,  default=None, help="Latitude in degrees N (-90 to 90).")
@click.option("--lon",       type=float,  default=None, help="Longitude in degrees E (0 to 360).")
@click.option("--elevation", type=float,  default=None, help="Surface elevation in metres vs Mars datum.")
@click.option("--ls",        type=float,  default=None, help="Initial Solar Longitude in degrees.")
@click.option("--ice-mass",       "ice_mass",          type=float,  default=None, help="Initial CO₂ ice mass in kg.")
@click.option("--surface-temp",   "surface_temp",      type=float,  default=None, help="Initial surface temperature in K.")
@click.option("--pressure",       type=float,          default=None, help="Initial surface pressure in Pa.")
@click.option("--albedo",         type=float,          default=None, help="Bond albedo (0–1).")
@click.option("--greenhouse-factor", "greenhouse_factor", type=float, default=None,
              help="Greenhouse enhancement factor (≥1).")
@click.option("--name",       default=None, metavar="TAG",
              help="Output subfolder tag under outputs/ (default: UTC timestamp).")
@click.option("--output-dir", "output_dir", default=None, metavar="PATH",
              help="Full custom output directory path (overrides --name).")
@click.option("--no-save",   is_flag=True, default=False, help="Skip saving CSV files.")
@click.option("--plot",      is_flag=True, default=False, help="Save matplotlib PNG plots (off by default).")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt (for scripting).")
@click.option("--years",  "n_years", type=int, default=None,
              help="Mars years to simulate (intervention mode).")
@click.option("--inject", "inject_list", multiple=True, default=(),
              metavar="COMPOUND:KG_PER_YEAR",
              help=(
                  "Inject a super-greenhouse gas at a fixed annual rate.  Repeatable.  "
                  f"Available: {', '.join(_list_compounds())}.  "
                  "Example: --inject CF4:1e9 --inject SF6:5e8"
              ))
@click.pass_obj
def mars_run(
    obj: dict,
    run_preset:        str | None,
    run_config:        str | None,
    exp_type:          str | None,
    sols:              float | None,
    accuracy:          str | None,
    dt:                float | None,
    lat:               float | None,
    lon:               float | None,
    elevation:         float | None,
    ls:                float | None,
    ice_mass:          float | None,
    surface_temp:      float | None,
    pressure:          float | None,
    albedo:            float | None,
    greenhouse_factor: float | None,
    name:              str | None,
    output_dir:        str | None,
    no_save: bool, plot: bool, yes: bool,
    n_years:           int | None,
    inject_list:       tuple,
) -> None:
    """Run a Mars climate simulation.

    An interactive wizard guides you through any parameters not already
    provided via flags or a config/preset file.  Pass -y to skip the
    confirmation prompt entirely (useful for scripting).

    \b
    Examples:
      tform mars run                              # full interactive wizard
      tform mars run --preset gale-crater
      tform mars run --lat 60 --sols 5 --accuracy accurate --dt 300
      tform mars run --preset early-mars --type year
      tform mars run --type multi --sols 2 -y
    """
    from cli import runner, output as out_mod

    _print_banner()

    # If --preset / --config were given on the run subcommand rather than the
    # parent mars group, reload the base config and update obj so the wizard
    # and the rest of the function see the right values.
    if run_preset is not None or run_config is not None:
        effective_preset = run_preset or obj.get("preset")
        effective_config = run_config or obj.get("config")
        obj["cfg"]    = config_loader.load(planet="mars",
                                           preset=effective_preset,
                                           config=effective_config)
        obj["preset"] = effective_preset
        obj["config"] = effective_config

    # Parse --inject COMPOUND:KG_PER_YEAR pairs
    inject: dict[str, float] | None = None
    if inject_list:
        inject = {}
        for item in inject_list:
            try:
                compound, kg_str = item.split(":", 1)
                inject[compound.strip()] = float(kg_str.strip())
            except ValueError:
                click.echo(_c(f"\n  ✖  Bad --inject value '{item}'. "
                              "Expected COMPOUND:KG_PER_YEAR (e.g. CF4:1e9)", "bright_red"))
                sys.exit(1)

    # Build a typed RunFlags from the raw Click values (str enums → Enum members)
    given_flags = RunFlags(
        exp_type          = ExpType(exp_type)     if exp_type   else None,
        accuracy          = Accuracy(accuracy)    if accuracy   else None,
        sols              = sols,
        dt                = dt,
        lat               = lat,
        lon               = lon,
        elevation         = elevation,
        ls                = ls,
        ice_mass          = ice_mass,
        surface_temp      = surface_temp,
        pressure          = pressure,
        albedo            = albedo,
        greenhouse_factor = greenhouse_factor,
        name              = name,
        output_dir        = output_dir,
        no_save           = no_save,
        plot              = plot,
        n_years           = n_years,
        inject            = inject,
    )

    # Always run the wizard — it skips prompts for already-provided values
    preset_name, final_flags = _wizard_mars(
        given_preset   = obj["preset"],
        given_config   = obj["config"],
        given_exp_type = given_flags.exp_type,
        given_flags    = given_flags,
    )

    # Reload base config if the wizard chose a different preset
    base: SimConfig = obj["cfg"]
    if preset_name != obj["preset"]:
        base = config_loader.load(planet="mars", preset=preset_name, config=obj["config"])

    cfg = config_loader.merge_overrides(base, final_flags)

    # Validate — Pydantic has already validated on construction, but surface
    # cross-field errors here if needed
    errors = config_loader.validate(cfg)
    if errors:
        click.echo("\n  " + _c("✖  Configuration errors:", "bright_red", bold=True))
        for err in errors:
            click.echo(f"    {_c('•', 'bright_red')} {err}")
        sys.exit(1)

    _echo_run_summary(cfg, preset_name or obj.get("preset"))

    if not yes:
        if not click.confirm(_c("  Start simulation?", "bright_white"), default=True):
            click.echo(_c("\n  Aborted.", "bright_black"))
            sys.exit(0)
        click.echo()

    # Run
    exp = cfg.experiment.type
    if exp == ExpType.sol:
        results = runner.run_sol(cfg)
    elif exp == ExpType.year:
        results = runner.run_year(cfg)
    elif exp == ExpType.multi:
        results = runner.run_multi(cfg)
    elif exp == ExpType.spots:
        results = runner.run_spots(cfg)
    elif exp == ExpType.intervention:
        results = runner.run_intervention(cfg)
    else:
        raise AssertionError(f"Unhandled experiment type: {exp}")

    out_mod.dispatch(results, cfg)

    click.echo()
    click.echo("  " + _c("✦  Simulation complete.", "bright_green", bold=True))
    click.echo(_stars("  ·  *  ·  ★  ·  *  ·  ★  ·  *  ·"))
    click.echo()


# ── Summary box ────────────────────────────────────────────────────────────────

def _echo_run_summary(cfg: SimConfig, preset: str | None) -> None:
    p, e, x, iv = cfg.planet, cfg.engine, cfg.experiment, cfg.intervention
    ns       = "N" if p.latitude >= 0 else "S"
    loc      = f"{abs(p.latitude):.2f}°{ns}  {p.longitude:.2f}°E  elev={p.elevation_m:.0f} m"
    preset_l = preset or cfg.preset.name

    is_iv = (x.type.value == "intervention")
    type_str = (f"intervention / {iv.n_years} yr" if is_iv
                else f"{x.type.value} / {x.sols:.0f} sols")

    click.echo(_divider())
    for key, val in [
        ("preset",     preset_l),
        ("type",       type_str),
        ("location",   loc),
        ("engine",     f"{e.accuracy.value}  dt={e.dt:.0f} s"),
        ("T₀",         f"{p.surface_temperature:.1f} K"),
        ("P₀",         f"{p.surface_pressure:.1f} Pa"),
        ("albedo",     f"{p.albedo:.3f}"),
        ("greenhouse", f"{p.greenhouse_factor:.3f}"),
    ]:
        click.echo(f"  {_label(f'{key:<12}')}" + _c(val, "bright_white"))

    if is_iv and iv.injection:
        click.echo(f"  {_label('inject      ')}" + _c(
            "  ".join(f"{k} {v:.2e} kg/yr" for k, v in iv.injection.items()),
            "bright_cyan"
        ))

    click.echo(_divider())


# ── mars config ────────────────────────────────────────────────────────────────

@mars_group.group("config")
def mars_config() -> None:
    """Manage Mars simulation presets and config files."""


@mars_config.command("list")
def mars_config_list() -> None:
    """List all built-in Mars presets."""
    _print_banner()
    click.echo(_divider())
    click.echo("  " + _c("✦", "bright_yellow") + _c("  Built-in Mars presets", "bright_white", bold=True))
    click.echo(_divider())
    click.echo()
    for name in _presets.MARS_PRESET_NAMES:
        desc = _presets.MARS_PRESET_DESCRIPTIONS[name]
        tags = "  ".join(_c(f"[{t}]", "bright_black") for t in _presets.MARS_PRESET_TAGS[name])
        click.echo(f"  {_c('●', 'bright_red')}  " + _c(name, "bright_white", bold=True))
        click.echo(f"     {_c(desc, 'bright_black')}")
        click.echo(f"     {tags}")
        click.echo()
    click.echo(_label("  Run with: ") + _c("tform mars --preset <name> run", "bright_white"))
    click.echo(_label("  Inspect:  ") + _c("tform mars config show <name>", "bright_white"))
    click.echo()


@mars_config.command("show")
@click.argument("preset_name", metavar="PRESET",
                type=click.Choice(_presets.MARS_PRESET_NAMES))
def mars_config_show(preset_name: str) -> None:
    """Display the fully resolved YAML for a preset.

    \b
    Examples:
      tform mars config show gale-crater
      tform mars config show early-mars
    """
    cfg = config_loader.load(planet="mars", preset=preset_name)
    click.echo()
    click.echo(_divider())
    click.echo(
        "  " + _c("●", "bright_red")
        + "  " + _c(preset_name, "bright_white", bold=True)
        + "  " + _c(_presets.MARS_PRESET_DESCRIPTIONS[preset_name], "bright_black")
    )
    click.echo(_divider())
    click.echo()
    click.echo(yaml.dump(cfg.model_dump(), default_flow_style=False,
                         sort_keys=False, allow_unicode=True))


@mars_config.command("validate")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def mars_config_validate(config_path: str) -> None:
    """Validate a YAML config file against the schema.

    \b
    Examples:
      tform mars config validate my_run.yaml
      tform mars config validate cli/configs/gale-crater.yaml
    """
    with open(config_path) as f:
        user_data = yaml.safe_load(f) or {}
    try:
        SimConfig.model_validate(user_data)
        click.echo(
            "\n  " + _c("✔", "bright_green", bold=True)
            + "  " + _c(f"{config_path} is valid.", "bright_white")
        )
    except ValidationError as exc:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        click.echo("\n  " + _c(f"✖  {len(errors)} error(s) in {config_path}:", "bright_red", bold=True))
        for err in errors:
            click.echo(f"    {_c('•', 'bright_red')} {err}")
        sys.exit(1)


# ── tform serve ────────────────────────────────────────────────────────────────

@cli.command("serve")
@click.option("--port",       default=8000,        show_default=True,
              help="Port to bind the server to.")
@click.option("--host",       default="127.0.0.1", show_default=True,
              help="Host address to bind.")
@click.option("--no-browser", is_flag=True, default=False,
              help="Don't open a browser tab automatically.")
@click.option("--dev",        is_flag=True, default=False,
              help="Also start the Vite dev server from ui/ (port 5173).")
def serve_cmd(port: int, host: str, no_browser: bool, dev: bool) -> None:
    """Start the tform visualisation web server.

    \b
    Launches a FastAPI server that:
      • Accepts simulation run requests from the browser UI
      • Streams per-step physics data back via Server-Sent Events
      • Serves the pre-built React app from cli/static/

    \b
    Examples:
      tform serve
      tform serve --port 9000
      tform serve --dev          # also starts Vite dev server on :5173
      tform serve --no-browser
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(_c("\n  ✖  uvicorn is not installed.", "bright_red"))
        click.echo(_c("     Run: uv sync", "bright_black"))
        sys.exit(1)

    import pathlib
    import subprocess

    static_dir = pathlib.Path(__file__).parent / "static"
    ui_dir     = pathlib.Path(__file__).parent.parent / "ui"

    # Auto-build the UI on first install if cli/static/index.html is missing
    if not (static_dir / "index.html").exists():
        if not ui_dir.exists():
            click.echo(_c("\n  ✖  cli/static/index.html not found and ui/ source is missing.", "bright_red"))
            click.echo(_c("     Re-install tform or run: make ui-build", "bright_black"))
            sys.exit(1)

        click.echo()
        click.echo(_divider())
        click.echo("  " + _c("⚙", "bright_yellow") + _c("  First run — building UI", "bright_white", bold=True))
        click.echo(_divider())

        npm = "npm"
        for step, args in [("npm install", [npm, "install"]),
                           ("npm run build", [npm, "run", "build"])]:
            click.echo(f"  {_label('running     ')}{_value(step)}")
            result = subprocess.run(args, cwd=str(ui_dir), capture_output=True, text=True)
            if result.returncode != 0:
                click.echo(_c(f"\n  ✖  {step} failed:", "bright_red"))
                click.echo(result.stderr[-2000:])
                sys.exit(1)

        click.echo("  " + _c("✔", "bright_green") + _c("  UI built successfully.", "bright_white"))
        click.echo()

    url = f"http://{host}:{port}"

    click.echo()
    click.echo(_divider())
    click.echo("  " + _planet("tform") + _c("  serve", "bright_white", bold=True))
    click.echo(f"  {_label('server      ')}{_value(url)}")
    click.echo(f"  {_label('api docs    ')}{_value(url + '/api/docs')}")
    if dev:
        click.echo(f"  {_label('vite dev    ')}{_value('http://localhost:5173')}")
    click.echo(_divider())
    click.echo()

    dev_proc = None
    if dev:
        if ui_dir.exists():
            dev_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_dir),
            )
        else:
            click.echo(_c("  ⚠  ui/ directory not found — skipping Vite dev server.", "bright_yellow"))

    if not no_browser:
        import threading
        import webbrowser
        open_url = "http://localhost:5173" if dev else url
        threading.Timer(1.5, lambda: webbrowser.open(open_url)).start()

    try:
        uvicorn.run("cli.server:app", host=host, port=port, reload=False)
    finally:
        if dev_proc is not None:
            dev_proc.terminate()





# ── tform benchmark ───────────────────────────────────────────────────────────

@cli.command("benchmark")
@click.option(
    "--sols",
    "sol_counts",
    multiple=True,
    type=click.IntRange(min=1),
    help=(
        "Sol count to benchmark. Repeat for multiple values. "
        "Defaults to 7, 30, 180, 600, 5000, and 10000."
    ),
)
@click.option(
    "--accuracy",
    type=click.Choice([a.value for a in Accuracy]),
    default=Accuracy.fast.value,
    show_default=True,
)
def benchmark_cmd(
    sol_counts: tuple[int, ...],
    accuracy: str,
) -> None:
    """Run repeatable Mars simulation benchmarks."""
    from cli.benchmark import run_benchmarks

    run_benchmarks(
        sol_counts=list(sol_counts) or None,
        accuracy=Accuracy(accuracy),
    )







if __name__ == "__main__":
    cli()
