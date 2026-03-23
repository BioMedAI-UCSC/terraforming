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
  tform mars run --config my_run.yaml --lat 45 --no-plot
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

from cli import presets as _presets
from cli import config_loader

VERSION = "0.1.0"

# ── Space-themed palette helpers ──────────────────────────────────────────────

def _c(text: str, fg: str, bold: bool = False, dim: bool = False) -> str:
    return click.style(text, fg=fg, bold=bold, dim=dim)


def _stars(text: str) -> str:
    return _c(text, "yellow", dim=True)


def _planet(text: str) -> str:
    return _c(text, "bright_red", bold=True)


def _label(text: str) -> str:
    return _c(text, "bright_black")


def _value(text: str) -> str:
    return _c(text, "bright_white")


def _divider(width: int = 62) -> str:
    return _c("─" * width, "bright_red", dim=True)


# ── ASCII banner ──────────────────────────────────────────────────────────────

_MARS_ART = [
    ("  ★  ·  *  ·   ·  *  ·  ★  ·   ·  *  ·  ★  ·  *  ", "yellow"      ),
    ("                                                    ", None         ),
    ("           .-~~~~~~~~~~~~~~~~~~~-.                  ", "bright_red"  ),
    ("         /  ░▒▓▓████████████▓▒░  \\                ", "bright_red"  ),
    ("        |  ░▓▒█████████████████░▓  |               ", "bright_red"  ),
    ("        |  ▓░░▒████████████████▒░  |               ", "bright_red"  ),
    ("        |  ░▓▒█████████████████░▓  |               ", "bright_red"  ),
    ("         \\  ▓░░▒████████████░▒░  /                ", "bright_red"  ),
    ("           `~~~~~~~~~~~~~~~~~~~`                    ", "bright_red"  ),
    ("                                                    ", None         ),
    ("  ★  ·  *  ·   ·  *  ·  ★  ·   ·  *  ·  ★  ·  *  ", "yellow"      ),
]


def _print_banner() -> None:
    click.echo()
    for line, color in _MARS_ART:
        click.echo(_c(line, color) if color else line)
    click.echo()
    click.echo(
        "  "
        + _planet("tform")
        + _c("  ·  Terraforming environments, starting with Mars ·  ", "bright_white")
        + _c(f"v{VERSION}", "bright_black")
    )
    click.echo()


# ── Man pages ─────────────────────────────────────────────────────────────────

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
    sol     Simulate N sols at a fixed location  (use --sols N)
    year    Simulate one full Martian year (668 sols)
    multi   Simulate N sols at 45°N, Equator, -40°S simultaneously

OUTPUT OPTIONS
    --no-save    Skip CSV output files
    --no-plot    Skip matplotlib plots
    --name TAG   Name tag for output folder (default: UTC timestamp)

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
      save_plot: true
      out_dir: null
"""

_MAN_PAGES = {"mars": _MAN_MARS}


# ── Interactive wizard ────────────────────────────────────────────────────────

def _wizard_mars(
    given_preset:   str | None,
    given_config:   str | None,
    given_exp_type: str | None,
    given_flags:    dict,
) -> tuple[str | None, dict]:
    """Interactive setup wizard for a Mars simulation.

    Only prompts for values not already provided via flags/preset/config.
    Returns (resolved_preset_name, additional_overrides_dict).

    Parameters
    ----------
    given_preset   : --preset value already provided (or None)
    given_config   : --config value already provided (or None)
    given_exp_type : --type value already provided (or None)
    given_flags    : dict of all other CLI flag values (None = not provided)
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

    overrides: dict = {}

    # ── Experiment type (skip if --type already given) ────────────────────────
    if given_exp_type is not None:
        exp_type = given_exp_type
        click.echo(
            _label("  type     ")
            + _c(exp_type, "bright_white")
            + _c("  (from --type flag)", "bright_black")
        )
    else:
        click.echo(_c("  Experiment type:", "bright_white", bold=True))
        type_choices = [
            ("sol",   "N sols at a fixed location"),
            ("year",  "Full Martian year (668 sols)"),
            ("multi", "3 latitudes simultaneously  (45°N / equator / 40°S)"),
        ]
        for i, (key, desc) in enumerate(type_choices, 1):
            click.echo(
                f"    {_c(str(i), 'bright_yellow', bold=True)}  "
                + _c(f"{key:<6}", "bright_white")
                + _c(f"  {desc}", "bright_black")
            )
        type_idx = click.prompt(
            click.style("\n  Select type", fg="bright_white"),
            type=click.IntRange(1, 3), default=1, show_default=True,
        )
        exp_type = type_choices[type_idx - 1][0]
    overrides["exp_type"] = exp_type

    # ── Preset (skip if --preset or --config already given) ───────────────────
    preset_name = given_preset
    if given_preset is not None:
        click.echo(
            _label("  preset   ")
            + _c(given_preset, "bright_white")
            + _c("  (from --preset flag)", "bright_black")
        )
    elif given_config is not None:
        click.echo(
            _label("  config   ")
            + _c(given_config, "bright_white")
            + _c("  (from --config flag)", "bright_black")
        )
    else:
        click.echo()
        use_preset = click.confirm(
            _c("  Use a built-in preset?", "bright_white"), default=True
        )
        if use_preset:
            click.echo()
            click.echo(_c("  Built-in presets:", "bright_white", bold=True))
            for i, name in enumerate(_presets.MARS_PRESET_NAMES, 1):
                desc = _presets.MARS_PRESET_DESCRIPTIONS[name]
                click.echo(
                    f"    {_c(str(i), 'bright_yellow', bold=True)}  "
                    + _c(f"{name:<26}", "bright_white")
                    + _c(desc, "bright_black")
                )
            preset_idx = click.prompt(
                click.style("\n  Select preset", fg="bright_white"),
                type=click.IntRange(1, len(_presets.MARS_PRESET_NAMES)),
                default=1, show_default=True,
            )
            preset_name = _presets.MARS_PRESET_NAMES[preset_idx - 1]
            click.echo(
                "  " + _c("✓", "bright_green")
                + _c(f"  Preset: {preset_name}", "bright_white")
            )

    # Load base config from the resolved preset so prompts show real defaults
    base_cfg = config_loader.load(planet="mars", preset=preset_name,
                                  config=given_config)
    p_def = base_cfg["planet"]
    e_def = base_cfg["engine"]
    x_def = base_cfg["experiment"]

    # ── Parameter overrides (skip each if that flag was already given) ────────
    click.echo()
    click.echo(
        _c("  Parameters", "bright_white", bold=True)
        + _c("  (Enter = keep default, shown in brackets):", "bright_black")
    )
    click.echo()

    def _ask_float(label: str, flag_key: str, cfg_default: float,
                   unit: str = "") -> None:
        if given_flags.get(flag_key) is not None:
            click.echo(
                f"    {_label(f'{label:<22}')}"
                + _c(f"{given_flags[flag_key]}", "bright_white")
                + _c(f"  (from --{flag_key.replace('_','-')} flag)", "bright_black")
            )
            return
        hint = _c(f"  ({unit})" if unit else "", "bright_black")
        val  = click.prompt(
            f"    {_c(label, 'bright_white')}{hint}",
            default=cfg_default, type=float, show_default=True,
        )
        if val != cfg_default:
            overrides[flag_key] = val

    def _ask_accuracy() -> None:
        if given_flags.get("accuracy") is not None:
            click.echo(
                f"    {_label('accuracy              ')}"
                + _c(given_flags["accuracy"], "bright_white")
                + _c("  (from --accuracy flag)", "bright_black")
            )
            return
        default = e_def["accuracy"]
        click.echo(
            f"    {_c('Integration accuracy  ', 'bright_white')}"
            + _c("(fast / accurate)", "bright_black")
            + f"  [{_c(default, 'bright_yellow')}]: ",
            nl=False,
        )
        raw = input().strip()
        if raw in ("fast", "accurate") and raw != default:
            overrides["accuracy"] = raw

    if exp_type == "sol":
        _ask_float("Sols to simulate",    "sols",      x_def["sols"],           "sols")
    _ask_accuracy()
    if exp_type != "year":
        _ask_float("Latitude",            "lat",       p_def["latitude"],       "°N")
        _ask_float("Longitude",           "lon",       p_def["longitude"],      "°E")
        _ask_float("Elevation",           "elevation", p_def["elevation_m"],    "m")
    _ask_float("Solar longitude (Ls)",    "ls",        p_def["initial_ls_deg"], "°")

    click.echo()
    return preset_name, overrides


# ── Root group ────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.version_option(VERSION, "--version", "-V",
                      message=_c("tform", "bright_red", bold=True)
                              + " %(version)s")
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


# ── man command ───────────────────────────────────────────────────────────────

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


# ── mars group ────────────────────────────────────────────────────────────────

@cli.group("mars", invoke_without_command=True)
@click.option("--preset", "-p",
              type=click.Choice(_presets.MARS_PRESET_NAMES),
              default=None,
              help="Use a built-in preset as the base configuration.")
@click.option("--config", "-c",
              type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Path to a YAML config file (merged over preset).")
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


# ── mars run ──────────────────────────────────────────────────────────────────

@mars_group.command("run")
@click.option("--type", "exp_type",
              type=click.Choice(["sol", "year", "multi"]), default=None,
              help="Experiment type (sol, year, or multi-coordinate).")
@click.option("--sols",      type=float, default=None,
              help="Number of sols to simulate.")
@click.option("--accuracy",  type=click.Choice(["fast", "accurate"]), default=None,
              help="Integration mode: fast (reduced-order) or accurate (RK4).")
@click.option("--dt",        type=float, default=None,
              help="Timestep in seconds.")
@click.option("--lat",       type=float, default=None,
              help="Latitude in degrees N (-90 to 90).")
@click.option("--lon",       type=float, default=None,
              help="Longitude in degrees E (0 to 360).")
@click.option("--elevation", type=float, default=None,
              help="Surface elevation in metres vs Mars datum.")
@click.option("--ls",        type=float, default=None,
              help="Initial Solar Longitude in degrees.")
@click.option("--ice-mass",  "ice_mass", type=float, default=None,
              help="Initial CO₂ ice mass in kg.")
@click.option("--surface-temp", "surface_temp", type=float, default=None,
              help="Initial surface temperature in K.")
@click.option("--pressure",  type=float, default=None,
              help="Initial surface pressure in Pa.")
@click.option("--albedo",    type=float, default=None,
              help="Bond albedo (0–1).")
@click.option("--greenhouse-factor", "greenhouse_factor", type=float, default=None,
              help="Greenhouse enhancement factor (≥1).")
@click.option("--name",       default=None, metavar="TAG",
              help="Output subfolder tag under outputs/ (default: UTC timestamp).")
@click.option("--output-dir", "output_dir", default=None, metavar="PATH",
              help="Full custom output directory path (overrides --name).")
@click.option("--no-save",   is_flag=True, default=False,
              help="Skip saving CSV files.")
@click.option("--no-plot",   is_flag=True, default=False,
              help="Skip saving plots.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt (for scripting).")
@click.pass_obj
def mars_run(
    obj: dict,
    exp_type, sols, accuracy, dt, lat, lon, elevation, ls,
    ice_mass, surface_temp, pressure, albedo, greenhouse_factor,
    name, output_dir, no_save, no_plot, yes,
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

    # ── Always run the wizard (skips prompts for values already given) ────────
    given_flags = {
        "accuracy":          accuracy,
        "dt":                dt,
        "lat":               lat,
        "lon":               lon,
        "elevation":         elevation,
        "ls":                ls,
        "sols":              sols,
        "ice_mass":          ice_mass,
        "surface_temp":      surface_temp,
        "pressure":          pressure,
        "albedo":            albedo,
        "greenhouse_factor": greenhouse_factor,
    }
    wizard_preset, wizard_overrides = _wizard_mars(
        given_preset=obj["preset"],
        given_config=obj["config"],
        given_exp_type=exp_type,
        given_flags=given_flags,
    )

    # If the wizard chose a different preset, reload the base config
    if wizard_preset != obj["preset"]:
        obj["cfg"] = config_loader.load(planet="mars", preset=wizard_preset,
                                        config=obj["config"])

    # ── Merge all overrides (CLI flags take priority over wizard choices) ─────
    all_overrides = {
        "exp_type":          exp_type or wizard_overrides.get("exp_type"),
        "sols":              sols     or wizard_overrides.get("sols"),
        "accuracy":          accuracy or wizard_overrides.get("accuracy"),
        "dt":                dt       or wizard_overrides.get("dt"),
        "lat":               lat      or wizard_overrides.get("lat"),
        "lon":               lon      or wizard_overrides.get("lon"),
        "elevation":         elevation or wizard_overrides.get("elevation"),
        "ls":                ls       or wizard_overrides.get("ls"),
        "ice_mass":          ice_mass,
        "surface_temp":      surface_temp,
        "pressure":          pressure,
        "albedo":            albedo,
        "greenhouse_factor": greenhouse_factor,
        "name":              name,
        "output_dir":        output_dir,
        "no_save":           no_save,
        "no_plot":           no_plot,
    }
    cfg = config_loader.merge_overrides(obj["cfg"], all_overrides)

    # ── Validate ──────────────────────────────────────────────────────────────
    errors = config_loader.validate(cfg)
    if errors:
        click.echo(
            "\n  " + _c("✖  Configuration errors:", "bright_red", bold=True)
        )
        for err in errors:
            click.echo(f"    {_c('•', 'bright_red')} {err}")
        sys.exit(1)

    # ── Confirm ───────────────────────────────────────────────────────────────
    _echo_run_summary(cfg, wizard_preset or obj.get("preset"))

    if not yes:
        if not click.confirm(
            _c("  Start simulation?", "bright_white"), default=True
        ):
            click.echo(_c("\n  Aborted.", "bright_black"))
            sys.exit(0)
        click.echo()

    # ── Run ───────────────────────────────────────────────────────────────────
    exp = cfg["experiment"]["type"]

    if exp == "sol":
        results = runner.run_sol(cfg)
    elif exp == "year":
        results = runner.run_year(cfg)
    elif exp == "multi":
        results = runner.run_multi(cfg)

    out_mod.dispatch(results, cfg, exp)

    click.echo()
    click.echo(
        "  " + _c("✦  Simulation complete.", "bright_green", bold=True)
    )
    click.echo(_stars("  ·  *  ·  ★  ·  *  ·  ★  ·  *  ·"))
    click.echo()


# ── Summary box ───────────────────────────────────────────────────────────────

def _echo_run_summary(cfg: dict, preset: str | None) -> None:
    p, e, x  = cfg["planet"], cfg["engine"], cfg["experiment"]
    ns       = "N" if p["latitude"] >= 0 else "S"
    loc      = (f"{abs(p['latitude']):.2f}°{ns}  "
                f"{p['longitude']:.2f}°E  "
                f"elev={p['elevation_m']:.0f} m")
    preset_l = preset or cfg.get("preset", {}).get("name", "custom")

    click.echo(_divider())
    rows = [
        ("preset",   preset_l),
        ("type",     f"{x['type']} / {x.get('sols', 668):.0f} sols"),
        ("location", loc),
        ("engine",   f"{e['accuracy']}  dt={e['dt']:.0f} s"),
        ("T₀",       f"{p['surface_temperature']:.1f} K"),
        ("P₀",       f"{p['surface_pressure']:.1f} Pa"),
        ("albedo",   f"{p['albedo']:.3f}"),
        ("greenhouse", f"{p['greenhouse_factor']:.3f}"),
    ]
    for key, val in rows:
        click.echo(
            f"  {_label(f'{key:<12}')}"
            + _c(val, "bright_white")
        )
    click.echo(_divider())


# ── mars config ───────────────────────────────────────────────────────────────

@mars_group.group("config")
def mars_config() -> None:
    """Manage Mars simulation presets and config files."""


@mars_config.command("list")
def mars_config_list() -> None:
    """List all built-in Mars presets."""
    _print_banner()
    click.echo(_divider())
    click.echo(
        "  " + _c("✦", "bright_yellow")
        + _c("  Built-in Mars presets", "bright_white", bold=True)
    )
    click.echo(_divider())
    click.echo()
    for name in _presets.MARS_PRESET_NAMES:
        desc = _presets.MARS_PRESET_DESCRIPTIONS[name]
        tags = "  ".join(_c(f"[{t}]", "bright_black") for t in _presets.MARS_PRESET_TAGS[name])
        click.echo(
            f"  {_c('●', 'bright_red')}  "
            + _c(f"{name}", "bright_white", bold=True)
        )
        click.echo(f"     {_c(desc, 'bright_black')}")
        click.echo(f"     {tags}")
        click.echo()
    click.echo(
        _label("  Run with: ")
        + _c("tform mars --preset <name> run", "bright_white")
    )
    click.echo(
        _label("  Inspect:  ")
        + _c("tform mars config show <name>", "bright_white")
    )
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
    click.echo(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))


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
    cfg    = config_loader._deep_merge(config_loader._MARS_BASE, user_data)
    errors = config_loader.validate(cfg)
    if errors:
        click.echo(
            "\n  " + _c(f"✖  {len(errors)} error(s) in {config_path}:", "bright_red", bold=True)
        )
        for err in errors:
            click.echo(f"    {_c('•', 'bright_red')} {err}")
        sys.exit(1)
    click.echo(
        "\n  " + _c("✔", "bright_green", bold=True)
        + "  " + _c(f"{config_path} is valid.", "bright_white")
    )


if __name__ == "__main__":
    cli()
