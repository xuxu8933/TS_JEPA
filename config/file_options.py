"""Load script options from a shared JSON or TOML experiment file."""

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    tomllib = None


CONFIG_SECTIONS = ("common", "runner", "analysis")


def _read_config_file(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open() as config_file:
            data = json.load(config_file)
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML configs require Python 3.11 or newer")
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    else:
        raise ValueError(
            f"Unsupported experiment config format {suffix!r}; use .json or .toml"
        )

    if not isinstance(data, dict):
        raise ValueError(f"Experiment config root must be an object: {config_path}")
    return config_path, data


def _section_options(data, section, config_path):
    uses_sections = any(name in data for name in CONFIG_SECTIONS)
    if not uses_sections:
        return dict(data)

    unknown_sections = sorted(set(data) - set(CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError(
            f"Unknown top-level config sections in {config_path}: "
            + ", ".join(unknown_sections)
        )

    common = data.get("common", {})
    script_options = data.get(section, {})
    if not isinstance(common, dict):
        raise ValueError(f"[common] must be an object in {config_path}")
    if not isinstance(script_options, dict):
        raise ValueError(f"[{section}] must be an object in {config_path}")
    return {**common, **script_options}


def _actions_by_destination(parser):
    actions = {}
    for action in parser._actions:
        actions.setdefault(action.dest, []).append(action)
    return actions


def _coerce_scalar(value, action, key, config_path):
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(
            f"Config option {key!r} must be a scalar in {config_path}"
        )
    if action.type is None:
        converted = value
    else:
        try:
            converted = action.type(value if isinstance(value, str) else str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for config option {key!r} in {config_path}: {value!r}"
            ) from exc
    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(map(str, action.choices))
        raise ValueError(
            f"Invalid value for config option {key!r} in {config_path}: "
            f"{converted!r}; choose from {choices}"
        )
    return converted


def _coerce_option(value, actions, key, config_path):
    if value is None:
        return None

    boolean_actions = (
        argparse._StoreTrueAction,
        argparse._StoreFalseAction,
        argparse.BooleanOptionalAction,
    )
    if any(isinstance(action, boolean_actions) for action in actions):
        if not isinstance(value, bool):
            raise ValueError(
                f"Config option {key!r} must be true or false in {config_path}"
            )
        return value

    action = actions[0]
    expects_sequence = action.nargs in ("+", "*") or isinstance(action.nargs, int)
    if expects_sequence:
        if not isinstance(value, list):
            raise ValueError(
                f"Config option {key!r} must be a list in {config_path}"
            )
        if action.nargs == "+" and not value:
            raise ValueError(
                f"Config option {key!r} cannot be empty in {config_path}"
            )
        return [
            _coerce_scalar(item, action, key, config_path)
            for item in value
        ]

    return _coerce_scalar(value, action, key, config_path)


def _explicit_actions(parser, argv):
    option_actions = parser._option_string_actions
    explicit = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            explicit.add(action)
    return explicit


def _drop_overridden_group_defaults(parser, defaults, argv):
    explicit = _explicit_actions(parser, argv)
    for group in parser._mutually_exclusive_groups:
        selected = [action for action in group._group_actions if action in explicit]
        if not selected:
            continue
        selected_destinations = {action.dest for action in selected}
        for action in group._group_actions:
            if action.dest not in selected_destinations:
                defaults.pop(action.dest, None)


def load_config_defaults(parser, config_path, section, argv):
    resolved_path, data = _read_config_file(config_path)
    raw_options = _section_options(data, section, resolved_path)
    actions = _actions_by_destination(parser)
    valid_destinations = set(actions) - {"help", "config"}
    unknown_options = sorted(set(raw_options) - valid_destinations)
    if unknown_options:
        raise ValueError(
            f"Unknown [{section}] options in {resolved_path}: "
            + ", ".join(unknown_options)
        )

    defaults = {
        key: _coerce_option(value, actions[key], key, resolved_path)
        for key, value in raw_options.items()
    }
    _drop_overridden_group_defaults(parser, defaults, argv)
    return defaults


def parse_args_with_config(parser, argv=None, section="runner"):
    """Apply file defaults before parsing, so explicit CLI values win."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_args, _ = config_probe.parse_known_args(argv)

    if config_args.config is not None:
        defaults = load_config_defaults(
            parser,
            config_args.config,
            section,
            argv,
        )
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)
