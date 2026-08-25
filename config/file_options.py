"""Load script options from a shared JSON, JSONC, or TOML experiment file."""

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    tomllib = None


CONFIG_SECTIONS = ("common", "runner", "analysis")
COMMON_ONLY_OPTIONS = frozenset(("stocks", "seeds"))
DERIVED_CONFIG_OPTIONS = frozenset(("results_dir",))
RUNNER_GROUPS = frozenset(
    (
        "execution",
        "download",
        "masking",
        "objectives",
        "pretraining",
        "preprocessing",
        "checkpoint",
        "downstream",
        "output",
    )
)
MASK_STRATEGY_FIELDS = {
    "random": {},
    "local_long": {
        "mae_window_patches": "mae_window_patches",
        "jepa_gap_patches": "jepa_gap_patches",
        "jepa_target_patches": "jepa_target_patches",
    },
    "future_block": {"target_patches": "future_target_patches"},
    "causal_multiblock": {
        "num_blocks": "causal_num_blocks",
        "block_patches": "causal_block_patches",
        "block_gap_patches": "causal_block_gap_patches",
    },
}


def results_dir_from_config(config_path):
    """Derive ``results/<config_stem>`` without a configured output option."""
    path = Path(config_path)
    resolved_path = path.resolve()
    if (
        resolved_path.parent.name == "experiments"
        and resolved_path.parent.parent.name == "config"
    ):
        project_root = resolved_path.parent.parent.parent
    else:
        project_root = resolved_path.parent
    return project_root / "results" / path.stem


def _strip_json_line_comments(text):
    """Replace ``//`` comments outside JSON strings while preserving lines."""
    output = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if (
            character == "/"
            and index + 1 < len(text)
            and text[index + 1] == "/"
        ):
            while index < len(text) and text[index] not in "\r\n":
                output.append(" ")
                index += 1
            continue
        output.append(character)
        index += 1
    return "".join(output)


def read_config_file(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".json", ".jsonc"}:
        text = config_path.read_text(encoding="utf-8")
        data = json.loads(_strip_json_line_comments(text))
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML configs require Python 3.11 or newer")
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    else:
        raise ValueError(
            "Unsupported experiment config format "
            f"{suffix!r}; use .json, .jsonc, or .toml"
        )

    if not isinstance(data, dict):
        raise ValueError(f"Experiment config root must be an object: {config_path}")
    return config_path, data


def _config_object(value, path, config_path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object in {config_path}")
    return value


def _validate_keys(value, allowed, path, config_path, *, required=()):
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown options in {path} in {config_path}: " + ", ".join(unknown)
        )
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(
            f"Missing required options in {path} in {config_path}: "
            + ", ".join(missing)
        )


def _copy_mapped_options(flattened, source, mappings):
    for source_name, destination in mappings.items():
        if source_name in source:
            flattened[destination] = source[source_name]


def flatten_runner_options(runner, config_path):
    """Translate the nested runner schema into existing runner destinations."""
    runner = _config_object(runner, "[runner]", config_path)
    nested_groups = set(runner) & RUNNER_GROUPS
    if not nested_groups:
        return dict(runner)

    flat_options = sorted(set(runner) - RUNNER_GROUPS)
    if flat_options:
        raise ValueError(
            "Nested [runner] groups cannot be mixed with flat runner options in "
            f"{config_path}: " + ", ".join(flat_options)
        )

    flattened = {}

    if "execution" in runner:
        execution = _config_object(
            runner["execution"], "[runner].execution", config_path
        )
        mappings = {
            "max_stocks": "max_stocks",
            "max_seeds": "max_seeds",
            "max_parallel_jobs": "max_parallel_jobs",
            "dry_run": "dry_run",
            "verbose": "verbose",
        }
        _validate_keys(execution, mappings, "[runner].execution", config_path)
        _copy_mapped_options(flattened, execution, mappings)

    if "download" in runner:
        download = _config_object(
            runner["download"], "[runner].download", config_path
        )
        download_keys = {"skip", "start_date", "end_date", "write_mode", "news"}
        _validate_keys(
            download,
            download_keys,
            "[runner].download",
            config_path,
            required=("skip",),
        )
        skip_download = download["skip"]
        flattened["skip_download"] = skip_download
        if skip_download is False:
            _copy_mapped_options(
                flattened,
                download,
                {
                    "start_date": "download_start_date",
                    "end_date": "download_end_date",
                    "write_mode": "write_mode",
                },
            )
        if "news" in download:
            news = _config_object(
                download["news"], "[runner].download.news", config_path
            )
            news_mappings = {
                "max_articles": "max_news_articles",
                "chunk_days": "news_chunk_days",
                "request_delay": "request_delay",
            }
            _validate_keys(
                news,
                {"skip", *news_mappings},
                "[runner].download.news",
                config_path,
                required=("skip",),
            )
            skip_news = news["skip"]
            if skip_download is False:
                flattened["skip_news"] = skip_news
                if skip_news is False:
                    _copy_mapped_options(flattened, news, news_mappings)

    if "masking" in runner:
        masking = _config_object(
            runner["masking"], "[runner].masking", config_path
        )
        _validate_keys(
            masking,
            {"strategies"},
            "[runner].masking",
            config_path,
            required=("strategies",),
        )
        strategies = _config_object(
            masking["strategies"],
            "[runner].masking.strategies",
            config_path,
        )
        if not strategies:
            raise ValueError(
                f"[runner].masking.strategies cannot be empty in {config_path}"
            )
        unknown_strategies = sorted(set(strategies) - set(MASK_STRATEGY_FIELDS))
        if unknown_strategies:
            raise ValueError(
                "Unknown strategies in [runner].masking.strategies in "
                f"{config_path}: " + ", ".join(unknown_strategies)
            )
        selected_strategies = []
        for strategy, options in strategies.items():
            path = f"[runner].masking.strategies.{strategy}"
            options = _config_object(options, path, config_path)
            mappings = MASK_STRATEGY_FIELDS[strategy]
            _validate_keys(
                options,
                {"enabled", *mappings},
                path,
                config_path,
                required={"enabled", *mappings},
            )
            enabled = options["enabled"]
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"{path}.enabled must be true or false in {config_path}"
                )
            if enabled:
                selected_strategies.append(strategy)
                _copy_mapped_options(flattened, options, mappings)
        if not selected_strategies:
            raise ValueError(
                f"At least one masking strategy must be enabled in {config_path}"
            )
        flattened["mask_strategies"] = selected_strategies

    if "objectives" in runner:
        objectives = _config_object(
            runner["objectives"], "[runner].objectives", config_path
        )
        _validate_keys(
            objectives,
            {"jepa", "mae"},
            "[runner].objectives",
            config_path,
            required=("jepa", "mae"),
        )
        for objective, prefix in (("jepa", "jepa"), ("mae", "mae")):
            path = f"[runner].objectives.{objective}"
            settings = _config_object(objectives[objective], path, config_path)
            mappings = {"weight": f"lambda_{prefix}", "loss": f"{prefix}_loss"}
            _validate_keys(
                settings,
                mappings,
                path,
                config_path,
                required=mappings,
            )
            _copy_mapped_options(flattened, settings, mappings)

    if "pretraining" in runner:
        pretraining = _config_object(
            runner["pretraining"], "[runner].pretraining", config_path
        )
        _validate_keys(
            pretraining,
            {"skip", "epochs", "windows"},
            "[runner].pretraining",
            config_path,
            required=("skip", "epochs", "windows"),
        )
        flattened["skip_pretrain"] = pretraining["skip"]
        flattened["pretrain_num_epochs"] = pretraining["epochs"]
        windows = _config_object(
            pretraining["windows"], "[runner].pretraining.windows", config_path
        )
        window_mappings = {
            "series_size": "series_split_size",
            "patch_size": "patch_size",
            "stride": "pretrain_stride",
            "sampling_mode": "sampling_mode",
        }
        _validate_keys(
            windows,
            window_mappings,
            "[runner].pretraining.windows",
            config_path,
            required=window_mappings,
        )
        _copy_mapped_options(flattened, windows, window_mappings)

    if "preprocessing" in runner:
        preprocessing = _config_object(
            runner["preprocessing"], "[runner].preprocessing", config_path
        )
        _validate_keys(
            preprocessing,
            {"preset", "custom"},
            "[runner].preprocessing",
            config_path,
        )
        preset = preprocessing.get("preset")
        custom = preprocessing.get("custom")
        if preset is not None and "custom" in preprocessing:
            raise ValueError(
                "[runner].preprocessing.preset and custom are mutually exclusive "
                f"in {config_path}"
            )
        if preset is None and custom is None:
            raise ValueError(
                "[runner].preprocessing requires a non-null preset or custom "
                f"settings in {config_path}"
            )
        if preset is not None:
            flattened["preprocessing_preset"] = preset
        else:
            flattened["preprocessing_preset"] = None
            custom_path = "[runner].preprocessing.custom"
            custom = _config_object(custom, custom_path, config_path)
            _validate_keys(
                custom,
                {"feature_transform", "normalization", "features", "forecast"},
                custom_path,
                config_path,
                required=("feature_transform", "normalization", "features", "forecast"),
            )
            flattened["feature_transform"] = custom["feature_transform"]

            normalization_path = f"{custom_path}.normalization"
            normalization = _config_object(
                custom["normalization"], normalization_path, config_path
            )
            _validate_keys(
                normalization,
                {"method", "robust_zscore"},
                normalization_path,
                config_path,
                required=("method",),
            )
            method = normalization["method"]
            flattened["normalization"] = method
            if "robust_zscore" in normalization:
                robust_path = f"{normalization_path}.robust_zscore"
                robust = _config_object(
                    normalization["robust_zscore"], robust_path, config_path
                )
                _validate_keys(
                    robust,
                    {"clip"},
                    robust_path,
                    config_path,
                    required=("clip",),
                )
                if method == "train_robust_zscore":
                    flattened["robust_zscore_clip"] = robust["clip"]

            features_path = f"{custom_path}.features"
            features = _config_object(custom["features"], features_path, config_path)
            _validate_keys(
                features,
                {"market", "sentiment"},
                features_path,
                config_path,
                required=("market", "sentiment"),
            )
            if not isinstance(features["market"], list) or not features["market"]:
                raise ValueError(
                    f"{features_path}.market must be a non-empty list in "
                    f"{config_path}"
                )
            flattened["market_features"] = features["market"]
            sentiment_path = f"{features_path}.sentiment"
            sentiment = _config_object(
                features["sentiment"], sentiment_path, config_path
            )
            _validate_keys(
                sentiment,
                {"enabled", "columns"},
                sentiment_path,
                config_path,
                required=("enabled",),
            )
            sentiment_enabled = sentiment["enabled"]
            flattened["use_sentiment"] = sentiment_enabled
            if "columns" not in sentiment:
                raise ValueError(
                    f"{sentiment_path}.columns is required "
                    f"in {config_path}"
                )
            if (
                not isinstance(sentiment["columns"], list)
                or not sentiment["columns"]
            ):
                raise ValueError(
                    f"{sentiment_path}.columns must be a non-empty list in "
                    f"{config_path}"
                )
            if sentiment_enabled:
                flattened["sentiment_features"] = sentiment["columns"]

            forecast_path = f"{custom_path}.forecast"
            forecast = _config_object(custom["forecast"], forecast_path, config_path)
            _validate_keys(
                forecast,
                {"target", "market_data"},
                forecast_path,
                config_path,
                required=("target", "market_data"),
            )
            target = forecast["target"]
            flattened["forecast_target"] = target
            market_data_path = f"{forecast_path}.market_data"
            market_data = _config_object(
                forecast["market_data"], market_data_path, config_path
            )
            _validate_keys(
                market_data,
                {"enabled", "name"},
                market_data_path,
                config_path,
                required=("enabled", "name"),
            )
            market_data_enabled = market_data["enabled"]
            if not isinstance(market_data_enabled, bool):
                raise ValueError(
                    f"{market_data_path}.enabled must be true or false in "
                    f"{config_path}"
                )
            if not market_data["name"]:
                raise ValueError(
                    f"{market_data_path}.name must be non-empty in {config_path}"
                )
            if target == "excess_log_return" and not market_data_enabled:
                raise ValueError(
                    f"{market_data_path} must be enabled for "
                    f"excess_log_return in {config_path}"
                )
            if target != "excess_log_return" and market_data_enabled:
                raise ValueError(
                    f"{market_data_path} can only be enabled for "
                    f"excess_log_return in {config_path}"
                )
            if market_data_enabled:
                flattened["market_data"] = market_data["name"]

    if "checkpoint" in runner:
        checkpoint = _config_object(
            runner["checkpoint"], "[runner].checkpoint", config_path
        )
        _validate_keys(
            checkpoint,
            {"selection", "encoder_weights"},
            "[runner].checkpoint",
            config_path,
            required=("selection", "encoder_weights"),
        )
        flattened["encoder_weights"] = checkpoint["encoder_weights"]
        selection_path = "[runner].checkpoint.selection"
        selection = _config_object(checkpoint["selection"], selection_path, config_path)
        _validate_keys(
            selection,
            {"mode", "epoch"},
            selection_path,
            config_path,
            required=("mode",),
        )
        mode = selection["mode"]
        if mode not in ("best", "epoch"):
            raise ValueError(
                f"{selection_path}.mode must be best or epoch in {config_path}"
            )
        flattened["use_best_checkpoint"] = mode == "best"
        if mode == "epoch" and "epoch" not in selection:
            raise ValueError(
                f"{selection_path}.epoch is required when mode is epoch in "
                f"{config_path}"
            )
        if mode == "best" and "epoch" in selection:
            raise ValueError(
                f"{selection_path}.epoch is not allowed when mode is best in "
                f"{config_path}"
            )
        if "epoch" in selection:
            flattened["checkpoint_to_use"] = selection["epoch"]

    if "downstream" in runner:
        downstream = _config_object(
            runner["downstream"], "[runner].downstream", config_path
        )
        _validate_keys(
            downstream,
            {"epochs", "forecast_horizon"},
            "[runner].downstream",
            config_path,
            required=("epochs",),
        )
        flattened["eval_num_epochs"] = downstream["epochs"]
        if "forecast_horizon" in downstream:
            flattened["forecast_horizon"] = downstream["forecast_horizon"]

    if "output" in runner:
        output = _config_object(runner["output"], "[runner].output", config_path)
        _validate_keys(
            output,
            {"skip_combined_plot"},
            "[runner].output",
            config_path,
            required=("skip_combined_plot",),
        )
        flattened["skip_combined_plot"] = output["skip_combined_plot"]

    return flattened


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

    if section == "runner":
        script_options = flatten_runner_options(script_options, config_path)
    elif section == "analysis":
        runner = data.get("runner", {})
        if "strategies" in script_options:
            raise ValueError(
                "[analysis].strategies is derived from "
                f"[runner].masking.strategies in {config_path}; remove the duplicate"
            )
        runner_options = flatten_runner_options(runner, config_path)
        if "mask_strategies" in runner_options:
            script_options = {
                **script_options,
                "strategies": runner_options["mask_strategies"],
            }

    configured_derived_options = sorted(
        DERIVED_CONFIG_OPTIONS & (set(common) | set(script_options))
    )
    if configured_derived_options:
        raise ValueError(
            "Options derived from the config filename must not be configured in "
            f"{config_path}: " + ", ".join(configured_derived_options)
        )

    misplaced_common_options = sorted(COMMON_ONLY_OPTIONS & set(script_options))
    if misplaced_common_options:
        raise ValueError(
            "Coverage options must be defined only in [common] in "
            f"{config_path}: " + ", ".join(misplaced_common_options)
        )
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
    resolved_path, data = read_config_file(config_path)
    raw_options = _section_options(data, section, resolved_path)
    actions = _actions_by_destination(parser)

    if any(name in data for name in CONFIG_SECTIONS):
        explicitly_overridden = {
            action.dest for action in _explicit_actions(parser, argv)
        }
        overridden_coverage = sorted(
            COMMON_ONLY_OPTIONS & explicitly_overridden
        )
        if overridden_coverage:
            raise ValueError(
                "Coverage options come only from [common] when --config is used: "
                + ", ".join(overridden_coverage)
            )
        overridden_derived = sorted(
            DERIVED_CONFIG_OPTIONS & explicitly_overridden
        )
        if overridden_derived:
            raise ValueError(
                "Options derived from the config filename cannot be overridden "
                "when --config is used: " + ", ".join(overridden_derived)
            )

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
    """Apply file defaults; configured stocks/seeds remain common-only."""
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
    args = parser.parse_args(argv)
    if config_args.config is not None and hasattr(args, "results_dir"):
        args.results_dir = str(results_dir_from_config(config_args.config))
    return args
