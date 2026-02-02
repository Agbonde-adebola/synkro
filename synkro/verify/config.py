"""Configuration parsing for SQL verifiers.

Supports:
- YAML config files with sql_file references
- Environment variable expansion (${VAR})
- DSN in config
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VerifierConfig:
    """Parsed verifier configuration."""

    sql: str  # Loaded SQL template with {{param}} placeholders
    params: list[str] = field(default_factory=list)
    dsn: str | None = None
    name: str = ""
    intent: str = ""
    on_no_results: str = "skip"
    config_path: Path | None = None


def expand_env_vars(value: str) -> str:
    """Expand ${VAR} to os.environ['VAR'].

    Args:
        value: String potentially containing ${VAR} patterns

    Returns:
        String with environment variables expanded

    Example:
        >>> os.environ["DATABASE_URL"] = "postgresql://localhost/db"
        >>> expand_env_vars("${DATABASE_URL}")
        'postgresql://localhost/db'
    """
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)


def load_sql(sql_file: str, config_dir: Path) -> str:
    """Load SQL from file path (relative to config file).

    Args:
        sql_file: Path to SQL file (relative or absolute)
        config_dir: Directory containing the config file

    Returns:
        SQL template string

    Raises:
        FileNotFoundError: If SQL file doesn't exist
    """
    sql_path = Path(sql_file)
    if not sql_path.is_absolute():
        sql_path = config_dir / sql_file

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    return sql_path.read_text()


def parse_config(config: dict | str | Path, base_dir: Path | None = None) -> VerifierConfig:
    """Parse config from dict, YAML file, or Path.

    Args:
        config: Dict, YAML file path, or Path
        base_dir: Base directory for resolving relative paths (default: cwd)

    Returns:
        Parsed VerifierConfig

    Raises:
        ValueError: If sql_file is missing

    Example:
        # From YAML file
        cfg = parse_config("./verifiers/funding.yaml")

        # From dict
        cfg = parse_config({
            "sql_file": "./funding.sql",
            "params": ["company_name"],
            "dsn": "${DATABASE_URL}",
        })
    """
    if base_dir is None:
        base_dir = Path.cwd()

    config_path: Path | None = None

    # Load from file if needed
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = yaml.safe_load(config_path.read_text())
        base_dir = config_path.parent

    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict, got {type(config)}")

    # Require sql_file
    if "sql_file" not in config:
        raise ValueError("Config must have 'sql_file' field")

    sql_file = config["sql_file"]
    if "${" in sql_file:
        sql_file = expand_env_vars(sql_file)
    sql = load_sql(sql_file, base_dir)

    # Expand DSN env vars
    dsn = config.get("dsn")
    if dsn and "${" in dsn:
        dsn = expand_env_vars(dsn)

    return VerifierConfig(
        sql=sql,
        params=config.get("params", []),
        dsn=dsn,
        name=config.get("name", ""),
        intent=config.get("intent", ""),
        on_no_results=config.get("on_no_results", "skip"),
        config_path=config_path,
    )


def parse_configs(
    *configs: str | Path | dict | list[Any],
    base_dir: Path | None = None,
) -> list[VerifierConfig]:
    """Parse multiple configs into a list of VerifierConfig.

    Handles:
    - str/Path to YAML file
    - str/Path to directory (loads all .yaml/.yml files)
    - dict (inline config)
    - list of any of the above

    Args:
        *configs: Any number of configs
        base_dir: Base directory for resolving relative paths

    Returns:
        List of parsed VerifierConfig objects
    """
    if base_dir is None:
        base_dir = Path.cwd()

    result = []

    for config in configs:
        if isinstance(config, dict):
            result.append(parse_config(config, base_dir))
        elif isinstance(config, (str, Path)):
            path = Path(config)
            if path.is_dir():
                # Load all YAML files in directory
                for yaml_file in sorted(path.glob("*.yaml")):
                    result.append(parse_config(yaml_file))
                for yaml_file in sorted(path.glob("*.yml")):
                    result.append(parse_config(yaml_file))
            elif path.is_file():
                result.append(parse_config(path))
            else:
                raise FileNotFoundError(f"Config not found: {path}")
        elif isinstance(config, list):
            # Recursively flatten lists
            result.extend(parse_configs(*config, base_dir=base_dir))
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

    return result
