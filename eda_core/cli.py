"""Click CLI for Dash EDA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from eda_core.analysis import load_dataframe
from eda_core.api import EDAAnalyzer


def _load(file: str) -> pd.DataFrame:
    path = Path(file)
    if not path.exists():
        click.echo(f"Error: file not found – {file}", err=True)
        sys.exit(1)
    return load_dataframe(str(path), path.name)


@click.group()
def main() -> None:
    """Dash EDA – command-line exploratory data analysis tool."""


@main.command("analyze")
@click.argument("file")
@click.option("--output", "-o", default=None, help="Path for JSON output (default: stdout)")
def analyze(file: str, output: str | None) -> None:
    """Run analysis on FILE and output results as JSON."""
    df = _load(file)
    analyzer = EDAAnalyzer(df)
    result = {
        "overview": analyzer.overview(),
        "summary_stats": analyzer.summary_stats(),
        "correlation": analyzer.correlation(),
    }
    payload = json.dumps(result, indent=2, default=str)
    if output:
        Path(output).write_text(payload, encoding="utf-8")
        click.echo(f"Analysis saved to {output}")
    else:
        click.echo(payload)


@main.command("report")
@click.argument("file")
@click.option("--output", "-o", default="report.json", show_default=True, help="Output JSON file path")
def report(file: str, output: str) -> None:
    """Generate a full EDA report for FILE and save it to OUTPUT."""
    df = _load(file)
    analyzer = EDAAnalyzer(df)
    analyzer.to_report(output)
    click.echo(f"Report saved to {output}")


@main.command("info")
@click.argument("file")
def info(file: str) -> None:
    """Print a concise dataset overview for FILE to the console."""
    df = _load(file)
    analyzer = EDAAnalyzer(df)
    ov = analyzer.overview()

    click.echo(f"\n{'=' * 50}")
    click.echo(f"  File   : {file}")
    click.echo(f"  Rows   : {ov['shape']['rows']:,}")
    click.echo(f"  Columns: {ov['shape']['columns']}")
    click.echo(f"  Memory : {ov['memory_usage_mb']} MB")
    click.echo(f"  Duplicates: {ov['duplicate_rows']:,}")
    total_missing = sum(ov["missing_counts"].values())
    click.echo(f"  Missing cells: {total_missing:,}")
    click.echo(f"{'=' * 50}")

    click.echo("\nColumn dtypes:")
    for col, dtype in ov["dtypes"].items():
        missing = ov["missing_counts"].get(col, 0)
        pct = ov["missing_pct"].get(col, 0.0)
        click.echo(f"  {col:<30} {dtype:<15} missing={missing} ({pct}%)")
    click.echo()
