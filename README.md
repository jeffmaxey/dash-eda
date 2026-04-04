# 📊 Dash EDA

An enterprise-grade **Exploratory Data Analysis** dashboard built with [Dash](https://dash.plotly.com/), [Plotly](https://plotly.com/python/), [pandas](https://pandas.pydata.org/), and [Mantine](https://mantine.dev/) components.

---

## Features

- **File Upload** – drag & drop CSV or Excel files directly into the browser
- **Overview** – row/column counts, memory usage, duplicate detection, missing-value summary
- **Statistics** – numeric `describe()` and categorical top-value tables
- **Correlation Heatmap** – Pearson correlation matrix rendered as an interactive heatmap
- **Distributions** – per-column histogram (numeric) or bar chart (categorical) with mean line
- **Missing Values** – colour-coded bar chart of missing percentage per column
- **Data Grid** – AG Grid table with sorting, filtering, and pagination
- **CLI** – `eda analyze / report / info` commands via Click
- **Python API** – `EDAAnalyzer` class for programmatic access
- **Flask-Caching** – in-memory cache layer on the Dash server
- **Light / Dark theme** – one-click colour-scheme toggle

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/dash-eda.git
cd dash-eda

# 2. Install in editable mode (creates the `eda` CLI entry point)
pip install -e .
```

---

## Usage

### Run the dashboard

```bash
python app.py
# or
python -m dashboard.app
```

Open <http://localhost:8050> in your browser, then drag & drop a CSV or Excel file.

### Python API

```python
from eda_core import EDAAnalyzer, from_csv, from_dataframe
import pandas as pd

# From a file
analyzer = from_csv("data/titanic.csv")

# From an existing DataFrame
df = pd.read_csv("data/titanic.csv")
analyzer = from_dataframe(df)

print(analyzer.overview())
print(analyzer.summary_stats())
print(analyzer.correlation())
print(analyzer.column_analysis("Age"))
print(analyzer.outliers("Fare"))

# Save a full JSON report
analyzer.to_report("report.json")
```

### CLI

```bash
# Print dataset overview to the console
eda info data/titanic.csv

# Output full analysis as JSON to stdout
eda analyze data/titanic.csv

# Save analysis JSON to a file
eda analyze data/titanic.csv -o analysis.json

# Save a full report
eda report data/titanic.csv -o report.json
```

---

## Project Structure

```
dash-eda/
├── pyproject.toml          # Package metadata & entry points
├── app.py                  # Root entry point
├── eda_core/               # Business logic package
│   ├── __init__.py
│   ├── analysis.py         # Core pandas analysis functions
│   ├── charts.py           # Plotly chart builders
│   ├── api.py              # EDAAnalyzer public API
│   └── cli.py              # Click CLI (eda command)
├── dashboard/              # Dash UI layer
│   ├── app.py              # App factory (create_app)
│   ├── layout.py           # MantineProvider + AppShell layout
│   ├── components/
│   │   ├── header.py       # AppShell header component
│   │   └── cards.py        # Stat/info card helpers
│   └── callbacks/
│       ├── data.py         # Upload, store, overview, stats, grid
│       └── charts.py       # Correlation, distribution, missing charts
└── tests/
    ├── conftest.py         # Shared fixtures
    ├── test_analysis.py    # Unit tests for analysis.py
    └── test_api.py         # Unit tests for api.py
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=eda_core --cov-report=term-missing
```

---

## License

[MIT](LICENSE)
