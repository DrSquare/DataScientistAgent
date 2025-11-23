# Data Scientist Agent

A LangGraph-powered, multi-agent data science assistant with a Flask API and lightweight front-end. The app plans, executes, and summarizes an end-to-end workflow (loading, cleaning, EDA, statistical tests, wrangling, prep, viz, modeling, insight summary) and generates a downloadable Jupyter notebook.

## Features
- **Planning agent** defines the workflow steps.
- **Specialized sub-agents** for loading, cleaning, EDA, statistical analysis, data wrangling, preparation, visualization, predictive modeling, and summary.
- **Flask API** to trigger runs, fetch health, and download notebooks.
- **Front-end** to upload CSVs, view the plan, inspect outputs, and grab the generated notebook.
- **Notebook generation** with markdown/code cells per step for reproducibility.

## Quickstart

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the app
```bash
flask --app app run --host 0.0.0.0 --port 5000
```
Open http://localhost:5000 to use the UI.

### API
- `GET /api/health` – liveness probe.
- `POST /api/run` – run the full flow. Form-data fields: `instructions` (text), optional `file` (CSV upload).
- `GET /api/notebook/<filename>` – download the generated notebook.

See [API_DOC.md](API_DOC.md) for full details and payload examples.

## How it works
The orchestrator builds a LangGraph with nodes representing the planning agent plus sub-agents for each data science task. Each node updates shared state, logs outputs, and appends notebook cells. After execution, the service returns:
- **Plan** – ordered steps.
- **Outputs** – structured text per step.
- **Summary** – concise highlights.
- **Notebook** – downloadable artifact covering the full run.

If no dataset is uploaded, the system uses a small synthetic dataset to keep the demo runnable.

## Development notes
- Outputs are persisted under `uploads/`, `notebooks/`, and `static/` (for generated plots).
- Visualization uses a headless Matplotlib backend.
- Predictive modeling defaults to `LogisticRegression`; if the dataset lacks a target column, a synthetic one is derived from numeric features to keep training functional. Target inference respects common names (`target`, `label`, `outcome`, `y`) and low-cardinality columns before synthesizing.
- Visualization gracefully skips plots when no numeric columns are present to avoid runtime errors.
