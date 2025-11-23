# API Documentation

## Base URL
```
http://localhost:5000
```

## Endpoints

### Health check
`GET /api/health`

**Response**
```json
{
  "status": "ok"
}
```

### Run data science flow
`POST /api/run`

Form-data body:
- `instructions` (string, optional): goal or questions to influence the workflow narrative.
- `file` (file, optional): CSV file to analyze. If omitted, a synthetic dataset is used.
- `plan_only` (boolean-like string, optional): if `true`, stops after planning.

**Response**
```json
{
  "plan": ["Step 1: Data Loading", ...],
  "outputs": [
    {"title": "Plan", "content": "Planned tasks based on request..."},
    {"title": "Data Loading", "content": "Loaded dataset..."},
    ...
  ],
  "summary": "- Key highlights...",
  "notebook": "notebook_<id>.ipynb"
}
```

### Download notebook
`GET /api/notebook/<filename>`

Returns the generated Jupyter notebook for the requested run.

## Notes on Agents
- **Planning agent** creates the ordered workflow.
- **Data Loading agent** ingests the CSV or produces a small synthetic dataset.
- **Data Cleaning agent** resolves missing values.
- **EDA agent** surfaces descriptive statistics.
- **Statistical agent** runs correlations and a t-test when numeric data are present.
- **Data Wrangling agent** engineers interaction and standardized features.
- **Preparation agent** separates features/target.
- **Visualization agent** adapts to requested visualization types (e.g., histogram, boxplot, scatter, heatmap, scatter matrix, line) inferred from the `instructions` field and saves plots to `static/` for reference in the outputs.
- **Predictive Modeling agent** trains/evaluates a logistic regression pipeline with preprocessing.
- **Insight Summary agent** compiles highlights for quick consumption.
