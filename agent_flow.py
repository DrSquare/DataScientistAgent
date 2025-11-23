import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import nbformat as nbf
import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DSState:
    instructions: str = ""
    dataset_path: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    plan: List[str] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    notebook_cells: List[Any] = field(default_factory=list)
    notebook_filename: Optional[str] = None


TASKS = [
    "Data Loading",
    "Data Cleaning",
    "Exploratory Data Analysis",
    "Statistical Analysis",
    "Data Wrangling",
    "Data Preparation",
    "Data Visualization",
    "Predictive Modeling",
    "Insight Summary",
]


def add_output(state: DSState, title: str, content: str, code: Optional[str] = None):
    state.outputs.append({"title": title, "content": content})
    state.notebook_cells.append(nbf.v4.new_markdown_cell(f"### {title}\n\n{content}"))
    if code:
        state.notebook_cells.append(nbf.v4.new_code_cell(code))


def plan_agent(state: DSState) -> DSState:
    plan_items = [f"Step {i+1}: {task}" for i, task in enumerate(TASKS)]
    state.plan = plan_items
    plan_text = "\n".join(plan_items)
    add_output(state, "Plan", f"Planned tasks based on request: \n{plan_text}")
    return state


def load_data_agent(state: DSState) -> DSState:
    if state.dataset_path and os.path.exists(state.dataset_path):
        df = pd.read_csv(state.dataset_path)
        source = os.path.basename(state.dataset_path)
    else:
        # create synthetic dataset
        rng = np.random.default_rng(42)
        size = 200
        df = pd.DataFrame(
            {
                "feature_a": rng.normal(loc=0, scale=1, size=size),
                "feature_b": rng.normal(loc=5, scale=2, size=size),
                "category": rng.choice(["A", "B", "C"], size=size),
            }
        )
        df["target"] = (df["feature_a"] + df["feature_b"] > 5).astype(int)
        source = "synthetic-data.csv"
    state.df = df
    add_output(
        state,
        "Data Loading",
        f"Loaded dataset `{source}` with shape {df.shape}. First rows:\n{df.head().to_markdown()}",
        code="df.head()",
    )
    return state


def clean_data_agent(state: DSState) -> DSState:
    df = state.df.copy()
    missing_before = df.isnull().sum().sum()
    df = df.fillna(method="ffill").fillna(method="bfill")
    missing_after = df.isnull().sum().sum()
    state.df = df
    add_output(
        state,
        "Data Cleaning",
        f"Handled missing values (before: {missing_before}, after: {missing_after}). Data types:\n{df.dtypes}",
        code="df.dtypes",
    )
    return state


def eda_agent(state: DSState) -> DSState:
    df = state.df
    desc = df.describe(include="all").transpose()
    add_output(
        state,
        "Exploratory Data Analysis",
        f"Summary statistics:\n{desc.to_markdown()}",
        code="df.describe(include='all')",
    )
    return state


def statistical_analysis_agent(state: DSState) -> DSState:
    df = state.df
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[:2]
        corr = df[col1].corr(df[col2])
        t_stat, p_value = stats.ttest_ind(df[col1], df[col2])
        text = f"Correlation between {col1} and {col2}: {corr:.3f}. t-test p-value: {p_value:.4f}."
        code = (
            "from scipy import stats\n"
            f"stats.ttest_ind(df['{col1}'], df['{col2}'])\n"
            f"df['{col1}'].corr(df['{col2}'])"
        )
    else:
        text = "Not enough numeric columns for statistical comparison."
        code = "df.head()"
    add_output(state, "Statistical Analysis", text, code=code)
    return state


def data_wrangling_agent(state: DSState) -> DSState:
    df = state.df
    df = df.copy()
    df["interaction"] = df.select_dtypes(include=["number"]).prod(axis=1)
    df["scaled_feature_a"] = (df["feature_a"] - df["feature_a"].mean()) / df["feature_a"].std()
    state.df = df
    add_output(
        state,
        "Data Wrangling",
        "Engineered `interaction` and standardized `feature_a`.",
        code="df[['interaction', 'scaled_feature_a']].head()",
    )
    return state


def data_preparation_agent(state: DSState) -> DSState:
    df = state.df
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    state.df = df
    add_output(
        state,
        "Data Preparation",
        f"Prepared features/target with target column `{target}`. Feature preview:\n{X.head().to_markdown()}",
        code="X.head()",
    )
    return state


def visualization_agent(state: DSState) -> DSState:
    df = state.df
    fig, ax = plt.subplots(figsize=(5, 4))
    df.select_dtypes(include=["number"]).hist(ax=ax)
    viz_id = f"viz_{uuid.uuid4().hex}.png"
    output_path = os.path.join("static", viz_id)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    add_output(
        state,
        "Data Visualization",
        f"Saved numeric distribution plot to `{output_path}`.",
        code="df.hist(figsize=(8,6))",
    )
    return state


def predictive_modeling_agent(state: DSState) -> DSState:
    df = state.df
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=500)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    try:
        proba = clf.predict_proba(X_test)[:, 1]
        metric = f"Accuracy: {accuracy_score(y_test, preds):.3f}, Brier score proxy: {mean_squared_error(y_test, proba):.3f}."
    except Exception:
        metric = f"Accuracy: {accuracy_score(y_test, preds):.3f}."

    report = classification_report(y_test, preds)
    add_output(
        state,
        "Predictive Modeling",
        f"LogisticRegression evaluation -> {metric}\n\nClassification report:\n````\n{report}\n````",
        code=(
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "train, test = train_test_split(df, test_size=0.2, random_state=42)"
        ),
    )
    return state


def insight_summary_agent(state: DSState) -> DSState:
    highlights = [
        "Data loaded and cleaned with standardized numeric features.",
        "Key statistical tests and EDA performed to surface distributions and correlations.",
        "Predictive model evaluated with accuracy metric for quick feedback.",
    ]
    summary = "\n".join(f"- {h}" for h in highlights)
    add_output(state, "Insight Summary", summary)
    return state


def build_notebook(state: DSState, notebook_dir: str):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("## Data Scientist Agent Run"))
    nb.cells.extend(state.notebook_cells)
    filename = f"notebook_{uuid.uuid4().hex}.ipynb"
    filepath = os.path.join(notebook_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    state.notebook_filename = filename


def run_data_science_flow(
    instructions: str,
    dataset_path: Optional[str] = None,
    plan_only: bool = False,
    notebook_dir: Optional[str] = None,
) -> Dict[str, Any]:
    state = DSState(instructions=instructions, dataset_path=dataset_path)
    graph = StateGraph(DSState)
    graph.add_node("plan", plan_agent)
    graph.add_node("load", load_data_agent)
    graph.add_node("clean", clean_data_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("stats", statistical_analysis_agent)
    graph.add_node("wrangle", data_wrangling_agent)
    graph.add_node("prepare", data_preparation_agent)
    graph.add_node("viz", visualization_agent)
    graph.add_node("model", predictive_modeling_agent)
    graph.add_node("summary", insight_summary_agent)

    graph.add_edge("plan", "load")
    graph.add_edge("load", "clean")
    graph.add_edge("clean", "eda")
    graph.add_edge("eda", "stats")
    graph.add_edge("stats", "wrangle")
    graph.add_edge("wrangle", "prepare")
    graph.add_edge("prepare", "viz")
    graph.add_edge("viz", "model")
    graph.add_edge("model", "summary")
    graph.add_edge("summary", END)
    graph.set_entry_point("plan")

    app = graph.compile()

    for _ in app.stream(state):
        if plan_only:
            break
        continue

    if notebook_dir:
        os.makedirs(notebook_dir, exist_ok=True)
        build_notebook(state, notebook_dir)

    result = {
        "plan": state.plan,
        "outputs": state.outputs,
        "summary": state.outputs[-1]["content"] if state.outputs else "",
        "notebook": state.notebook_filename,
    }
    return result
