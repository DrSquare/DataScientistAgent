import os
import uuid
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from agent_flow import run_data_science_flow

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "notebooks")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=STATIC_DIR)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/run", methods=["POST"])
def run_flow():
    user_instructions = request.form.get("instructions", "")
    plan_only = request.form.get("plan_only", "false").lower() == "true"
    dataset_path = None

    if "file" in request.files and request.files["file"]:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        if filename:
            dataset_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
            file.save(dataset_path)

    result = run_data_science_flow(
        instructions=user_instructions,
        dataset_path=dataset_path,
        plan_only=plan_only,
        notebook_dir=NOTEBOOK_DIR,
    )

    return jsonify(result)


@app.route("/api/notebook/<path:filename>")
def download_notebook(filename: str):
    return send_from_directory(NOTEBOOK_DIR, filename, as_attachment=True)


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
