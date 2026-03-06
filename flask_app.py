import os
import io
import base64
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Import existing logic safely
from model import train_and_evaluate
from utils import plot_confusion_matrix, plot_correlation_matrix

# Use 'Agg' backend to avoid GUI issues
plt.switch_backend('Agg')

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Store uploaded dataframe temporarily in memory for the session (simple approach for single-user dev)
# In production, this should be stored more robustly.
data_store = {}

def fig_to_base64(fig):
    """Utility to convert matplotlib figure to base64 string."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        df = pd.read_csv(file)
        data_store["df"] = df
        
        # Calculate stats
        shape = df.shape
        dtypes = df.dtypes.astype(str).to_dict()
        missing = df.isnull().sum().to_dict()
        describe = df.describe().to_dict()
        
        corr_fig = plot_correlation_matrix(df)
        corr_img = fig_to_base64(corr_fig)
        
        # Possible targets
        possible_targets_class = df.select_dtypes(exclude=["number"]).columns.tolist()
        possible_targets_reg = df.select_dtypes(include=["number"]).columns.tolist()
        all_columns = df.columns.tolist()

        return jsonify({
            "shape": shape,
            "dtypes": dtypes,
            "missing": missing,
            "describe": describe,
            "corr_img": corr_img,
            "targets_classification": possible_targets_class,
            "targets_regression": possible_targets_reg,
            "all_columns": all_columns
        })

@app.route("/api/run_model", methods=["POST"])
def run_model():
    if "df" not in data_store:
        return jsonify({"error": "No dataset uploaded"}), 400
    
    df = data_store["df"]
    data = request.json
    
    problem_type = data.get("problem_type")
    target = data.get("target")
    features = data.get("features", [])
    test_size = float(data.get("test_size", 0.2))
    model_type = data.get("model_type")

    if not features:
        return jsonify({"error": "Please select at least one feature."}), 400

    X = df[features]
    y = df[target]
    
    # Handle NaN values automatically by filling them
    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    result = {
        "problem_type": problem_type
    }

    if problem_type == "Classification":
        model, train_acc, train_cm, test_acc, test_cm = train_and_evaluate(
            X, y, model_type, test_size
        )
        train_cm_fig = plot_confusion_matrix(train_cm, "Train Confusion Matrix")
        test_cm_fig = plot_confusion_matrix(test_cm, "Test Confusion Matrix")
        
        result.update({
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "train_cm_img": fig_to_base64(train_cm_fig),
            "test_cm_img": fig_to_base64(test_cm_fig)
        })

    else:
        # Regression
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        result.update({
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "train_mse": round(train_mse, 4),
            "test_mse": round(test_mse, 4)
        })

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
