# app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    r2_score,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from model import train_and_evaluate
from utils import plot_confusion_matrix, plot_correlation_matrix

st.set_page_config(layout="wide")
st.title("🧠 Machine Learning Model Lab")

st.sidebar.header("Model Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ---------------- DESCRIPTIVE ANALYSIS ---------------- #
    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.write("Data Types:")
        st.write(df.dtypes)

    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    corr_fig = plot_correlation_matrix(df)
    if corr_fig:
        st.pyplot(corr_fig)

    # ---------------- PROBLEM TYPE ---------------- #
    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    # Auto filter target columns
    if problem_type == "Classification":
        possible_targets = df.select_dtypes(exclude=["number"]).columns.tolist()
    else:
        possible_targets = df.select_dtypes(include=["number"]).columns.tolist()

    if not possible_targets:
        st.error("No suitable target variables found for selected problem type.")
        st.stop()

    target = st.sidebar.selectbox("Select Target Variable", possible_targets)

    features = st.sidebar.multiselect(
        "Select Feature Variables",
        [col for col in df.columns if col != target]
    )

    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

    if problem_type == "Classification":
        model_type = st.sidebar.selectbox(
            "Select Classification Model",
            ["Gaussian", "Multinomial", "Bernoulli"]
        )
    else:
        model_type = st.sidebar.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Random Forest"]
        )

    if st.sidebar.button("🚀 Run Model"):

        if not features:
            st.warning("Please select at least one feature variable.")
            st.stop()

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        st.subheader("📊 Model Evaluation")

        # ---------------- CLASSIFICATION ---------------- #
        if problem_type == "Classification":

            model, train_acc, train_cm, test_acc, test_cm = train_and_evaluate(
                X, y, model_type, test_size
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Train Accuracy", round(train_acc, 4))
                fig1 = plot_confusion_matrix(train_cm, "Train Confusion Matrix")
                st.pyplot(fig1)

            with col2:
                st.metric("Test Accuracy", round(test_acc, 4))
                fig2 = plot_confusion_matrix(test_cm, "Test Confusion Matrix")
                st.pyplot(fig2)

        # ---------------- REGRESSION ---------------- #
        else:

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

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Train R² Score", round(train_r2, 4))
                st.metric("Train MSE", round(train_mse, 4))

            with col2:
                st.metric("Test R² Score", round(test_r2, 4))
                st.metric("Test MSE", round(test_mse, 4))

else:
    st.info("Upload a CSV file to begin.")