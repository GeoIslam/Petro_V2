import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# Streamlit page configuration
st.set_page_config(page_title="Petrophysical Property Predictor", layout="wide")

st.title("📊 Petrophysical Property Predictor")

# Global variables
dfs = []
target_log = None
input_logs = []
models = {}
updated_X = None

# Load file
uploaded_files = st.file_uploader("Upload LAS or CSV files", type=["las", "csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        if file.name.endswith(".las"):
            las = lasio.read(file)
            temp_df = las.df()
            temp_df.reset_index(inplace=True)
        else:
            temp_df = pd.read_csv(file)
        
        if "Depth" in temp_df.columns:
            temp_df.set_index("Depth", inplace=True)
        dfs.append(temp_df)

    st.success(f"✅ {len(uploaded_files)} files loaded successfully!")

    # Show logs
    selected_well = st.selectbox("Select Well to Display Logs", range(1, len(dfs) + 1))
    df = dfs[selected_well - 1]

    st.write("### Well Logs")
    fig, axes = plt.subplots(ncols=len(df.columns), figsize=(15, 6))
    for j, col in enumerate(df.columns):
        axes[j].plot(df[col], df.index, label=col)
        axes[j].invert_yaxis()
        axes[j].set_xlabel(col)
        axes[j].set_ylabel("Depth")
        axes[j].grid()
    st.pyplot(fig)

# Select Logs
if dfs:
    df = dfs[0]  # Assume first well for selection
    st.sidebar.header("🔍 Log Selection")

    target_log = st.sidebar.selectbox("Select Target Log (Y)", df.columns)
    input_logs = st.sidebar.multiselect("Select Input Logs (X)", df.columns, default=df.columns.tolist())

# Fix Logs
if st.sidebar.button("🧹 Remove Null Values"):
    for df in dfs:
        df.dropna(inplace=True)
    st.sidebar.success("Null values removed!")

# Plot Histograms
if st.sidebar.button("📊 Plot Histograms"):
    combined_df = pd.concat(dfs, axis=0)
    fig, axes = plt.subplots(ncols=len(input_logs) + 1, figsize=(15, 6))
    for i, col in enumerate(input_logs):
        axes[i].hist(combined_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(col)
    axes[-1].hist(combined_df[target_log].dropna(), bins=30, edgecolor='black', alpha=0.7, color='red')
    axes[-1].set_title(target_log)
    st.pyplot(fig)

# Correlation Matrix
if st.sidebar.button("📈 Plot Correlation Matrix"):
    combined_df = pd.concat(dfs, axis=0)
    corr_matrix = combined_df[input_logs].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, ax=ax_corr, cmap="coolwarm")
    st.pyplot(fig_corr)

    # Drop highly correlated features
    high_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.add(corr_matrix.columns[i])
    
    updated_X = combined_df[input_logs].drop(columns=high_corr)
    st.sidebar.success(f"Removed {len(high_corr)} highly correlated features.")

# Train Models
model_name = st.sidebar.selectbox("🔬 Select Model", ["Linear Regression", "Random Forest", "Neural Network", "SVR"])
train_button = st.sidebar.button("🚀 Train Model")

if train_button:
    combined_df = pd.concat(dfs, axis=0)
    X = updated_X.dropna() if updated_X is not None else combined_df[input_logs].dropna()
    y = combined_df[target_log].dropna()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
        max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == "Neural Network":
        hidden_layers = st.sidebar.text_input("Hidden Layers (e.g., 64,64)", "64,64")
        max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 500)
        model = MLPRegressor(hidden_layer_sizes=tuple(map(int, hidden_layers.split(','))), max_iter=max_iter, random_state=42)
    elif model_name == "SVR":
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0)
        model = SVR(kernel=kernel, C=C)
    
    model.fit(X_train, y_train)
    models[model_name] = model

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.sidebar.success(f"{model_name} trained successfully! R²: {r2:.2f}, RMSE: {rmse:.2f}")

    # Model Visualization (Tree or NN Summary)
    if model_name == "Random Forest":
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_tree(model.estimators_[0], filled=True, ax=ax, feature_names=X.columns, max_depth=3)
        st.pyplot(fig)

# Predictions
if st.sidebar.button("🔮 Show Predictions"):
    combined_df = pd.concat(dfs, axis=0)
    X = updated_X.dropna() if updated_X is not None else combined_df[input_logs].dropna()
    y = combined_df[target_log].dropna()
    X_scaled = StandardScaler().fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y.index, y.values, label="Actual", color="black")

    for model_name, model in models.items():
        if model is not None:
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            rmse = mean_squared_error(y, y_pred, squared=False)
            ax.plot(y.index, y_pred, label=f"{model_name} (R²: {r2:.2f}, RMSE: {rmse:.2f})")

    ax.legend()
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Values")
    ax.grid()
    st.pyplot(fig)
