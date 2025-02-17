import streamlit as st
import io
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
from sklearn.tree import plot_tree

# Streamlit page configuration
st.set_page_config(page_title="Petrophysics Expert Robot", layout="wide")
st.title("📊 Petrophysics Expert Robot")

# Global variables
dfs = []  # List to store dataframes for each well
target_log = None
input_logs = None
models = {
    "Linear Regression": None,
    "Random Forest": None,
    "Neural Network": None,
    "SVR": None,
    "Gaussian Process": None,
    "KNN": None
}
updated_X = None

# Load LAS or CSV files
def load_file():
    global dfs
    uploaded_files = st.file_uploader("Upload LAS or CSV files", type=["las", "csv"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.warning("No file uploaded yet!")
        return

    dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith(".las"):
                las = lasio.read(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))
                temp_df = las.df().reset_index()
            elif uploaded_file.name.endswith(".csv"):
                temp_df = pd.read_csv(uploaded_file)

            dfs.append(temp_df)
            st.success(f"Loaded: {uploaded_file.name} ({len(temp_df)} rows)")

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

load_file()

# Show input logs
def show_input_logs():
    if dfs:
        for i, df in enumerate(dfs):
            st.write(f"Well {i+1} Logs")
            fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(15, 6))
            for j, col in enumerate(df.columns):
                axes[j].plot(df[col], df.index, label=col)
                axes[j].set_ylim(df.index.max(), df.index.min())  # Invert depth axis
                axes[j].set_xlabel(col)
                axes[j].set_ylabel("Depth")
                axes[j].grid()
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("No data loaded!")

# Fix missing values
def fix_logs():
    global dfs
    if not dfs:
        st.warning("⚠ No data loaded!")
        return

    missing_values = st.text_input("Enter missing values to replace (comma separated, e.g., -999.25, -999)", "-999.25,-999,-9999")
    missing_values = [float(val.strip()) for val in missing_values.split(",")]

    cleaned_dfs = []  # To store cleaned dataframes

    for df in dfs:
        df.replace(missing_values, np.nan, inplace=True)
        fill_method = st.selectbox("Choose method to fill missing values", ["Drop Rows", "Fill with Mean", "Fill with Median", "Interpolate"])
        
        if fill_method == "Drop Rows":
            df.dropna(inplace=True)
        elif fill_method == "Fill with Mean":
            df.fillna(df.mean(), inplace=True)
        elif fill_method == "Fill with Median":
            df.fillna(df.median(), inplace=True)
        elif fill_method == "Interpolate":
            df.interpolate(inplace=True)

        cleaned_dfs.append(df)  # Store the cleaned dataframe

    # Store cleaned data in session state for future use
    st.session_state["cleaned_dfs"] = cleaned_dfs

    st.success("✔ Data cleaned successfully!")
    show_input_logs()

    # Save button to store cleaned data
    if st.button("Save Cleaned Logs"):
        # Store cleaned data in session state
        st.session_state["cleaned_dfs"] = cleaned_dfs
        st.success("✔ Cleaned logs saved to session state!")
    else:
        st.warning("⚠ Cleaned logs not saved!")

# Select target and input logs for Training
def select_training_data():
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    st.write("### Select Training Data")
    df = st.session_state["cleaned_dfs"][0]
    
    # Select target log and input logs from cleaned data
    st.session_state["target_log"] = st.selectbox("Select Target Log:", df.columns)
    st.session_state["input_logs"] = st.multiselect("Select Input Logs:", df.columns, 
                                                     default=[col for col in df.columns if col != st.session_state["target_log"]])

    if st.button("Confirm Selection"):
        if not st.session_state["target_log"] or not st.session_state["input_logs"]:
            st.warning("⚠ Please select both input and target logs!")
        else:
            st.success(f"✔ Logs selected successfully!\nTarget: {st.session_state['target_log']}\nInputs: {st.session_state['input_logs']}")

# Plot histograms of input logs and target log
def plot_histograms():
    if "input_logs" not in st.session_state or "target_log" not in st.session_state:
        st.warning("⚠ No training data selected!")
        return
    
    input_logs = st.session_state["input_logs"]
    target_log = st.session_state["target_log"]

    # Ensure cleaned data is available
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if input_logs and target_log:
        st.write("### Histograms")
        
        # Use the first cleaned dataframe 
        combined_df = pd.concat(st.session_state["cleaned_dfs"], axis=0)  # Combine all cleaned data
        fig, axes = plt.subplots(nrows=1, ncols=len(input_logs) + 1, figsize=(15, 6))

        for i, col in enumerate(input_logs):
            if col in combined_df.columns:
                axes[i].hist(combined_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(col)

        # Plot the target log
        if target_log in combined_df.columns:
            axes[-1].hist(combined_df[target_log].dropna(), bins=30, edgecolor='black', alpha=0.7, color='red')
            axes[-1].set_title(target_log)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠ No data loaded or logs selected!")


# Plot correlation matrix and update X data
def plot_correlation_matrix():
    global updated_X
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or not st.session_state["input_logs"]:
        st.warning("⚠ No logs selected!")
        return

    input_logs = st.session_state["input_logs"]
    
    # Use cleaned data
    combined_df = pd.concat(st.session_state["cleaned_dfs"], axis=0)
    
    if input_logs:
        st.write("### Correlation Matrix")

        # Calculate the correlation matrix
        corr_matrix = combined_df[input_logs].corr()

        # Drop highly correlated features
        high_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.add(corr_matrix.columns[i])

        updated_X = combined_df[input_logs].drop(columns=high_corr)

        # Plot updated X data as logs
        fig, axes = plt.subplots(nrows=1, ncols=len(updated_X.columns), figsize=(15, 6))
        for i, col in enumerate(updated_X.columns):
            axes[i].plot(updated_X[col], updated_X.index, label=col)
            axes[i].set_ylim(updated_X.index.max(), updated_X.index.min())  # Invert depth axis
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Depth")
            axes[i].grid()
        plt.tight_layout()
        st.pyplot(fig)

        # Plot correlation matrix
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, ax=ax_corr, cmap="coolwarm")
        ax_corr.set_title("Correlation Matrix")
        st.pyplot(fig_corr)
    else:
        st.warning("⚠ No logs selected!")

# Train models with hyperparameter selection
def train_models():
    global models
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or "target_log" not in st.session_state:
        st.warning("⚠ No logs selected!")
        return

    input_logs = st.session_state["input_logs"]
    target_log = st.session_state["target_log"]

    if st.session_state["cleaned_dfs"] and input_logs and target_log:
        model_name = st.selectbox("Choose Model", list(models.keys()))

        # Model selection and hyperparameter tuning
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_name == "Neural Network":
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 64,64)", "64,64")
            max_iter = st.slider("Max Iterations", 100, 1000, 100)
            model = MLPRegressor(hidden_layer_sizes=tuple(map(int, hidden_layer_sizes.split(','))), max_iter=max_iter, random_state=42)
        elif model_name == "SVR":
            kernel = st.text_input("Kernel (e.g., 'rbf', 'linear')", "rbf")
            C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0)
            gamma = st.text_input("Gamma (Kernel coefficient)", "scale")
            model = SVR(kernel=kernel, C=C, gamma=gamma)
        elif model_name == "Gaussian Process":
            # Define the kernel using RBF and Constant Kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        elif model_name == "KNN":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)

        if st.button("Train Model"):
            # Use cleaned data and updated_X for training
            combined_df = pd.concat(st.session_state["cleaned_dfs"], axis=0)
            X = updated_X.dropna() if updated_X is not None else combined_df[input_logs].dropna()
            y = combined_df[target_log].dropna()

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the model
            model.fit(X_train, y_train)
            models[model_name] = model

            # Visualize model structure
            if model_name == "Random Forest":
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tree(model.estimators_[0], filled=True, ax=ax, feature_names=X.columns, max_depth=3)
                plt.title("Random Forest Tree")
                st.pyplot(fig)
            elif model_name == "Neural Network":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f"NN Architecture: {hidden_layer_sizes}", fontsize=12, ha="center")
                ax.axis("off")
                plt.title("Neural Network Architecture")
                st.pyplot(fig)

            st.success(f"{model_name} trained successfully!")
    else:
        st.warning("⚠ No data or logs selected!")


# Show predictions
def show_predictions():
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or "target_log" not in st.session_state:
        st.warning("⚠ No logs selected!")
        return

    input_logs = st.session_state["input_logs"]
    target_log = st.session_state["target_log"]

    if st.session_state["cleaned_dfs"] and input_logs and target_log and models:
        # Combine cleaned data
        combined_df = pd.concat(st.session_state["cleaned_dfs"], axis=0)

        # Align X and y based on input and target logs
        X = updated_X.dropna() if updated_X is not None else combined_df[input_logs].dropna()
        y = combined_df[target_log].dropna()

        # Ensure X and y have the same index after dropping NaNs
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        # Standardize the data
        X_scaled = StandardScaler().fit_transform(X)

        # Plot actual vs predicted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y.index, y.values, label="Actual", color="black")

        # Predictions from each trained model
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
    else:
        st.warning("⚠ No data or models trained!")

# Load and predict new data
def load_and_predict_new_data():
    uploaded_file = st.file_uploader("Upload new LAS or CSV file", type=["las", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".las"):
            las = lasio.read(uploaded_file)
            new_df = las.df()
            new_df.reset_index(inplace=True)
        elif uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)

        # Set Depth column as index
        if "Depth" in new_df.columns:
            new_df.set_index("Depth", inplace=True)

        # Select input logs
        input_logs_new = st.multiselect("Select Input Logs for Prediction", new_df.columns)
        if input_logs_new:
            X_new = new_df[input_logs_new].dropna()
            X_new_scaled = StandardScaler().fit_transform(X_new)

            # Predictions
            predictions = {}
            for model_name, model in models.items():
                if model is not None:
                    y_pred_new = model.predict(X_new_scaled)
                    predictions[model_name] = y_pred_new

            # Create a DataFrame for predictions
            pred_df = pd.DataFrame(predictions, index=X_new.index)
            pred_df["Depth"] = pred_df.index

            # Show new logs
            st.write("New Logs")
            fig, axes = plt.subplots(nrows=1, ncols=len(input_logs_new), figsize=(15, 6))
            for i, col in enumerate(input_logs_new):
                axes[i].plot(new_df[col], new_df.index, label=col)
                axes[i].set_ylim(new_df.index.max(), new_df.index.min())  # Invert depth axis
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Depth")
                axes[i].grid()
            plt.tight_layout()
            st.pyplot(fig)

            # Show predicted log
            st.write("Predicted Log")
            fig, ax = plt.subplots(figsize=(10, 6))
            for model_name in pred_df.columns:
                if model_name != "Depth":
                    ax.plot(pred_df["Depth"], pred_df[model_name], label=model_name)
            ax.set_xlabel("Depth")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Predicted Log")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            # Export results
            if st.button("Export Results"):
                export_path = st.text_input("Enter file path to save results (e.g., results.las or results.csv)")
                if export_path:
                    if export_path.endswith(".las"):
                        las = lasio.LASFile()
                        las.set_data_from_df(pred_df)
                        las.write(export_path)
                    elif export_path.endswith(".csv"):
                        pred_df.to_csv(export_path, index=False)
                    st.success("Results exported successfully!")
    else:
        st.warning("No file selected!")

# Main UI
def main():
    st.title("Petrophysical Property Predictor")

    menu = ["Load File", "Show Input Logs", "Fix Logs", "Select Training Data", "Plot Histograms", "Plot Correlation Matrix", "Train Models", "Show Predictions", "Load & Predict New Data"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Load File":
        load_file()
    elif choice == "Show Input Logs":
        show_input_logs()
    elif choice == "Fix Logs":
        fix_logs()   
    elif choice == "Select Training Data":
        select_training_data()
    elif choice == "Plot Histograms":
        plot_histograms()
    elif choice == "Plot Correlation Matrix":
        plot_correlation_matrix()
    elif choice == "Train Models":
        train_models()
    elif choice == "Show Predictions":
        show_predictions()
    elif choice == "Load & Predict New Data":
        load_and_predict_new_data()

if __name__ == "__main__":
    main()
