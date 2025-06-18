# Single-file ML Pipeline for Robotic Arm RUL Prediction

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

# --- 1. Configuration ---

# Data Generation Parameters
N_SAMPLES_MODEL_A = 1000
N_SAMPLES_MODEL_B = 500
NOISE_LEVEL = 0.5

# Model Parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42
}

# File and Directory Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'training_data.csv')
TESTING_DATA_PATH_MODEL_A = os.path.join(DATA_DIR, 'testing_data_model_a.csv')
TESTING_DATA_PATH_MODEL_B = os.path.join(DATA_DIR, 'testing_data_model_b.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'rul_predictor.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')


# --- 2. Data Generation Module ---

def generate_data(n_samples, model_type='A', seed=42):
    """
    Generates sensor data and Remaining Useful Life (RUL) for a given robot model.

    Args:
        n_samples (int): The number of data points to generate.
        model_type (str): 'A' for the standard model, 'B' for the newer model.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with sensor data and RUL.
    """
    np.random.seed(seed)

    # Base characteristics
    age = np.random.uniform(0, 10, n_samples)  # Age in years

    # Model-specific sensor characteristics
    if model_type == 'A':
        # Sensor readings for Model A
        temperature = 60 + 10 * age + np.random.normal(0, 2, n_samples)
        vibration = 5 + 2 * age + np.random.normal(0, 1.5, n_samples)
        current_draw = 10 + 0.5 * age + np.random.normal(0, 0.5, n_samples)
    elif model_type == 'B':
        # Sensor readings for Model B
        temperature = 55 + 8 * age + np.random.normal(0, 1.5, n_samples)
        vibration = 4 + 1.5 * age + np.random.normal(0, 1, n_samples)
        current_draw = 9 + 0.4 * age + np.random.normal(0, 4, n_samples)

    # Remaining Useful Life (RUL) Calculation
    rul = 100 - (10 * age + 0.2 * temperature + 0.5 * vibration + 0.1 * current_draw) + np.random.normal(0, NOISE_LEVEL,
                                                                                                         n_samples)
    rul = np.clip(rul, 0, 100)

    data = pd.DataFrame({
        'age': age,
        'temperature': temperature,
        'vibration': vibration,
        'current_draw': current_draw,
        'rul': rul
    })

    return data


def create_datasets():
    """
    Generates and saves the training and testing datasets.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate Data for Model A (Our Training and Primary Test Data)
    data_model_a = generate_data(N_SAMPLES_MODEL_A, model_type='A', seed=42)

    # Split Model A data into training and a corresponding test set
    train_df, test_model_a_df = train_test_split(data_model_a, test_size=0.2, random_state=42)

    train_df.to_csv(TRAINING_DATA_PATH, index=False)
    test_model_a_df.to_csv(TESTING_DATA_PATH_MODEL_A, index=False)

    # Generate Data for Model B (Our Secondary Test Data)
    # Use a different seed for this dataset to ensure it's distinct
    test_model_b_df = generate_data(N_SAMPLES_MODEL_B, model_type='B', seed=101)
    test_model_b_df.to_csv(TESTING_DATA_PATH_MODEL_B, index=False)

    print("Data generation complete.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data (Model A) shape: {test_model_a_df.shape}")
    print(f"Test data (Model B) shape: {test_model_b_df.shape}")


# --- 3. Preprocessing Module ---

def preprocess_data(df, scaler=None, fit_scaler=False):
    """
    Preprocesses the data by separating features/target and scaling features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        scaler (StandardScaler, optional): A fitted scaler. Defaults to None.
        fit_scaler (bool): Whether to fit a new scaler.

    Returns:
        tuple: A tuple containing the processed features (X), target (y), and the scaler.
    """
    X = df.drop('rul', axis=1)
    y = df['rul']

    features = X.columns

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided if fit_scaler is False.")
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    return X_scaled_df, y, scaler


# --- 4. Training Module ---

def train_model():
    """
    Loads training data, preprocesses it, trains a model, and saves artifacts.
    """
    print("\nStarting model training...")

    # Load data
    df_train = pd.read_csv(TRAINING_DATA_PATH)

    # Preprocess data
    X_train, y_train, scaler = preprocess_data(df_train, fit_scaler=True)

    # Train model
    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)

    print(f"Model trained and saved to {MODEL_PATH}")


# --- 5. Evaluation Module ---

def evaluate_model():
    """
    Loads the trained model and evaluates it on the two test sets.
    """
    print("\nStarting model evaluation...")

    # Load model and scaler
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)

    # Load test data
    df_model_a = pd.read_csv(TESTING_DATA_PATH_MODEL_A)
    df_model_b = pd.read_csv(TESTING_DATA_PATH_MODEL_B)

    # Preprocess both test datasets
    X_model_a, y_model_a, _ = preprocess_data(df_model_a, scaler=scaler)
    X_model_b, y_model_b, _ = preprocess_data(df_model_b, scaler=scaler)

    # --- Evaluation on Test Data from Model A ---
    predictions_a = model.predict(X_model_a)
    mse_a = mean_squared_error(y_model_a, predictions_a)
    mae_a = mean_absolute_error(y_model_a, predictions_a)

    print("\n--- Evaluation on Test Set (Model A) ---")
    print(f"Mean Squared Error: {mse_a:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse_a):.4f}")
    print(f"Mean Absolute Error: {mae_a:.4f}")

    # --- Evaluation on Test Data from Model B ---
    predictions_b = model.predict(X_model_b)
    mse_b = mean_squared_error(y_model_b, predictions_b)
    mae_b = mean_absolute_error(y_model_b, predictions_b)

    print("\n--- Evaluation on Test Set (Model B) ---")
    print(f"Mean Squared Error: {mse_b:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse_b):.4f}")
    print(f"Mean Absolute Error: {mae_b:.4f}")


# --- 6. Main Pipeline Execution ---

def run_pipeline():
    """
    Orchestrates the pipeline: data generation, training, and evaluation.
    """
    # Step 1: Generate Datasets
    create_datasets()

    # Step 2: Train Model
    train_model()

    # Step 3: Evaluate Model
    evaluate_model()


if __name__ == '__main__':
    run_pipeline()