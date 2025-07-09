import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
import joblib
import os
import time

from sqlalchemy.orm import Session
from sqlalchemy import select, inspect
from datetime import datetime, timedelta
from shared.db_conn import SessionLocal
from shared.db_dto import BusesData, ModelStatus, CEST 
from shared.db_utils import save_log

# --- Container startup delay ---
print("Container ml_class_bus_trainer is starting...")
time.sleep(60) # Give database time to start (uncomment if needed)

# --- Basic settings ---
MODEL_DIR = '/app/shared/classification/models/'
MODEL_SAVE_PATH_BINARY = os.path.join(MODEL_DIR, 'bus_binary_model_new.pkl')
MODEL_SAVE_PATH_MULTICLASS = os.path.join(MODEL_DIR, 'bus_multiclass_model_new.pkl')
MODEL_SAVE_PATH_REGRESSION = os.path.join(MODEL_DIR, 'bus_regression_model_new.pkl')
TRAINING_DATA_INTERVAL_DAYS = 90
TRAINING_CYCLE_INTERVAL_HOURS = 24

os.makedirs(MODEL_DIR, exist_ok=True)

# +--------------------------------------------------+
# |                GLOBAL COLUMN LISTS               |
# +--------------------------------------------------+

FEATURE_COLUMNS_BASE_BUS = [
    'speed', 'direction', 'stops_count', 
    'maximum_delay_seconds', 'minimum_delay_seconds', 
    'delay_variance_value', 'delay_standard_deviation', 'delay_range_seconds', 
    'stops_on_time_count', 'stops_arrived_early_count', 'stops_arrived_late_count', 
    'delay_trend', # This is likely the problematic one
    'delay_consistency_score', # This might also be problematic
    'on_time_stop_ratio', 
    'avg_positive_delay_seconds', 'avg_negative_delay_seconds', 
    'temperature', 'feelslike', 'humidity', 'wind_kph', 
    'precip_mm', 'cloud', 'visibility_km', 'uv_index', 'daylight', 
    'fine_particles_pm2_5', 'coarse_particles_pm10', 'carbon_monoxide_ppb', 
    'nitrogen_dioxide_ppb', 'ozone_ppb', 'sulfur_dioxide_ppb', 
    'cluster_id', 'cluster_prediction_success'
]

TARGET_CALCULATION_COLUMNS_BUS = ['average_delay_seconds']

# +--------------------------------------------------+
# |             HELPER FUNCTIONS FOR TRAINING        |
# +--------------------------------------------------+

def fetch_bus_data_for_training(session: Session, interval_days: int) -> pd.DataFrame:
    """
    Retrieves bus data from the database for a specified time interval,
    selecting only the necessary columns for training and target calculation.
    """
    start_time = datetime.now(CEST) - timedelta(days=interval_days)

    mapper = inspect(BusesData)
    columns_to_select_from_dto = []
    
    all_needed_columns = list(set(FEATURE_COLUMNS_BASE_BUS + TARGET_CALCULATION_COLUMNS_BUS))

    for col_name in all_needed_columns:
        if hasattr(BusesData, col_name):
            columns_to_select_from_dto.append(getattr(BusesData, col_name))
        else:
            save_log("class_bus_trainer", "warning", f"Column '{col_name}' defined as needed, but does not exist in BusesData DTO. Skipping.")

    if not columns_to_select_from_dto:
        save_log("class_bus_trainer", "error", "No columns to fetch from BusesData DTO. Check column definitions and DTO.")
        return pd.DataFrame()

    query_result = session.execute(
        select(*columns_to_select_from_dto)
        .filter(BusesData.timestamp >= start_time)
    ).all()

    df = pd.DataFrame(query_result, columns=[c.name for c in columns_to_select_from_dto])

    save_log("class_bus_trainer", "info", f"Fetched {len(df)} bus data records for training from the last {interval_days} days.")
    return df

def prepare_features_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]: # Return dict for all encoders
    """
    Prepares features and target variables for model training for bus data.
    Returns the processed DataFrame with features, target variables, and a dictionary of LabelEncoders.
    """
    
    df_processed = df.copy()

    if not all(col in df_processed.columns for col in TARGET_CALCULATION_COLUMNS_BUS):
        missing = [col for col in TARGET_CALCULATION_COLUMNS_BUS if col not in df_processed.columns]
        save_log("class_bus_trainer", "error", f"Missing key target columns {missing} in data. Cannot prepare targets.")
        raise ValueError(f"Required column(s) for target variables are missing: {missing}.")
    
    save_log("class_bus_trainer", "info", f"prepare_features_and_targets: Shape of df_processed after fetching: {df_processed.shape}")
    save_log("class_bus_trainer", "info", f"prepare_features_and_targets: Columns in df_processed: {df_processed.columns.tolist()}")

    initial_rows = len(df_processed)
    
    label_encoders_dict = {} # Dictionary to store all LabelEncoders

    # Handle missing values and identify columns for encoding
    columns_to_process = df_processed.columns.tolist() # Take a copy
    for col in columns_to_process:
        if col not in df_processed.columns: 
            continue
            
        save_log("class_bus_trainer", "debug", f"prepare_features_and_targets: Processing column '{col}' of type: {df_processed[col].dtype}")

        if df_processed[col].dtype in ['int64', 'float64']:
            if df_processed[col].isnull().any():
                mean_val = df_processed[col].mean()
                if pd.isna(mean_val):
                    df_processed[col] = df_processed[col].fillna(0)
                    save_log("class_bus_trainer", "warning", f"Column '{col}' contained only NaNs, filled with 0.")
                else:
                    df_processed[col] = df_processed[col].fillna(mean_val)
        elif df_processed[col].dtype == 'object':
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()
                if not mode_val.empty:
                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                else:
                    df_processed[col] = df_processed[col].fillna('unknown')
                    save_log("class_bus_trainer", "warning", f"String column '{col}' contained only NaNs, filled with 'unknown'.")

            # Handle non-numeric object columns that are NOT target columns
            # Ensure it's not 'average_delay_seconds' or any other target-related column if it was somehow object
            if col not in TARGET_CALCULATION_COLUMNS_BUS: # and col != 'daylight' (daylight is handled separately below)
                if col == 'daylight': # Specific handling for daylight if it's still an object
                     save_log("class_bus_trainer", "info", f"Encoding 'daylight' column (object type).")
                     le = LabelEncoder()
                     # It's good practice to ensure the encoder sees all possible labels,
                     # but for 'yes'/'no' it's simple.
                     # Handle potential unexpected values gracefully
                     known_daylight_labels = ['yes', 'no']
                     # Map any unexpected values to 'no' before encoding
                     df_processed['daylight'] = df_processed['daylight'].apply(lambda x: x if x in known_daylight_labels else 'no')
                     df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
                     label_encoders_dict[f'{col}_le'] = le
                     label_encoders_dict[f'{col}_categories'] = le.classes_.tolist()
                     df_processed = df_processed.drop(col, axis=1)
                elif df_processed[col].dtype == 'object': # Generic handling for other object columns
                    save_log("class_bus_trainer", "info", f"Encoding categorical column: '{col}'.")
                    le = LabelEncoder()
                    # Fit on unique values, handle potential new values during prediction later by mapping to unknown or most frequent
                    df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
                    label_encoders_dict[f'{col}_le'] = le
                    label_encoders_dict[f'{col}_categories'] = le.classes_.tolist()
                    df_processed = df_processed.drop(col, axis=1) # Drop original column
    
    # Check 'daylight' specifically one more time if it's already numeric.
    # This block ensures if it started as 'yes'/'no' and got encoded, it's handled.
    # If it came as 0/1, it's handled here.
    if 'daylight' in df_processed.columns and df_processed['daylight'].dtype in ['int64', 'float64']:
        # If it's already numeric (0/1), ensure it's named consistently
        df_processed['daylight_encoded'] = df_processed['daylight'].astype(int)
        df_processed = df_processed.drop('daylight', axis=1)
        save_log("class_bus_trainer", "info", "Daylight column is already numeric, renamed to 'daylight_encoded'.")


    df_processed.dropna(inplace=True) 

    if len(df_processed) < initial_rows:
        save_log("class_bus_trainer", "warning", f"Filled/Dropped {initial_rows - len(df_processed)} rows with missing data during feature preparation.")

    if df_processed.empty:
        save_log("class_bus_trainer", "error", "No data after removing rows with missing values. Cannot train model.")
        raise ValueError("No data after removing rows with missing values.")

    # Prepare target variables
    df_processed['average_delay_seconds_reg'] = df_processed['average_delay_seconds']

    # Binary classification: is_late (0 = on time, 1 = late)
    df_processed['is_late'] = (df_processed['average_delay_seconds'] > 0).astype(int)
    
    # Multiclass classification: delay_category (0-on_time, 1-slightly_late, 2-very_late)
    def categorize_delay_level(delay_seconds):
        if delay_seconds <= 0:
            return 0  # on time (early or on time)
        elif delay_seconds <= 300:
            return 1  # slightly late
        else:
            return 2  # very late
    
    df_processed['delay_category'] = df_processed['average_delay_seconds'].apply(categorize_delay_level)
    
    # Update feature columns for the final X DataFrame
    # Start with base numerical features that actually exist in df_processed
    final_feature_columns = [f for f in FEATURE_COLUMNS_BASE_BUS if f in df_processed.columns and df_processed[f].dtype != 'object']
    
    # Add encoded categorical features
    for col in FEATURE_COLUMNS_BASE_BUS:
        if f'{col}_encoded' in df_processed.columns and f'{col}_encoded' not in final_feature_columns:
            final_feature_columns.append(f'{col}_encoded')

    # Ensure all final_feature_columns exist in df_processed
    X = df_processed[[col for col in final_feature_columns if col in df_processed.columns]].copy()
    
    # Verify that X only contains numeric columns
    non_numeric_cols_in_X = X.select_dtypes(include='object').columns
    if not non_numeric_cols_in_X.empty:
        save_log("class_bus_trainer", "error", f"Found non-numeric columns in X after processing: {non_numeric_cols_in_X.tolist()}. This will cause errors during scaling/training.")
        # Optionally drop them or raise a more severe error
        raise TypeError(f"Non-numeric columns detected in features (X): {non_numeric_cols_in_X.tolist()}")


    # Target variables
    y_binary = df_processed['is_late']
    y_multiclass = df_processed['delay_category']
    y_regression = df_processed['average_delay_seconds_reg']
    
    save_log("class_bus_trainer", "info", f"Successfully prepared features and targets for training. Number of samples: {len(X)}.")
    save_log("class_bus_trainer", "info", f"Final feature columns for X: {X.columns.tolist()}")
    return X, y_binary, y_multiclass, y_regression, label_encoders_dict # Now returns the dictionary of encoders

def train_and_save_model(X: pd.DataFrame, y: pd.Series, model_type: str, model_path: str, label_encoders: dict = None) -> float:
    """
    Trains and saves a classification/regression model.
    Returns the quality metric (Accuracy for classification, RMSE for regression).
    """
    
    if len(X) < 2 or len(y) < 2:
        save_log("class_bus_trainer", "warning", f"Insufficient data for training model type: {model_type}. Requires min. 2 samples, available {len(X)}. Skipping training.")
        return 0.0

    if model_type in ['binary_classification', 'multiclass_classification'] and len(y.unique()) < 2:
        save_log("class_bus_trainer", "warning", f"Insufficient unique classes ({len(y.unique())}) for classification {model_type}. Requires min. 2. Skipping training.")
        return 0.0


    save_log("class_bus_trainer", "info", f"Starting training for model {model_type}...")

    # Data splitting
    if model_type in ['binary_classification', 'multiclass_classification'] and len(y.unique()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_model = None
    best_score = -float('inf') if model_type != 'regression' else float('inf') # Accuracy vs RMSE
    best_model_name = ""
    
    if model_type == 'binary_classification':
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
        }
        scoring_metric = 'accuracy'
        label_mapping = {0: "on_time", 1: "late"} # For saving in the model
    elif model_type == 'multiclass_classification':
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, multi_class='ovr')
        }
        scoring_metric = 'accuracy'
        label_mapping = {0: "on_time", 1: "slightly_late", 2: "very_late"} # For saving in the model
    elif model_type == 'regression':
        models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        scoring_metric = 'neg_mean_squared_error'
        label_mapping = None # No label_mapping for regression
    else:
        save_log("class_bus_trainer", "error", f"Unknown model type: {model_type}")
        return 0.0

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=scoring_metric)
            
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)

            if model_type in ['binary_classification', 'multiclass_classification']:
                test_score = accuracy_score(y_test, y_pred)
                save_log("class_bus_trainer", "info", f"{name} - CV Accuracy: {cv_scores.mean():.4f}, Test Accuracy: {test_score:.4f}")
                if test_score > best_score:
                    best_score = test_score
                    best_model = model
                    best_model_name = name
            elif model_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                save_log("class_bus_trainer", "info", f"{name} - CV RMSE: {np.sqrt(-cv_scores.mean()):.4f}, Test RMSE: {rmse:.4f}, Test R²: {r2:.4f}")
                if rmse < best_score: # For RMSE, we look for minimum
                    best_score = rmse
                    best_model = model
                    best_model_name = name
        except Exception as e:
            save_log("class_bus_trainer", "error", f"Error during training model {name} ({model_type}): {e}")
            continue

    if best_model is None:
        save_log("class_bus_trainer", "error", f"Failed to train any model for {model_type}.")
        return 0.0

    # Save the best model
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'model_type': model_type,
        'model_name': best_model_name,
        'label_encoders': label_encoders, # Save LabelEncoders
        'quality_metric': best_score,
        'label_mapping': label_mapping # Save textual mapping for classification
    }
    
    # Additional metrics to save based on model type
    if model_type in ['binary_classification', 'multiclass_classification']:
        model_data['test_accuracy'] = best_score
    elif model_type == 'regression':
        model_data['test_rmse'] = best_score
        model_data['test_r2'] = r2_score(y_test, best_model.predict(X_test_scaled))

    joblib.dump(model_data, model_path)
    save_log("class_bus_trainer", "info", f"Model '{best_model_name}' ({model_type}) saved to: {model_path} with quality metric: {best_score:.4f}.")
    
    return best_score

def update_model_status(session: Session, model_name: str, quality_metric: float, metric_type: str):
    """
    Updates the model status in the database.
    :param session: SQLAlchemy session to the database.
    :param model_name: Name of the model (e.g., 'buses_binary_classifier', 'buses_regression_predictor').
    :param quality_metric: Value of the model quality metric (e.g., accuracy, RMSE).
    :param metric_type: Type of metric ('accuracy', 'rmse').
    """
    try:
        model_status_entry = session.execute(
            select(ModelStatus).filter_by(model_name=model_name)
        ).scalar_one_or_none()

        current_version = 0
        if model_status_entry:
            current_version = model_status_entry.version if model_status_entry.version is not None else 0
            model_status_entry.is_new_model_available = True
            model_status_entry.last_updated = datetime.now(CEST)
            model_status_entry.quality_metric = float(quality_metric)
            model_status_entry.version = current_version + 1
            if hasattr(model_status_entry, 'metric_type'): 
                model_status_entry.metric_type = metric_type 
            save_log(f"trainer_{model_name.replace('_', '-')}", "info", 
                     f"Updated status of model '{model_name}' in database. "
                     f"New version: {model_status_entry.version}, Metric: {metric_type}={quality_metric:.4f}.")
        else:
            new_entry = ModelStatus(
                model_name=model_name,
                is_new_model_available=True,
                last_updated=datetime.now(CEST),
                quality_metric=float(quality_metric),
                version=1,
            )
            session.add(new_entry)
            save_log(f"trainer_{model_name.replace('_', '-')}", "warning", 
                     f"Created new status entry for model '{model_name}'. "
                     f"Version: 1, Metric: {metric_type}={quality_metric:.4f}.")
        
        session.commit()
    except Exception as e:
        session.rollback()
        save_log(f"trainer_{model_name.replace('_', '-')}", "error", 
                     f"Error updating status of model '{model_name}' in database: {e}")

# +--------------------------------------------------+
# |                MAIN TRAINING FUNCTION            |
# +--------------------------------------------------+

def run_bus_classification_training_cycle():
    """Main function to run the training cycle for bus classifiers."""
    save_log("class_bus_trainer", "info", "🚀 Starting training cycle for bus classifiers...")
    
    db_session = SessionLocal()
    try:
        # 1. Fetch training data
        save_log("class_bus_trainer", "info", f"Fetching bus data from database for the last {TRAINING_DATA_INTERVAL_DAYS} days...")
        bus_df = fetch_bus_data_for_training(db_session, TRAINING_DATA_INTERVAL_DAYS)
        
        if bus_df.empty:
            save_log("class_bus_trainer", "warning", "No bus data available for training in the specified interval. Skipping training.")
            return

        # 2. Prepare features and targets
        save_log("class_bus_trainer", "info", "Preparing features and targets for buses...")
        try:
            X, y_binary, y_multiclass, y_regression, label_encoders_dict = prepare_features_and_targets(bus_df)
        except ValueError as e:
            save_log("class_bus_trainer", "error", f"Error during data preparation: {e}. Skipping training.")
            return
        except TypeError as e: # Catch the new TypeError for non-numeric columns
            save_log("class_bus_trainer", "error", f"Data type error during feature preparation: {e}. Skipping training.")
            return


        if X.empty:
            save_log("class_bus_trainer", "warning", "No data after feature preparation. Skipping training.")
            return
        
        # Pass the dictionary of all label encoders to the training function
        # No need to extract daylight_le specifically, it's part of the dict now.

        # 3. Train and save models
        
        # Binary model
        binary_score = train_and_save_model(X, y_binary, 'binary_classification', MODEL_SAVE_PATH_BINARY, label_encoders_dict)
        update_model_status(db_session, 'buses_binary_classifier', binary_score, 'accuracy')

        # Multiclass model
        multiclass_score = train_and_save_model(X, y_multiclass, 'multiclass_classification', MODEL_SAVE_PATH_MULTICLASS, label_encoders_dict)
        update_model_status(db_session, 'buses_multiclass_classifier', multiclass_score, 'accuracy')
        
        # Regression model
        regression_score = train_and_save_model(X, y_regression, 'regression', MODEL_SAVE_PATH_REGRESSION, label_encoders_dict)
        update_model_status(db_session, 'buses_regression_predictor', regression_score, 'rmse')
        
        save_log("class_bus_trainer", "info", "✅ Bus classifier training cycle completed successfully.")

    except Exception as e:
        save_log("class_bus_trainer", "error", f"❌ Critical error in bus classifier training cycle: {e}")
    finally:
        db_session.close()


if __name__ == "__main__":
    while True:
        run_bus_classification_training_cycle()
        save_log("class_bus_trainer", "info", f"Next training cycle in {TRAINING_CYCLE_INTERVAL_HOURS} hours.")
        time.sleep(TRAINING_CYCLE_INTERVAL_HOURS * 3600)