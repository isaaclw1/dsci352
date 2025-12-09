"""
Starter File – Telco Customer Churn (Keras + Sklearn + Optional AWS)

This is a starter script for your final project.

Your high-level goals:

1. Load and clean the Telco Customer Churn dataset.
2. Build a preprocessing pipeline that:
   - Handles numeric and categorical variables.
   - Imputes missing values.
   - Scales numeric features and one-hot encodes categorical features.
3. Train several ML models (scikit-learn + Keras):
   - Compare their performance (AUC, accuracy).
   - Interpret which model performs best and why.
4. (Bonus) Build a lightweight artifact suitable for deployment in AWS Lambda:
   - Export only the parameters needed to recreate the preprocessing + a simple model.
   - Upload this artifact to S3 if AWS export is enabled.

IMPORTANT:
- All places marked with "TODO (Student)" or "TODO" are for you to implement.
- As given, this file will raise NotImplementedError in several places.
  Your job is to replace those with working code. Disclaimer: this piece was AI generated.

Files:
- Input CSV: WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import os
import io
import json

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from pathlib import Path

# telecom_churn root directory (one level above scripts/)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Input CSV path
LOCAL_CSV = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# AWS Config (for Bonus)
S3_BUCKET = os.environ.get("MODEL_BUCKET", "769341382710-model")
S3_KEY_LIGHT = os.environ.get("MODEL_KEY", "models/telco_churn_light.json")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Flag: make AWS completely optional
#   - False: run everything locally; no S3 upload
#   - True : in addition, build a lightweight artifact and upload to S3 (bonus)
ENABLE_AWS_EXPORT = False


# Data Loading and helpers (given to you)
def load_telco(path: str) -> pd.DataFrame:
    """
    Load the Telco CSV and perform basic cleaning:
      - Convert TotalCharges to numeric (coerce errors to NaN).
      - Convert Churn to 0/1.
      - Drop customerID (not useful as a feature).

    You can inspect and modify this function if you want, but it's provided
    as a starting point.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"], errors="ignore")
    return df


def build_preprocessor(df: pd.DataFrame):
    """
    Build a ColumnTransformer that:
      - Applies a numeric pipeline to numeric columns.
      - Applies a categorical pipeline to categorical columns.

    Numeric pipeline:
      - SimpleImputer(strategy="median")
      - StandardScaler()

    Categorical pipeline:
      - SimpleImputer(strategy="most_frequent")
      - OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    Returns:
      preprocessor, X, y, numeric_feature_names, categorical_feature_names
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, X, y, numeric_features, categorical_features


def make_numpy_for_keras(preprocessor, X_train, X_val):
    """
    Fit the preprocessor on X_train, transform train+val, and return dense
    numpy arrays so that Keras can use them.

    Also returns feature_names for inspection (not required for training).
    """
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out()
    num_names = preprocessor.transformers_[0][2]
    feature_names = np.concatenate([num_names, cat_names])

    return X_train_proc.astype("float32"), X_val_proc.astype("float32"), feature_names


class GradientTracker(keras.callbacks.Callback):
    """
    Tracks gradient statistics after each epoch for a Keras model.

    This helps you reason about gradient health
    (e.g. vanishing/exploding gradients).
    """

    def __init__(self, model, sample_x, sample_y):
        super().__init__()
        self.model_for_grads = model
        self.sample_x = tf.convert_to_tensor(sample_x)
        self.sample_y = tf.convert_to_tensor(sample_y, dtype=tf.float32)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            preds = self.model_for_grads(self.sample_x, training=True)  # (batch, 1)
            preds = tf.squeeze(preds, axis=-1)  # -> (batch,)
            loss = keras.losses.binary_crossentropy(self.sample_y, preds)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model_for_grads.trainable_weights)
        grad_vals = [tf.reshape(g, [-1]) for g in grads if g is not None]
        if grad_vals:
            all_grads = tf.concat(grad_vals, axis=0)
            grad_mean = tf.reduce_mean(tf.abs(all_grads)).numpy().item()
            grad_max = tf.reduce_max(tf.abs(all_grads)).numpy().item()
            grad_std = tf.math.reduce_std(all_grads).numpy().item()
        else:
            grad_mean = grad_max = grad_std = 0.0

        self.history.append(
            {
                "epoch": int(epoch),
                "grad_abs_mean": grad_mean,
                "grad_abs_max": grad_max,
                "grad_std": grad_std,
                "loss": float(loss.numpy()),
                "logs": logs or {},
            }
        )


# ##################
# Keras Model – TODO
def build_keras_model(input_dim: int):
    """
    TODO:
    Build and compile a Keras model that takes as input a vector of length `input_dim`
    (the preprocessed features) and predicts churn probability.

    Requirements / suggestions:
      - Use a Sequential model starting with layers.Input(shape=(input_dim,))
      - Add several Dense layers with ReLU activation, optionally with Dropout.
      - Final layer: Dense(1, activation="sigmoid") (binary classification).
      - Compile with:
          optimizer = Adam(learning_rate=1e-3) (or similar)
          loss      = "binary_crossentropy"
          metrics   = [AUC, "accuracy"]

    Return:
      - a compiled Keras model ready for training.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout_1'),

        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_2'),

        layers.Dense(32, activation='relu', name='dense_3'),
        layers.Dropout(0.2, name='dropout_3'),

        layers.Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            'accuracy'
        ]
    )

    return model

############################
# Scikit-Learn Models – TODO
def train_sklearn_models(preprocessor, X, y):
    """
    TODO (Student):
    1. Split X, y into train and validation sets:
         X_train, X_val, y_train, y_val = train_test_split(
             X, y, test_size=0.2, random_state=42, stratify=y
         )

    2. Build and train several scikit-learn models using a Pipeline that
       includes the `preprocessor` plus a classifier, for example:
         - LogisticRegression(max_iter=500)
         - RandomForestClassifier(...)
         - GradientBoostingClassifier(...)
         - SGDClassifier(loss="log_loss", ...)

       For each model:
         - Fit on X_train, y_train.
         - Use predict_proba on X_val to get positive-class probabilities.
         - Compute validation AUC via roc_auc_score(y_val, proba).
         - Store results in a list as tuples:
               ("model_name_string", trained_pipeline, auc_value)

    3. Sort the results list by AUC (descending), so that results[0]
       is your best sklearn model.

    4. For the AWS lightweight artifact, pick ONE "simple" model that
       is Lambda-friendly (for example, an SGDClassifier with log_loss).
       Return:
         results, (X_train, X_val, y_train, y_val), chosen_sgd_pipeline, auc_of_sgd

    Returns:
      - results: list of (name, model_pipeline, auc)
      - splits: (X_train, X_val, y_train, y_val)
      - sgd_model: the chosen SGD-based pipeline for deployment
      - sgd_auc: AUC of that SGD model
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models_to_train = [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=500, random_state=42)
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        ),
        (
            "SGD Classifier",
            SGDClassifier(
                loss='log_loss',
                max_iter=1000,
                tol=1e-3,
                random_state=42
            )
        )
    ]

    results = []
    sgd_model = None
    sgd_auc = 0.0

    for model_name, classifier in models_to_train:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', classifier)
        ])

        pipeline.fit(X_train, y_train)

        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        y_val_pred = pipeline.predict(X_val)

        auc_score = roc_auc_score(y_val, y_val_proba)
        acc_score = accuracy_score(y_val, y_val_pred)

        results.append((model_name, pipeline, auc_score, acc_score))

        if "SGD" in model_name:
            sgd_model = pipeline
            sgd_auc = auc_score

    results.sort(key=lambda x: x[2], reverse=True)

    splits = (X_train, X_val, y_train, y_val)

    return results, splits, sgd_model, sgd_auc


################################################
# 4) Lightweight Artifact for AWS – TODO (Bonus)
def build_lightweight_artifact(
    df,
    preprocessor,
    numeric_features,
    categorical_features,
    sgd_model,
):
    """
    TODO (Bonus):
    Build a lightweight artifact (a plain Python dict) that contains JUST enough
    information to reconstruct the preprocessing + SGD classifier inside
    an AWS Lambda function (without sklearn).

    Suggested steps:

    1. Fit `preprocessor` on the full feature set X_full (df.drop("Churn", axis=1)).
       This ensures the imputer, scaler, and OneHotEncoder see all categories.

    2. Extract numeric imputer and scaler parameters:
         num_pipe = preprocessor.named_transformers_["num"]
         imputer  = num_pipe.named_steps["imputer"]
         scaler   = num_pipe.named_steps["scaler"]

       Save:
         - numeric_medians = imputer.statistics_.tolist()
         - numeric_means   = scaler.mean_.tolist()
         - numeric_scales  = scaler.scale_.tolist()

    3. Extract categorical one-hot encoder info:
         cat_pipe = preprocessor.named_transformers_["cat"]
         ohe      = cat_pipe.named_steps["onehot"]

       Save:
         - categories for each categorical feature: [c.tolist() for c in ohe.categories_]

    4. Extract SGD classifier coefficients and intercept:
         clf = sgd_model.named_steps["clf"]
         coef = clf.coef_.tolist()
         intercept = clf.intercept_.tolist()

    5. Construct and return a dict like:

       artifact = {
         "numeric_features": numeric_features,
         "numeric_imputer_medians": numeric_medians,
         "numeric_means": numeric_means,
         "numeric_scales": numeric_scales,
         "categorical_features": categorical_features,
         "categories": [...],
         "coef": coef,
         "intercept": intercept,
       }

    This dict should be fully JSON-serializable.
    """
    raise NotImplementedError("TODO (Bonus): implement build_lightweight_artifact(...).")


#######################
# AWS Upload (given)
def upload_json_to_s3(obj, bucket, key, region="us-east-1"):
    """
    Upload a JSON-serializable object to S3. Used only if ENABLE_AWS_EXPORT is True.
    """
    s3 = boto3.client("s3", region_name=region)
    body = json.dumps(obj).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Uploaded to s3://{bucket}/{key}")


def main():
    # Step 0: Load data
    if not os.path.exists(LOCAL_CSV):
        raise FileNotFoundError(f"CSV not found at {LOCAL_CSV}")

    print("Loading telco data...")
    df = load_telco(LOCAL_CSV)

    # Step 1: Preprocessor
    print("Building preprocessor...")
    preprocessor, X, y, num_feats, cat_feats = build_preprocessor(df)

    # Step 2: Scikit-Learn models
    print("Training sklearn models...")
    sk_results, splits, sgd_model, sgd_auc = train_sklearn_models(preprocessor, X, y)
    (X_train, X_val, y_train, y_val) = splits

    # sk_results should be sorted by AUC (descending)
    best_sklearn_name, best_sklearn_model, best_sklearn_auc, best_sklearn_acc = sk_results[0]

    # Step 3: Keras model
    print("Preparing numpy for Keras...")
    X_train_np, X_val_np, feature_names = make_numpy_for_keras(
        preprocessor, X_train, X_val
    )

    print("Building Keras model...")
    keras_model = build_keras_model(input_dim=X_train_np.shape[1])

    # pick a small fixed batch for gradient tracking
    sample_x = X_train_np[:256]
    sample_y = y_train.values[:256]
    grad_tracker = GradientTracker(keras_model, sample_x, sample_y)

    print("Training Keras model...")
    history = keras_model.fit(
        X_train_np,
        y_train.values,
        validation_data=(X_val_np, y_val.values),
        epochs=20,          # you can tune
        batch_size=256,     # you can tune
        callbacks=[grad_tracker],
        verbose=1,
    )

    # make sure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save gradient history locally for inspection
    with open(OUTPUTS_DIR / "keras_gradients.json", "w") as f:
        json.dump(grad_tracker.history, f, indent=2)
    print(f"Saved gradient stats to {OUTPUTS_DIR / 'keras_gradients.json'}")

    # Step 4: Evaluate Keras on validation set
    y_val_pred = keras_model.predict(X_val_np).ravel()
    keras_auc = roc_auc_score(y_val.values, y_val_pred)
    keras_acc = accuracy_score(y_val.values, (y_val_pred >= 0.5).astype(int))

    # Example: simple gradient quality score (for analysis)
    last_grad = grad_tracker.history[-1]
    grad_quality = {
        "last_epoch_grad_abs_mean": last_grad["grad_abs_mean"],
        "last_epoch_grad_abs_max": last_grad["grad_abs_max"],
        "last_epoch_grad_std": last_grad["grad_std"],
    }

    print("\n=== MODEL SUMMARY (local) ===")
    print(f"Best sklearn: {best_sklearn_name} AUC={best_sklearn_auc:.4f} ACC={best_sklearn_acc:.4f}")
    print(f"Keras model : AUC={keras_auc:.4f} ACC={keras_acc:.4f}")
    print(f"SGD (for potential Lambda): AUC={sgd_auc:.4f}")
    print(f"Gradient quality (keras): {grad_quality}")

    # Step 5: Choose a final model for reporting (policy is up to you)
    """
    TODO:
    Decide on a *model selection policy* for your report.
    For example:
      - If Keras is within 0.01 AUC of the best sklearn and has "healthy" gradients,
        pick Keras as your final model.
      - Otherwise, pick the best sklearn model.

    Set:
      final_model_name = ...
      final_auc        = ...
    """

    if keras_auc >= best_sklearn_auc - 0.01:
        final_model_name = "keras_mlp"
        final_auc = keras_auc
    else:
        final_model_name = best_sklearn_name
        final_auc = best_sklearn_auc

    print(f"\nFINAL CHOICE (for your report): {final_model_name} AUC={final_auc:.4f}")

    # Step 6: Build leaderboard (for analysis/report)
    model_leaderboard = []
    for name, model_obj, auc_val, acc_val in sk_results:
        model_leaderboard.append(
            {
                "name": name,
                "type": "sklearn",
                "auc": float(auc_val),
                "accuracy": float(acc_val),
            }
        )

    # Add Keras to leaderboard
    model_leaderboard.append(
        {
            "name": "keras_mlp",
            "type": "keras",
            "auc": float(keras_auc),
            "accuracy": float(keras_acc),
            "grad_abs_mean": float(grad_tracker.history[-1]["grad_abs_mean"]),
            "grad_abs_max": float(grad_tracker.history[-1]["grad_abs_max"]),
            "grad_std": float(grad_tracker.history[-1]["grad_std"]),
        }
    )

    # At this point you can:
    #   - Save model_leaderboard to disk as JSON or CSV for inspection.
    #   - Create plots (e.g., bar charts of AUC/accuracy) in a separate script/notebook.
    with open(OUTPUTS_DIR / "model_leaderboard_telco.json", "w") as f:
        json.dump(model_leaderboard, f, indent=2)
    print(f"Saved model_leaderboard_telco.json to {OUTPUTS_DIR}")
    # Step 7 (Bonus): Build lightweight artifact for AWS Lambda
    if ENABLE_AWS_EXPORT:
        print("\nENABLE_AWS_EXPORT=True → building lightweight artifact and uploading to S3.")

        artifact = build_lightweight_artifact(
            df,
            preprocessor,
            num_feats,
            cat_feats,
            sgd_model,
        )

        wrapped_artifact = {
            "artifact": artifact,
            "training_meta": {
                "final_choice": final_model_name,
                "final_choice_auc": float(final_auc),
                "best_sklearn": best_sklearn_name,
                "best_sklearn_auc": float(best_sklearn_auc),
                "keras_auc": float(keras_auc),
                "keras_acc": float(keras_acc),
                "model_leaderboard": model_leaderboard,
            },
        }

        upload_json_to_s3(wrapped_artifact, S3_BUCKET, S3_KEY_LIGHT, AWS_REGION)
        print("Uploaded lightweight artifact with metadata to S3.")
    else:
        print(
            "\nENABLE_AWS_EXPORT=False → skipping S3 upload.\n"
            "You can enable AWS export later by:\n"
            "  1) Setting MODEL_BUCKET and MODEL_KEY environment variables.\n"
            "  2) Setting ENABLE_AWS_EXPORT = True near the top of this file.\n"
            "  3) Implementing build_lightweight_artifact(...).\n"
        )


if __name__ == "__main__":
    main()
