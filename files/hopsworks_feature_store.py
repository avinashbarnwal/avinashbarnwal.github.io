"""
Hopsworks Feature Store - End-to-end example
Covers: connection, feature groups, feature views, training datasets, online store
"""

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─── 1. Connect to Hopsworks ─────────────────────────────────────────────────

project = hopsworks.login(
    host="your-instance.hopsworks.ai",   # or set HOPSWORKS_HOST env var
    api_key_value="YOUR_API_KEY",        # or set HOPSWORKS_API_KEY env var
)

fs = project.get_feature_store()


# ─── 2. Create sample raw data ────────────────────────────────────────────────

def generate_transactions(n=1000):
    np.random.seed(42)
    now = datetime.utcnow()
    return pd.DataFrame({
        "transaction_id":  range(n),
        "customer_id":     np.random.randint(1, 200, n),
        "amount":          np.round(np.random.exponential(100, n), 2),
        "merchant_id":     np.random.randint(1, 50, n),
        "is_fraud":        np.random.binomial(1, 0.02, n),
        "event_time":      [now - timedelta(minutes=i) for i in range(n)],
    })

def generate_customer_profiles(n=200):
    return pd.DataFrame({
        "customer_id":      range(1, n + 1),
        "age":              np.random.randint(18, 80, n),
        "credit_score":     np.random.randint(300, 850, n),
        "account_age_days": np.random.randint(0, 3650, n),
        "country":          np.random.choice(["US", "UK", "DE", "FR", "IN"], n),
    })

transactions_df   = generate_transactions()
customer_df       = generate_customer_profiles()


# ─── 3. Feature engineering ───────────────────────────────────────────────────

def compute_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_log"]     = np.log1p(df["amount"])
    df["hour_of_day"]    = pd.to_datetime(df["event_time"]).dt.hour
    df["day_of_week"]    = pd.to_datetime(df["event_time"]).dt.dayofweek
    return df

def compute_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 7-day spend & transaction count per customer (simplified)."""
    agg = (
        df.groupby("customer_id")
          .agg(
              total_spend=("amount", "sum"),
              txn_count=("transaction_id", "count"),
              avg_txn_amount=("amount", "mean"),
              max_txn_amount=("amount", "max"),
          )
          .reset_index()
    )
    agg["spend_per_txn"] = agg["total_spend"] / agg["txn_count"]
    return agg

txn_features      = compute_transaction_features(transactions_df)
customer_agg      = compute_customer_aggregates(transactions_df)


# ─── 4. Create / upsert Feature Groups ───────────────────────────────────────

# --- Transaction feature group (batch, append-only) -------------------------
txn_fg = fs.get_or_create_feature_group(
    name="transactions",
    version=1,
    primary_key=["transaction_id"],
    event_time="event_time",
    description="Raw + engineered transaction features",
    online_enabled=True,
)
txn_fg.insert(txn_features, write_options={"wait_for_job": True})

# --- Customer profile feature group (SCD-style, upsert) ---------------------
customer_fg = fs.get_or_create_feature_group(
    name="customer_profiles",
    version=1,
    primary_key=["customer_id"],
    description="Static customer demographic features",
    online_enabled=True,
)
customer_fg.insert(customer_df, write_options={"wait_for_job": True})

# --- Customer aggregates feature group (batch) ------------------------------
customer_agg_fg = fs.get_or_create_feature_group(
    name="customer_aggregates",
    version=1,
    primary_key=["customer_id"],
    description="Rolling spend & frequency aggregates per customer",
    online_enabled=True,
)
customer_agg_fg.insert(customer_agg, write_options={"wait_for_job": True})


# ─── 5. Add feature descriptions (optional but good practice) ────────────────

txn_fg.update_feature_description("amount_log",    "log(1 + amount) to reduce skew")
txn_fg.update_feature_description("hour_of_day",   "UTC hour extracted from event_time")
customer_agg_fg.update_feature_description("spend_per_txn", "avg spend per transaction")


# ─── 6. Create a Feature View ────────────────────────────────────────────────

query = (
    txn_fg
    .select(["transaction_id", "customer_id", "amount_log",
             "hour_of_day", "day_of_week", "is_fraud"])
    .join(
        customer_fg.select(["credit_score", "account_age_days", "country"]),
        on="customer_id",
    )
    .join(
        customer_agg_fg.select(["total_spend", "txn_count",
                                "avg_txn_amount", "spend_per_txn"]),
        on="customer_id",
    )
)

feature_view = fs.get_or_create_feature_view(
    name="fraud_detection_fv",
    version=1,
    query=query,
    labels=["is_fraud"],
    description="Features for fraud detection model",
)


# ─── 7. Create Training Dataset ──────────────────────────────────────────────

version, job = feature_view.create_train_test_split(
    test_size=0.2,
    description="Initial fraud detection training split",
    data_format="csv",           # or "tfrecord", "parquet", "numpy"
    write_options={"wait_for_job": True},
)

X_train, X_test, y_train, y_test = feature_view.get_train_test_split(training_dataset_version=version)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ─── 8. Train a quick model ──────────────────────────────────────────────────

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Encode categorical
le = LabelEncoder()
X_train["country"] = le.fit_transform(X_train["country"])
X_test["country"]  = le.transform(X_test["country"])

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print(f"ROC-AUC: {auc:.4f}")


# ─── 9. Register model in Hopsworks Model Registry ───────────────────────────

import joblib, os

mr = project.get_model_registry()

model_dir = "/tmp/fraud_model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf, f"{model_dir}/model.pkl")

model = mr.sklearn.create_model(
    name="fraud_detector",
    version=1,
    metrics={"roc_auc": auc},
    description="GBM fraud detection model",
    feature_view=feature_view,         # links model → feature view
)
model.save(model_dir)
print(f"Model saved: {model.name} v{model.version}")


# ─── 10. Online inference (low-latency lookup) ───────────────────────────────

feature_view.init_serving(training_dataset_version=version)

# Fetch online features for a single transaction
sample_keys = [{"customer_id": 42, "transaction_id": 7}]
online_features = feature_view.get_feature_vectors(entry=sample_keys)
print("Online features:", online_features)


# ─── 11. Batch inference ─────────────────────────────────────────────────────

batch_data = feature_view.get_batch_data()  # returns all offline features
batch_data["country"] = le.transform(batch_data["country"])
batch_data["predicted_fraud"] = clf.predict(batch_data.drop(columns=["is_fraud"], errors="ignore"))
print(batch_data[["transaction_id", "predicted_fraud"]].head())


# ─── 12. Monitoring — compute feature statistics ─────────────────────────────

# Statistics are auto-computed on insert; retrieve them like this:
stats = txn_fg.get_feature_group_statistics()
print(stats)

# Or trigger manually:
txn_fg.compute_statistics()


# ─── 13. Cleanup helpers (optional) ──────────────────────────────────────────

def teardown():
    """Remove all objects created above — useful in CI/dev."""
    feature_view.delete_training_dataset(version=version)
    feature_view.delete()
    txn_fg.delete()
    customer_fg.delete()
    customer_agg_fg.delete()
    model.delete()
