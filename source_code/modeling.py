import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pathlib

def _predict_autoencoder(X: np.ndarray, model_dir: str, tag: str, thresh=None):
    ae: Model = load_model(os.path.join(model_dir, f"ae_{tag}.h5"), compile=False)
    recon  = ae.predict(X, verbose=0)
    errors = np.mean(np.square(X - recon), axis=1)
    if thresh is None:
        thresh = np.percentile(errors, 99)
    labels = (errors > thresh).astype(int)
    return errors, labels, thresh

def _build_autoencoder(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,))
    x   = Dense(64, activation="relu")(inp)
    x   = Dense(32, activation="relu")(x)
    enc = Dense(16, activation="relu")(x)
    x   = Dense(32, activation="relu")(enc)
    x   = Dense(64, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)
    ae  = Model(inp, out)
    ae.compile(optimizer=Adam(1e-3), loss="mse")
    return ae

def _train_autoencoder(X: np.ndarray, model_dir: str, tag: str) -> Model:
    ae = _build_autoencoder(X.shape[1])
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    ae.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0, callbacks=[es])
    ae.save(os.path.join(model_dir, f"ae_{tag}.h5"))
    return ae

def _predict_isolation_forest(X: np.ndarray, model_dir: str, tag: str):
    pipe: Pipeline = joblib.load(os.path.join(model_dir, f"iso_{tag}.pkl"))
    scores  = pipe.decision_function(X)
    labels  = (pipe.predict(X) == -1).astype(int)   # −1 ⇒ anomaly ⇒ 1
    return scores, labels

def _train_isolation_forest(X: np.ndarray, model_dir: str, tag: str) -> Pipeline:
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", IsolationForest(contamination=0.01,
                                                            random_state=42))]
    )
    pipe.fit(X)
    path_classifier = pathlib.Path(os.path.join(model_dir, f"iso_{tag}.pkl"))
    path_classifier.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, str(path_classifier))
    return pipe

def llm_modeling(
    embedding_df: pd.DataFrame,
    csv_in_path: str,
    path_models: str,
    model_tag: str,
) -> None:
    
    os.makedirs(path_models, exist_ok=True)

    X = embedding_df.values.astype(float)

    _train_isolation_forest(X, path_models, tag=model_tag)
    iso_scores, iso_labels = _predict_isolation_forest(X, path_models, tag=model_tag)

    _train_autoencoder(X, path_models, tag=model_tag)
    ae_scores, ae_labels, _ = _predict_autoencoder(X, path_models, tag=model_tag)

    df_orig = pd.read_csv(csv_in_path)
    df_orig["iso_score"] = iso_scores
    df_orig["iso_label"] = iso_labels
    df_orig["ae_score"]  = ae_scores
    df_orig["ae_label"]  = ae_labels
    df_orig["predicted_label"] = (df_orig["iso_label"] | df_orig["ae_label"]).astype(int)

    keep_cols = ["minute_ts", "imsi", "true_label",  
                "iso_score", "iso_label",
                "ae_score",  "ae_label",
                "predicted_label"]

    df_pred = df_orig[keep_cols]

    out_path = os.path.splitext(csv_in_path)[0] + "_predictions.csv"
    df_pred.to_csv(out_path, index=False)
    print("Predictions written at", out_path)