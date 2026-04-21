from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np
import json

app = FastAPI(title="BTC Price Predictor")

MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(os.getcwd(), 'models'))
SELECTION_FILE = os.path.join(MODEL_DIR, 'selection.json')


def load_models():
    # Determine model type
    model_type = os.getenv('MODEL_TYPE', None)
    if not model_type and os.path.exists(SELECTION_FILE):
        try:
            with open(SELECTION_FILE) as f:
                sel = json.load(f)
                model_type = sel.get('model')
        except Exception:
            model_type = None

    if not model_type:
        raise RuntimeError('No model selected. Set MODEL_TYPE env or run choose_best_model.py')

    model = None
    scaler = None
    if os.path.exists(os.path.join(MODEL_DIR, 'scaler.save')):
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.save'))

    if model_type.lower() in ('lstm', 'keras'):
        import tensorflow as tf
        mpath = os.path.join(MODEL_DIR, 'keras_model.h5')
        if not os.path.exists(mpath):
            raise RuntimeError(f'Model file not found: {mpath}')
        model = tf.keras.models.load_model(mpath)
    elif model_type.lower() in ('rf', 'random_forest', 'randomforest'):
        mpath = os.path.join(MODEL_DIR, 'rf_model.save')
        if not os.path.exists(mpath):
            raise RuntimeError(f'Model file not found: {mpath}')
        model = joblib.load(mpath)
    elif model_type.lower() in ('lr', 'linear', 'linear_regression'):
        mpath = os.path.join(MODEL_DIR, 'lr_model.save')
        if not os.path.exists(mpath):
            raise RuntimeError(f'Model file not found: {mpath}')
        model = joblib.load(mpath)
    else:
        raise RuntimeError(f'Unsupported MODEL_TYPE: {model_type}')

    return model_type.lower(), model, scaler


MODEL_TYPE_LOADED = None
MODEL_OBJ = None
SCALER = None


class SeriesIn(BaseModel):
    series: list  # list of floats (last SEQ_LEN values)


@app.on_event('startup')
def startup():
    global MODEL_TYPE_LOADED, MODEL_OBJ, SCALER
    MODEL_TYPE_LOADED, MODEL_OBJ, SCALER = load_models()


@app.get('/')
def root():
    return {'status': 'ok', 'model_serving': MODEL_TYPE_LOADED}


@app.post('/predict')
def predict(payload: SeriesIn):
    series = np.array(payload.series, dtype=float)
    if series.ndim != 1:
        raise HTTPException(status_code=400, detail='`series` must be a 1-D list of floats')

    # Expect either flat input for LR/RF (length=SEQ_LEN) or last sequence for LSTM
    if MODEL_TYPE_LOADED in ('lstm', 'keras'):
        # reshape to (1, timesteps, 1)
        x = series.reshape(1, -1, 1)
        pred = MODEL_OBJ.predict(x)
        if SCALER is not None:
            pred = SCALER.inverse_transform(pred.reshape(-1, 1)).flatten()[0]
        else:
            pred = float(pred.flatten()[0])
    else:
        x = series.reshape(1, -1)
        pred = MODEL_OBJ.predict(x)
        if SCALER is not None:
            pred = SCALER.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
        else:
            pred = float(np.array(pred).flatten()[0])

    return {'prediction': float(pred), 'model': MODEL_TYPE_LOADED}
