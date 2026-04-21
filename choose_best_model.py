"""
choose_best_model.py

Attempt to choose the best model from MLflow's sqlite tracking DB (mlflow/mlflow.db).
If found, writes `models/selection.json` with {'model': '<lstm|rf|lr>'} so the FastAPI app can load it.

Usage: python choose_best_model.py
"""
import sqlite3
import os
import json

DB_PATHS = [
    os.path.join('mlflow', 'mlflow.db'),
    os.path.join('notebooks', 'mlruns', 'mlflow.db'),
]

OUT_FILE = os.path.join('models', 'selection.json')


def find_db():
    for p in DB_PATHS:
        if os.path.exists(p):
            return p
    return None


def choose_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Try to find the best R² metric (higher is better)
        cur.execute("SELECT m.run_uuid, m.value, r.run_name FROM metrics m LEFT JOIN runs r ON m.run_uuid = r.run_uuid WHERE m.key='r2' ORDER BY m.value DESC LIMIT 1;")
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        run_uuid, val, run_name = row
        name = (run_name or '').lower()
        if 'lstm' in name:
            return 'lstm'
        if 'random' in name or 'rf' in name:
            return 'rf'
        if 'linear' in name or 'lr' in name:
            return 'lr'
        # fallback: infer from run_name words
        if 'keras' in name:
            return 'lstm'
        return None
    except Exception:
        return None


def main():
    db = find_db()
    if not db:
        print('No MLflow DB found. Please set MODEL_TYPE manually or ensure mlflow tracking DB is present.')
        return

    print(f'Using MLflow DB: {db}')
    model = choose_from_db(db)
    if not model:
        print('Could not determine best model from MLflow runs. Please set MODEL_TYPE manually.')
        return

    os.makedirs('models', exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump({'model': model}, f)

    print(f'Wrote models/selection.json -> {{"model": "{model}"}}')


if __name__ == '__main__':
    main()
