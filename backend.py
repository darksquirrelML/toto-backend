from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import time
import threading
from supabase import create_client

# ─── Config ───────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://fcibqtbavrltcvzhfgjy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZjaWJxdGJhdnJsdGN2emhmZ2p5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk0MDIwNjMsImV4cCI6MjA4NDk3ODA2M30.dZCE7TpUZWHnT3vUvuviAZqfi9_MFwqkQHBk0RZNb9A"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "lstm_model.h5"

# ─── Global training progress ─────────────────────────────────────────────────
training_progress = {
    "epoch": 0,
    "total": 0,
    "loss": 0,
    "val_loss": 0,
    "percent": 0,
    "remaining": 0,
    "status": "idle",
    "message": ""
}

# ─── Load draws from Supabase ─────────────────────────────────────────────────
def load_draws(limit=2000):
    response = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=False) \
        .limit(limit) \
        .execute()
    return response.data

# ─── Convert draws to multihot ────────────────────────────────────────────────
def draws_to_multihot(draws):
    X = []
    for row in draws:
        v = np.zeros(49, dtype=np.float32)
        nums = [int(n.strip()) for n in str(row["winning_no"]).split(",")]
        for n in nums:
            v[n - 1] = 1.0
        if row["additional_no"]:
            v[int(row["additional_no"]) - 1] = 1.0
        X.append(v)
    return np.array(X)

# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "TOTO backend running"}

# ─── Scrape endpoint ──────────────────────────────────────────────────────────
@app.get("/scrape")
def scrape():
    try:
        url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("table tbody tr")
        draws = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                try:
                    draws.append({
                        "draw_no": int(cols[0].text.strip()),
                        "draw_date": cols[1].text.strip(),
                        "winning_no": cols[2].text.strip(),
                        "additional_no": cols[3].text.strip() or None
                    })
                except Exception:
                    continue
        if draws:
            supabase.table("toto_results").upsert(
                draws, on_conflict="draw_no"
            ).execute()
        print(f"Scraped {len(draws)} draws", flush=True)
        return draws
    except Exception as e:
        print(f"Scrape error: {e}", flush=True)
        return []

# ─── Train params ─────────────────────────────────────────────────────────────
class TrainParams(BaseModel):
    epochs: int = 500
    batchSize: int = 64
    trainRatio: float = 0.85
    windowSize: int = 15

# ─── Background training ──────────────────────────────────────────────────────
def do_training(params):
    global training_progress
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    try:
        training_progress["status"] = "loading"
        training_progress["message"] = "Loading draws from Supabase..."
        print("Loading draws...", flush=True)

        draws = load_draws()
        if not draws:
            training_progress["status"] = "error"
            training_progress["message"] = "No draws found"
            return

        print(f"Loaded {len(draws)} draws", flush=True)
        data_X = draws_to_multihot(draws)
        window = params.windowSize

        training_progress["status"] = "preparing"
        training_progress["message"] = "Preparing sequences..."

        sequences, targets = [], []
        for i in range(len(data_X) - window):
            sequences.append(data_X[i:i + window])
            targets.append(data_X[i + window])

        sequences = np.array(sequences)
        targets = np.array(targets)
        print(f"Prepared {len(sequences)} sequences", flush=True)

        training_progress["status"] = "building"
        training_progress["message"] = "Building LSTM model..."

        tf.random.set_seed(42)
        model = keras.Sequential([
            keras.layers.Input(shape=(window, 49)),
            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(49, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

        val_split = 1.0 - params.trainRatio
        start = time.time()
        total_epochs = params.epochs

        training_progress["status"] = "training"
        training_progress["total"] = total_epochs

        for ep in range(total_epochs):
            hist = model.fit(
                sequences, targets,
                epochs=1,
                batch_size=params.batchSize,
                validation_split=val_split,
                verbose=0
            )
            loss = float(hist.history['loss'][0])
            val_loss = float(hist.history.get('val_loss', [0])[0])
            elapsed = time.time() - start
            avg = elapsed / (ep + 1)
            remaining = avg * (total_epochs - (ep + 1))
            percent = int(((ep + 1) / total_epochs) * 100)

            training_progress.update({
                "epoch": ep + 1,
                "total": total_epochs,
                "loss": round(loss, 4),
                "val_loss": round(val_loss, 4),
                "percent": percent,
                "remaining": round(remaining, 1),
                "status": "training",
                "message": f"Epoch {ep+1}/{total_epochs}"
            })

            if (ep + 1) % 10 == 0:
                print(f"Epoch {ep+1}/{total_epochs} - loss: {loss:.4f} - ETA: {remaining:.1f}s", flush=True)

        model.save(MODEL_PATH)
        elapsed_total = time.time() - start
        print(f"Training complete in {elapsed_total:.1f}s", flush=True)

        training_progress.update({
            "status": "complete",
            "percent": 100,
            "message": f"Training done in {elapsed_total:.1f}s"
        })

    except Exception as e:
        print(f"Training error: {e}", flush=True)
        training_progress["status"] = "error"
        training_progress["message"] = str(e)

# ─── Train endpoint ───────────────────────────────────────────────────────────
@app.post("/train")
def train(params: TrainParams):
    global training_progress
    if training_progress.get("status") == "training":
        return {"status": "already_running", "message": "Training already in progress"}
    training_progress = {
        "epoch": 0, "total": 0, "loss": 0,
        "val_loss": 0, "percent": 0, "remaining": 0,
        "status": "starting", "message": "Starting..."
    }
    thread = threading.Thread(target=do_training, args=(params,))
    thread.daemon = True
    thread.start()
    return {"status": "started", "message": "Training started"}

# ─── Progress endpoint ────────────────────────────────────────────────────────
@app.get("/train/progress")
def get_progress():
    return training_progress

# ─── Reset endpoint ───────────────────────────────────────────────────────────
@app.get("/train/reset")
def reset_training():
    global training_progress
    training_progress = {
        "epoch": 0, "total": 0, "loss": 0,
        "val_loss": 0, "percent": 0, "remaining": 0,
        "status": "idle", "message": ""
    }
    return {"status": "ok", "message": "Training reset"}

# ─── Predict params ───────────────────────────────────────────────────────────
class PredictParams(BaseModel):
    windowSize: int = 15
    mcSamples: int = 20
    lastNPriority: int = 10

# ─── Predict endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
def predict(params: PredictParams):
    import tensorflow as tf
    from tensorflow import keras

    if not os.path.exists(MODEL_PATH):
        return {"status": "error", "message": "No trained model found. Train first."}

    model = keras.models.load_model(MODEL_PATH)
    draws = load_draws()
    if not draws:
        return {"status": "error", "message": "No draws found"}

    data_X = draws_to_multihot(draws)
    window = params.windowSize
    last_seq = data_X[-window:].reshape((1, window, 49)).astype(np.float32)

    probs_accum = np.zeros(49, dtype=np.float64)
    for _ in range(params.mcSamples):
        pred = model(last_seq, training=True).numpy().reshape(-1)
        probs_accum += pred
    avg_probs = probs_accum / params.mcSamples

    recent = draws[-params.lastNPriority:]
    recent_numbers = set()
    for row in recent:
        nums = [int(n.strip()) for n in str(row["winning_no"]).split(",")]
        recent_numbers.update(nums)
        if row["additional_no"]:
            recent_numbers.add(int(row["additional_no"]))

    all_sorted = sorted(
        [(i + 1, float(avg_probs[i])) for i in range(49)],
        key=lambda x: x[1], reverse=True
    )

    top7 = []
    for num, prob in all_sorted:
        if num in recent_numbers:
            top7.append((num, prob))
        if len(top7) == 7:
            break
    if len(top7) < 7:
        for num, prob in all_sorted:
            if num not in [x[0] for x in top7]:
                top7.append((num, prob))
            if len(top7) == 7:
                break

    table = [
        {
            "number": num,
            "prob": round(prob, 4),
            "recent": "Yes" if num in recent_numbers else "No"
        }
        for num, prob in top7
    ]
    predicted = [row["number"] for row in table]

    return {"predicted": predicted, "table": table}