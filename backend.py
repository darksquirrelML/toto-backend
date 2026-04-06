from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import time
import threading
import pickle
from supabase import create_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# ─── Config ───────────────────────────────────────────────────────────────────

import os
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "toto_model.pkl"

# ─── Global training progress ─────────────────────────────────────────────────
training_progress = {
    "step": 0,
    "total": 0,
    "percent": 0,
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
    url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
    response = requests.get(url, timeout=10)
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
    return draws

# ─── Train params ─────────────────────────────────────────────────────────────
class TrainParams(BaseModel):
    epochs: int = 100
    batchSize: int = 64
    trainRatio: float = 0.85
    windowSize: int = 15

# ─── Background training ──────────────────────────────────────────────────────
def do_training(params):
    global training_progress

    try:
        training_progress["status"] = "loading"
        training_progress["message"] = "Loading draws from Supabase..."
        print("Loading draws...", flush=True)

        draws = load_draws()
        print(f"Loaded {len(draws)} draws", flush=True)
        
        if not draws:
            training_progress["status"] = "error"
            training_progress["message"] = "No draws found"
            return

        print(f"Loaded {len(draws)} draws")
        data_X = draws_to_multihot(draws)
        window = params.windowSize

        training_progress["status"] = "preparing"
        training_progress["message"] = "Preparing sequences..."

        sequences, targets = [], []
        for i in range(len(data_X) - window):
            sequences.append(data_X[i:i + window].flatten())
            targets.append(data_X[i + window])

        sequences = np.array(sequences)
        targets = np.array(targets)

        print(f"Prepared {len(sequences)} sequences")

        training_progress["status"] = "training"
        training_progress["message"] = "Training Random Forest model..."
        training_progress["total"] = 49
        training_progress["step"] = 0

        start = time.time()

        # Train one classifier per number (49 total)
        models = []
        for i in range(49):
            clf = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=1,
                max_depth=10,
                min_samples_split=5
            )
            
            clf.fit(sequences, targets[:, i])
            models.append(clf)

            percent = int(((i + 1) / 49) * 100)
            training_progress.update({
                "step": i + 1,
                "total": 49,
                "percent": percent,
                "status": "training",
                "message": f"Training number {i+1}/49"
            })

            if (i + 1) % 10 == 0:
                print(f"Trained {i+1}/49 classifiers")

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)

        elapsed = time.time() - start
        print(f"Training complete in {elapsed:.1f}s")

        training_progress.update({
            "status": "complete",
            "percent": 100,
            "message": f"Training done in {elapsed:.1f}s"
        })

    except Exception as e:
        print(f"Training error: {e}")
        training_progress["status"] = "error"
        training_progress["message"] = str(e)

# ─── Train endpoint ───────────────────────────────────────────────────────────
@app.post("/train")
def train(params: TrainParams):
    global training_progress
    if training_progress.get("status") == "training":
        return {"status": "already_running", "message": "Training already in progress"}

    training_progress = {
        "step": 0, "total": 49, "percent": 0,
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

# ─── Predict params ───────────────────────────────────────────────────────────
class PredictParams(BaseModel):
    windowSize: int = 15
    mcSamples: int = 20
    lastNPriority: int = 10

# ─── Predict endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
def predict(params: PredictParams):
    if not os.path.exists(MODEL_PATH):
        return {"status": "error", "message": "No trained model found. Train first."}

    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)

    draws = load_draws()
    if not draws:
        return {"status": "error", "message": "No draws found"}

    data_X = draws_to_multihot(draws)
    window = params.windowSize
    last_seq = data_X[-window:].flatten().reshape(1, -1)

    # Get probabilities for each number
    probs = []
    for i, clf in enumerate(models):
        prob = clf.predict_proba(last_seq)[0]
        # prob[1] = probability of number appearing
        p = prob[1] if len(prob) > 1 else prob[0]
        probs.append((i + 1, float(p)))

    # Recent numbers priority
    recent = draws[-params.lastNPriority:]
    recent_numbers = set()
    for row in recent:
        nums = [int(n.strip()) for n in str(row["winning_no"]).split(",")]
        recent_numbers.update(nums)
        if row["additional_no"]:
            recent_numbers.add(int(row["additional_no"]))

    all_sorted = sorted(probs, key=lambda x: x[1], reverse=True)

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