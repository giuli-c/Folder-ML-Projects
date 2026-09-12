"""
Configurazione centralizzata del repository: un solo punto da modificare
per costanti presenti in:
- predictor.py, 
- train.py, 
- monitor.py,
- deploy_to_hf.py 
"""

# Modello usato in produzione (predictor.py, app.py) e come punto di
# partenza per il retraining dimostrativo (train.py).
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# ---------- DEPLOY_TO_HF.PY
# HuggingFace Space di destinazione per il deploy.
SPACE_REPO_ID = "GiuliaC/sentiment-reputation-monitor"

# ---------- TRAIN.PY
# Dataset per il retraining dimostrativo 
RETRAIN_DATASET = "mteb/tweet_sentiment_extraction"
# Dataset di valutazione BENCHMARK
ORIGINAL_BENCHMARK_DATASET = "cardiffnlp/tweet_eval"
# Repo su cui train.py pubblica il modello riaddestrato - 
# solo se supera il controllo di regressione sul benchmark originale 
# (vedi REGRESSION_TOLERANCE sotto).
RETRAINED_MODEL_REPO_ID = "GiuliaC/sentiment-reputation-monitor-retrained"
# Calo massimo accettabile di f1_macro sul benchmark originale, oltre il
# quale il modello riaddestrato viene considerato peggiorativo e scartato
# (nessun push_to_hub, il job train fallisce esplicitamente).
REGRESSION_TOLERANCE = 0.02

# ---------- MONITOR.PY
MASTODON_INSTANCE = "https://mastodon.social"
N_POSTS = 40
MIN_HISTORY_FOR_BASELINE = 2
# Soglia di business condivisa concettualmente con cfg.max_negative_share_increase
# nel notebook (sezione 2): stesso valore, per restare confrontabili.
MAX_NEGATIVE_SHARE_INCREASE = 0.15
