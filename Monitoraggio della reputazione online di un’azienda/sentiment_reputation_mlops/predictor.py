from transformers import pipeline

from config import MODEL_NAME


class SentimentPredictor:
    """
    Carica il modello una sola volta ed espone un'unica funzione di predizione.

    Usata sia da app.py (demo Gradio) sia da tests/test_app.py, cosi' la logica
    di caricamento del modello e di normalizzazione dell'output vive in un solo
    posto invece di essere duplicata in due file diversi del repository.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        # truncation/max_length come nelle pipeline di train.py: senza,
        # un testo abbastanza lungo (oltre le 514 posizioni supportate da
        # RoBERTa) manda in crash il modello invece di essere troncato.
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=128,
        )

    def predict(self, text: str) -> dict:
        result = self._pipeline(text)[0]
        return {
            "sentiment": result["label"],
            "confidence": round(float(result["score"]), 4),
        }
