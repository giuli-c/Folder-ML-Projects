from transformers import pipeline


class SentimentPredictor:
    """Carica il modello una sola volta ed espone un'unica funzione di predizione.

    Usata sia da app.py (demo Gradio) sia da tests/test_smoke.py, cosi' la logica
    di caricamento del modello e di normalizzazione dell'output vive in un solo
    posto invece di essere duplicata in due file diversi del repository.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self._pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    def predict(self, text: str) -> dict:
        result = self._pipeline(text)[0]
        return {
            "sentiment": result["label"],
            "confidence": round(float(result["score"]), 4),
        }
