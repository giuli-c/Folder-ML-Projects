import pytest

from predictor import SentimentPredictor


@pytest.fixture(scope="module")
def predictor():
    """Carica il modello una sola volta e lo condivide tra tutti i test del modulo."""
    return SentimentPredictor()


def test_model_loads(predictor):
    """
    Non verifica che la predizione sia "giusta": verifica solo che la pipeline
    non si rompa e restituisca il formato atteso. Se fallisce, il problema e'
    nel codice o nelle dipendenze (import, nome del modello, versione di
    transformers), non nella qualita' del modello.
    """
    result = predictor.predict("I love this product")
    assert "sentiment" in result
    assert "confidence" in result


def test_known_examples(predictor):
    """
    Il test non serve a testare il modello, ma solo a far scattare un allarme se 
    qualcosa di grosso si rompe: un aggiornamento del modello su HuggingFace, 
    una libreria che cambia il formato di output, un bug nel codice che collega testo e predizione. 
    Su un caso ambiguo un fallimento sarebbe normale; su un caso ovvio, e' un segnale da non ignorare.
    """
    examples = {
        "I absolutely love this, best experience ever!": "positive",
        "This is the worst service I have ever received.": "negative",
    }
    for text, expected_label in examples.items():
        result = predictor.predict(text)
        assert result["sentiment"].lower() == expected_label, (
            f"Atteso '{expected_label}' per '{text}', ottenuto '{result['sentiment']}'"
        )
