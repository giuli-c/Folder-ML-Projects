"""Test di applicazione e predizioni: schema, casi limite e interfaccia Gradio."""
import pytest
from predictor import SentimentPredictor
import app


# ======================================================================
# Predizioni del modello e casi limite
# ======================================================================

VALID_LABELS = {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def predictor():
    """
    Carica il modello una sola volta e lo condivide tra tutti i test del modulo.
    """
    return SentimentPredictor()


def _assert_valid_schema(result: dict) -> None:
    """
    Contratto di output: qualunque sia il testo in ingresso, il predictor
    deve restituire sempre queste due chiavi, con un'etichetta tra le tre
    valide e una confidence in [0, 1] - indipendentemente da quanto la
    predizione sia "giusta".
    """
    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"].lower() in VALID_LABELS, (
        f"Etichetta inattesa: '{result['sentiment']}' (attese: {VALID_LABELS})"
    )
    assert 0.0 <= result["confidence"] <= 1.0, f"Confidence fuori range: {result['confidence']}"


def test_model_loads(predictor):
    """
    Non verifica che la predizione sia "giusta": verifica solo che la pipeline
    non si rompa e restituisca il formato atteso. Se fallisce, il problema e'
    nel codice o nelle dipendenze (import, nome del modello, versione di
    transformers), non nella qualita' del modello.
    """
    result = predictor.predict("I love this product")
    _assert_valid_schema(result)


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
        _assert_valid_schema(result)
        assert result["sentiment"].lower() == expected_label, (
            f"Atteso '{expected_label}' per '{text}', ottenuto '{result['sentiment']}'"
        )


@pytest.mark.parametrize(
    "text",
    [
        "a",  # testo minimo non vuoto
        "Great! " * 200,  # ben oltre i 128 token di truncation usati in produzione
        "Ciao, questo prodotto e' fantastico!",  # lingua diversa dall'inglese
        "🔥🔥🔥 amazing!!! 😍😍 #bestday",  # emoji e punteggiatura ripetuta
    ],
)
def test_edge_cases_do_not_crash(predictor, text):
    """
    Il modello in produzione ricevera' testi social reali, non solo frasi
    pulite in inglese: testi cortissimi, molto lunghi, in altre lingue o pieni
    di emoji devono comunque produrre un risultato con lo schema corretto,
    senza sollevare eccezioni - anche se la predizione nel merito puo' essere
    sbagliata (non e' quello che questo test verifica).
    """
    result = predictor.predict(text)
    _assert_valid_schema(result)


def test_empty_string_does_not_crash(predictor):
    """
    Caso limite separato dagli altri: una stringa vuota e' un input degenere
    che potrebbe comportarsi diversamente (es. tokenizzazione) rispetto a un
    testo normale molto corto.
    """
    result = predictor.predict("")
    _assert_valid_schema(result)


# ======================================================================
# Applicazione Gradio
# ======================================================================

def test_app_predict_has_valid_schema():
    """
    app.predict() è la funzione che Gradio chiama davvero ad ogni invio
    dell'utente (vedi gr.Interface(fn=predict, ...) in app.py): verifica
    che anche passando da lì - non solo chiamando SentimentPredictor
    direttamente come negli altri test - lo schema di output resti valido.
    """
    result = app.predict("I love this product")
    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"].lower() in {"negative", "neutral", "positive"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_gradio_interface_is_built():
    """
    Verifica che l'oggetto gr.Interface sia stato costruito correttamente
    a import-time (stesso oggetto che HuggingFace Space lancia in produzione),
    senza dover avviare un vero server (demo.launch() resta dietro
    if __name__ == "__main__", mai eseguito in fase di test).
    """
    assert app.demo is not None
