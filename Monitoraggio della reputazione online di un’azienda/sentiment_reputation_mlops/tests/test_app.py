"""
Test di integrazione per app.py: 
verifica che la funzione predict()
collegata a Gradio restituisca un risultato con lo schema corretto
(sentiment valido, confidence tra 0 e 1), e che l'interfaccia gr.Interface
venga costruita senza errori al momento dell'import.
"""
import app


def test_app_predict_has_valid_schema():
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
