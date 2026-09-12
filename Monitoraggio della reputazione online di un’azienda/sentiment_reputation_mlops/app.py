"""
Demo interattiva del modello: a differenza di tests/test_smoke.py eseguito
in automatico dalla CI, questo script va lanciato a mano (`python app.py`) 
e apre una pagina web dove una persona puo' scrivere un testo e vedere 
subito sentiment e confidence.
"""

import gradio as gr

from predictor import SentimentPredictor

predictor = SentimentPredictor()

def predict(text):
    """
    Funzione richiamata da Gradio ogni volta che l'utente invia un testo
    dalla pagina web.
    """
    return predictor.predict(text)

# gr.Interface costruisce automaticamente l'interfaccia web a partire dalla
# funzione: una casella di testo in input, un riquadro JSON in output con
# {"sentiment": ..., "confidence": ...}.
demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="json",
    title="Online Reputation Sentiment Monitor",
)

if __name__ == "__main__":
    # Avvia un server web locale con l'interfaccia sopra.
    demo.launch()
