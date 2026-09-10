"""Demo interattiva del modello: a differenza di tests/test_smoke.py (eseguito
in automatico dalla CI, senza interfaccia, solo pass/fail), questo script va
lanciato a mano (`python app.py`) e apre una pagina web dove una persona puo'
scrivere un testo e vedere subito sentiment e confidence — la stessa idea
della demo Gradio opzionale nel notebook (sezione 14), ma pensata per girare
fuori dal notebook (es. come HuggingFace Space)."""

import gradio as gr

from predictor import SentimentPredictor

# Caricato una sola volta all'avvio dell'app, non ad ogni richiesta: se piu'
# persone usano la pagina, il modello resta in memoria invece di essere
# ricaricato ogni volta.
predictor = SentimentPredictor()


def predict(text):
    # Funzione richiamata da Gradio ogni volta che l'utente invia un testo
    # dalla pagina web.
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
