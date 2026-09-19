"""
Monitoraggio continuo del sentiment su testi social reali, per il job
schedulato .github/workflows/monitor.yml (cron + trigger manuale).

I testi arrivano da Mastodon (timeline pubblica).

Ogni esecuzione:
1. scarica N post pubblici recenti da un'istanza Mastodon;
2. li classifica con il modello attuale (predictor.SentimentPredictor);
3. calcola la quota di sentiment negativo del batch;
4. la confronta con la baseline storica;
5. salva la coda di revisione (senza approvare etichette) e il risultato
   in monitoring/history.json, che il workflow ricommitta
   nel repository: senza questo passaggio ogni esecuzione ripartirebbe da
   zero, senza nessuna storia su cui costruire una baseline.
"""
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import requests

from config import MASTODON_INSTANCE, MAX_NEGATIVE_SHARE_INCREASE, MIN_HISTORY_FOR_BASELINE, N_POSTS
from predictor import SentimentPredictor
from review_data import enqueue
from config import MONITOR_KEYWORDS, REVIEW_CONFIDENCE_THRESHOLD, REVIEW_AUDIT_RATE

HISTORY_PATH = Path(__file__).parent / "monitoring" / "history.json"


def fetch_recent_posts(instance: str, limit: int, keywords=None) -> list:
    """
    Scarica gli ultimi `limit` post dalla timeline pubblica federata di Mastodon,
    che include contenuti pubblici provenienti sia dall'istanza corrente sia
    da altre istanze con cui essa comunica.
    Ripulisce HTML e tiene solo post esplicitamente marcati come inglesi,
    escludendo boost e applicando le eventuali parole chiave. Restituisce
    testo e metadati necessari alla revisione.
    """
    try:
        response = requests.get(
            f"{instance}/api/v1/timelines/public",
            params={"limit": min(limit, 40), "local": "false"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SystemExit(
            f"Impossibile leggere la timeline pubblica di {instance} ({exc}). "
            "Se l'istanza ha disattivato l'accesso anonimo, serve un access "
            "token Mastodon (gratuito) passato come header Authorization."
        ) from exc

    texts = []
    for post in response.json():
        if post.get("language") != "en" or post.get("reblog") is not None:
            continue
        text = re.sub(r"<[^>]+>", " ", post.get("content", ""))
        text = html.unescape(text).strip()
        if text and (not keywords or any(word.casefold() in text.casefold() for word in keywords)):
            texts.append({
                "post_id": post.get("uri") or f"{instance}/statuses/{post['id']}",
                "text": text, "url": post.get("url"), "instance": instance,
                "language": "en",
            })
    return texts


def load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def save_history(history: list) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def main() -> None:
    """
    Esegue un solo ciclo di monitoraggio (i 5 passaggi descritti in cima
    al file) e termina - pensata per essere lanciata una volta per
    esecuzione dal job schedulato (monitor.yml), non per girare in loop.
    """
    print(f"Scarico gli ultimi {N_POSTS} post pubblici da {MASTODON_INSTANCE}...")
    posts = fetch_recent_posts(MASTODON_INSTANCE, N_POSTS, MONITOR_KEYWORDS)
    if not posts:
        print("Nessun post inglese pertinente: nessun aggiornamento di storia o coda.")
        return
    texts = [post["text"] for post in posts]
    scope = "keywords:" + "|".join(sorted(MONITOR_KEYWORDS)) if MONITOR_KEYWORDS else "general_mastodon"
    print("Ambito monitorato:", scope)
    print(f"Testi utilizzabili (inglese, non vuoti): {len(texts)}")

    predictor = SentimentPredictor()
    results = [predictor.predict(text) for text in texts]
    added = enqueue(posts, results, predictor.model_name, scope,
                    REVIEW_CONFIDENCE_THRESHOLD, REVIEW_AUDIT_RATE)
    print(f"Coda revisione: {added} nuovi post, nessuna approvazione automatica.")
    predictions = [result["sentiment"].lower() for result in results]
    negative_share = predictions.count("negative") / len(predictions)
    print(f"Quota di sentiment negativo in questo batch: {negative_share:.2%}")

    # La baseline si calcola SOLO sulle esecuzioni precedenti, mai includendo
    # il batch corrente.
    history = load_history()
    # prende le settimane negative in history
    baseline_shares = [record["negative_share"] for record in history
                       if record.get("scope", "general_mastodon") == scope]

    if len(baseline_shares) >= MIN_HISTORY_FOR_BASELINE:
        baseline_mean = mean(baseline_shares) # calcola la media
        baseline_std = pstdev(baseline_shares) # calcolo della devazione standard
        statistical_threshold = baseline_mean + baseline_std # Soglia base
        # Incremento della quota negativa
        share_increase = negative_share - baseline_mean
        # ALERT STATISTICO: verifica se la quota negativa corrente è alta 
        # rispetto alla soglia base.
        statistical_alert = negative_share > statistical_threshold
        # ALERT BUSINESS: verifica se l'incremento della quota negativa
        # supera una soglia massima definita.
        business_alert = share_increase > MAX_NEGATIVE_SHARE_INCREASE

        print(
            f"Baseline storica ({len(baseline_shares)} esecuzioni precedenti): "
            f"media {baseline_mean:.2%}, soglia statistica {statistical_threshold:.2%}"
        )
        print(f"Alert statistico: {statistical_alert}")
        print(f"Alert di business (incremento > {MAX_NEGATIVE_SHARE_INCREASE:.0%}): {business_alert}")

        if statistical_alert or business_alert:
            # Annotazione riconosciuta da GitHub Actions: compare come warning
            # visibile nella pagina del job, non solo in mezzo ai log.
            print("::warning::Possibile picco di sentiment negativo rilevato nel monitoraggio continuo.")
    else:
        print(
            f"Storia insufficiente per calcolare una baseline "
            f"(servono almeno {MIN_HISTORY_FOR_BASELINE} esecuzioni precedenti, "
            f"disponibili {len(baseline_shares)})."
        )

    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_posts": len(predictions),
            "scope": scope,
            "negative_share": negative_share,
        }
    )
    save_history(history)
    print(f"\nStoria aggiornata: {len(history)} esecuzioni salvate in {HISTORY_PATH}.")


if __name__ == "__main__":
    main()
