"""
Completa a mano il flusso di approvazione della coda di revisione umana.

Uso tipico:
1. Apri monitoring/review_queue.json e, per le righe "review_status": "pending"
   che vuoi approvare, scrivi SOLO validated_label a mano (una delle tre
   classi: "negative"/"neutral"/"positive") - è l'unica decisione che deve
   restare davvero umana. Per scartare un post invece, imposta tu stesso
   "review_status": "excluded" (questo script lo lascia stare).
2. Lancia questo script:

       python approve_reviewed.py

   Per ogni riga con validated_label ormai compilato ma ancora
   "review_status": "pending", completa da solo i tre campi di corredo che
   approved_splits() (in review_data.py) richiede per accettarla nel
   training: "review_status" -> "approved", "reviewer" e "reviewed_at".
   Non li devi più scrivere a mano riga per riga.

Questo script NON decide mai un'etichetta al posto tuo: si limita a
completare la bookkeeping di una decisione che hai già preso scrivendo
validated_label. review_is_simulated resta False (lo imposta già enqueue()
in review_data.py): questo script è pensato per revisioni reali, non per
generare dati fittizi - per quello ci sono i test in tests/test_data.py.
"""
import argparse
from datetime import datetime, timezone

from review_data import LABELS, QUEUE_PATH, load_queue, save_queue


def main(reviewer: str, path=QUEUE_PATH) -> None:
    rows = load_queue(path)
    updated = 0
    for row in rows:
        if row.get("review_status") == "pending" and row.get("validated_label") in LABELS:
            row["review_status"] = "approved"
            row["reviewer"] = reviewer
            row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1
    save_queue(rows, path)
    print(f"Righe completate (pending -> approved): {updated}. Coda salvata in {path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviewer", default="Giulia", help="Nome da registrare come revisore (default: Giulia).")
    args = parser.parse_args()
    main(reviewer=args.reviewer)
