"""
Punto di ingresso usato da .github/workflows/train-reviewed.yml (il workflow
scatta quando un umano aggiorna monitoring/review_queue.json con nuove
approvazioni - vedi review_data.py per come si arriva a quella coda).

Decide SE vale la pena avviare train.py sul dataset revisionato, e in caso
lo lancia come sotto-processo. "Vale la pena" vuol dire due cose insieme:
- readiness() dice che ci sono abbastanza esempi approvati per ogni
  classe/split (vedi review_data.py per le soglie);
- il dataset approvato e' diverso dall'ultimo tentativo gia' registrato -
  altrimenti si rilancerebbe lo stesso identico training ad ogni esecuzione
  schedulata, anche senza nessun dato nuovo da valutare.

Flag da riga di comando:
- --check: usato solo dal workflow per decidere se installare le dipendenze
  pesanti (transformers/torch) PRIMA di sapere se serviranno davvero. Scrive
  ready=true/false in GITHUB_OUTPUT e non esegue nessun training.
- --publish: autorizza train.py a pubblicare il modello su HuggingFace se
  supera i controlli di qualita'. Senza questo flag (uso locale o di test)
  train.py gira comunque per intero, ma con --no-push: utile per vedere se
  il training funzionerebbe senza pubblicare nulla per davvero.
- --force: ripete un training anche se il dataset approvato non e' cambiato
  dall'ultimo tentativo. Pensato per riprovare dopo aver cambiato a mano gli
  iperparametri qui sotto, non per rincorrere un errore di rete transitorio.

Ogni tentativo (superato o rifiutato) viene registrato in
monitoring/retraining_state.json insieme alla firma del dataset usato: e'
cosi' che l'esecuzione successiva sa se c'e' davvero qualcosa di nuovo da provare.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from review_data import QUEUE_PATH, readiness, dataset_fingerprint, training_budget

STATE_PATH = QUEUE_PATH.parent / "retraining_state.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Non allena nulla: scrive solo ready=true/false in GITHUB_OUTPUT.")
    parser.add_argument("--force", action="store_true", help="Riprova lo stesso dataset gia' elaborato.")
    parser.add_argument("--publish", action="store_true", help="Consenti pubblicazione del candidato SOLO se supera i gate.")
    args = parser.parse_args()
    ready, counts = readiness()
    fingerprint = dataset_fingerprint()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    changed = fingerprint != state.get("fingerprint")
    should_run = ready and (changed or args.force)
    print(json.dumps(counts, indent=2))
    print(f"Dati sufficienti: {ready}; dataset approvato cambiato: {changed}; avvio: {should_run}")
    if args.check:
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                stream.write(f"ready={str(should_run).lower()}\n")
        return
    if not should_run:
        return
    # Il budget vero e proprio (quanti testi nuovi + di replay chiedere a
    # train.py) lo calcola training_budget() in review_data.py, restringendosi
    # da solo se la classe approvata meno numerosa non ne ha abbastanza.
    n_train, n_replay, _ = training_budget(QUEUE_PATH)
    # Iperparametri più conservativi dei default "demo" di train.py (meno
    # epoche di rischio, learning rate più basso, patience più bassa): qui
    # il dataset arriva da revisioni umane reali, non da un dataset pubblico
    # generico, quindi si preferisce un aggiornamento piu' cauto del modello.
    command = [sys.executable, "-u", "train.py", "--reviewed-data", str(QUEUE_PATH),
               "--n-train", str(n_train), "--n-replay", str(n_replay),
               "--epochs", "3", "--learning-rate", "1e-6", "--patience", "2",
               "--report-dir", "retraining_output/human_review"]
    # --publish arriva solo dal job che ha davvero accesso a HF_TOKEN (il
    # workflow reale); senza, il training gira comunque per intero (utile per
    # verificarne la qualita'), ma non pubblica nulla su HuggingFace.
    if not args.publish:
        command.append("--no-push")
    code = subprocess.run(command, cwd=Path(__file__).parent, check=False).returncode
    STATE_PATH.write_text(json.dumps({
        "fingerprint": fingerprint, "attempted_at": datetime.now(timezone.utc).isoformat(),
        "return_code": code, "publication_requested": args.publish,
    }, indent=2) + "\n", encoding="utf-8")
    # Salviamo lo stato anche quando il training viene rifiutato (return code
    # diverso da zero): senza, dataset_fingerprint() non essendo cambiato, la
    # prossima esecuzione riproverebbe lo stesso identico training scartato.
    # Un fallimento TECNICO (bug, rete) va ritentato esplicitamente con
    # --force, non aggirato automaticamente da questa funzione.
    raise SystemExit(code)


if __name__ == "__main__":
    main()
