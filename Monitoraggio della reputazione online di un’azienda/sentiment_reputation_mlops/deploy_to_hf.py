"""
Pubblica la cartella (app.py, predictor.py, requirements.txt, README.md)
come HuggingFace Space. Eseguito dal job "deploy" della pipeline CI/CD
(.github/workflows/ci.yml, alla radice del repository), dopo che i test sono
passati - non va lanciato a mano se non per debug locale.

Se lo Space indicato in SPACE_REPO_ID non esiste ancora, viene creato al primo
deploy (create_repo con exist_ok=True).
"""
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from config import SPACE_REPO_ID

FOLDER = Path(__file__).parent


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    try:
        api.create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        # 402 = HuggingFace richiede un abbonamento PRO per creare NUOVI Space
        # con SDK Gradio/Docker (anche su hardware gratuito) - non e' un bug di
        # questo script. Il job "deploy" ha `continue-on-error: true` in
        # ci.yml, quindi questo non blocca la pipeline; l'annotazione
        # "::error::" la rende comunque ben visibile (rossa) nel job invece di
        # un traceback da decifrare.
        if getattr(exc.response, "status_code", None) == 402:
            print(
                "::error::Deploy dello Space saltato: creare un nuovo Space con SDK "
                "Gradio/Docker richiede un abbonamento HuggingFace PRO su questo "
                "account (402 Payment Required). Non e' un problema del codice: "
                "il modello riaddestrato resta comunque pubblicato su HuggingFace "
                "tramite il repository modello (vedi train.py), e tutto il resto "
                "e' tracciato su GitHub. Per avere comunque una demo online "
                "gratuita, servirebbe riscriverla come Space 'static' che chiama "
                "la Inference API pubblica di HuggingFace invece di eseguire un "
                "server Gradio proprio - oppure sottoscrivere HuggingFace PRO "
                "(https://huggingface.co/pro)."
            )
            raise SystemExit(1) from exc
        raise

    api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        # Non serve pubblicare i test o la cache locale dentro lo Space.
        ignore_patterns=["tests/*", "__pycache__/*", "*.pyc", ".gitignore", "retraining_output/*"],
    )

    print(f"Deploy completato: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
