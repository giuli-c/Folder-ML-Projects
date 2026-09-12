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

from config import SPACE_REPO_ID

FOLDER = Path(__file__).parent


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    api.create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

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
