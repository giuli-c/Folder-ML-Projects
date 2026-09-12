"""
Training/retraining eseguibile da riga di comando, per il job "train" della
pipeline CI/CD (.github/workflows/train.yml, trigger manuale workflow_dispatch).

A differenza della valutazione principale del notebook (sezioni 1-9, sul test
set di cardiffnlp/tweet_eval), qui il fine-tuning avviene su un dataset
DIVERSO: mteb/tweet_sentiment_extraction (fonte: competizione Kaggle 2020, non
TweetEval). Il motivo: cardiffnlp/twitter-roberta-base-sentiment-latest e'
stato originariamente fine-tuned proprio su TweetEval per il task di sentiment
(vedi model card HuggingFace) - "riaddestrare" sullo stesso benchmark non
introdurrebbe nessuna informazione nuova al modello. Il nuovo dataset usa lo
stesso schema di etichette (0=negative, 1=neutral, 2=positive), quindi nessun
remapping aggiuntivo e' necessario.

Il confronto prima/dopo viene fatto su due test set separati:
- il test set del dataset NUOVO (tweet_sentiment_extraction): misura se il
  fine-tuning aiuta davvero su dati mai visti dal modello di base;
- il test set di tweet_eval, lo stesso usato nella valutazione principale del
  notebook: e' il controllo di regressione, per verificare che il retraining
  non abbia peggiorato le prestazioni sul benchmark originale (catastrophic
  forgetting).

Il modello riaddestrato viene pubblicato su un repo HuggingFace dedicato
(RETRAINED_MODEL_REPO_ID) SOLO se supera il controllo di regressione: in
caso contrario lo script si interrompe con un errore esplicito e non
pubblica nulla. La promozione a "modello in produzione" (aggiornare
MODEL_NAME in config.py) resta comunque una decisione manuale, non
automatica.
"""
import argparse
import os

from datasets import Dataset as HFDataset, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

from config import (
    LABEL_MAP,
    MODEL_NAME,
    ORIGINAL_BENCHMARK_DATASET,
    REGRESSION_TOLERANCE,
    RETRAIN_DATASET,
    RETRAINED_MODEL_REPO_ID,
)

LABEL2ID = {name: idx for idx, name in LABEL_MAP.items()}


def normalize_label(label: str) -> str:
    """
    Converte eventuali label tipo LABEL_0 nei nomi sentiment 
    """
    label = label.lower()
    if label.startswith("label_"):
        return LABEL_MAP[int(label.replace("label_", ""))]
    return label

def evaluate(model_pipeline, texts, true_labels) -> dict:
    """Valuta `model_pipeline` su `texts`, confrontando le predizioni con
    `true_labels`.

    Restituisce accuracy e precision/recall/F1 macro - le stesse metriche
    della valutazione principale del notebook (sezione 9), cosi' i numeri
    prima/dopo il retraining (vedi main()) restano confrontabili con quelli.
    """
    predictions = [
        normalize_label(out["label"])
        # Una pipeline di HuggingFace, quando chiamata come una funzione su una lista di testi, 
        # accetta parametri extra che passa internamente.
        # pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first")
        # OUTPUT: # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
        for out in model_pipeline(texts, batch_size=16, truncation=True, max_length=128)
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

def print_comparison(title: str, before: dict, after: dict) -> None:
    print(f"\n{title}")
    for key in before:
        print(f"  {key}: {before[key]:.4f} -> {after[key]:.4f}")


def resolve_base_model() -> str:
    """Decide da quale modello ripartire per il retraining.

    Se RETRAINED_MODEL_REPO_ID esiste gia' su HuggingFace Hub (pubblicato da
    un'esecuzione precedente di questo script), si riparte da li' - i
    retraining si incatenano invece di ripartire sempre dal modello base
    originale. Altrimenti si usa MODEL_NAME. Attenzione: questo NON promuove
    nulla in produzione - decide solo il punto di partenza per il PROSSIMO
    training. Cosa usano predictor.py/app.py/monitor.py resta deciso da
    MODEL_NAME in config.py, una decisione manuale separata (vedi il
    messaggio finale di main()).

    Controlla solo i metadati del repo (model_info), non scarica i pesi:
    resta leggero anche quando non c'e' ancora nulla da trovare.
    """
    try:
        HfApi().model_info(RETRAINED_MODEL_REPO_ID)
    except RepositoryNotFoundError:
        print(f"Nessun modello riaddestrato trovato su {RETRAINED_MODEL_REPO_ID}: "
              f"riparto dal modello base {MODEL_NAME}.")
        return MODEL_NAME
    print(f"Trovato un modello gia' riaddestrato su {RETRAINED_MODEL_REPO_ID}: riparto da li'.")
    return RETRAINED_MODEL_REPO_ID


def sample_df(hf_split, n, seed):
    """
    Converte uno split HuggingFace (es. retrain_dataset["test"]) in un
    campione pandas pronto all'uso: 
    lo sottocampiona a n righe casuali e aggiunge la colonna leggibile 
    "sentiment" accanto a "label" (0/1/2).
    """
    df = hf_split.to_pandas()
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["sentiment"] = df["label"].map(LABEL_MAP)
    return df


def main(n_train: int, n_eval: int, epochs: int, seed: int) -> None:
    # 1. MODELLO BASE E TOKENIZER (vedi resolve_base_model per la logica di scelta)
    base_model_name = resolve_base_model()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2. DATASET DI RETRAINING (dati NUOVI, mai visti dal modello base)
    print(f"Carico dataset di retraining: {RETRAIN_DATASET} (diverso da TweetEval)...")
    retrain_dataset = load_dataset(RETRAIN_DATASET)

    # Split "train" del dataset nuovo.
    retrain_train_df = retrain_dataset["train"].to_pandas()
    retrain_train_df["sentiment"] = retrain_train_df["label"].map(LABEL_MAP)
    # Split "test" dello stesso dataset. > pipeline prende DataFrame pandas
    retrain_test_df = sample_df(retrain_dataset["test"], n_eval, seed)

    # ------------------------------------------------------------
    # 3. DATASET ORIGINALE (benchmark su cui il modello e' gia' stato addestrato)
    # ------------------------------------------------------------
    # Serve solo come controllo di regressione: dopo il fine-tuning sui dati
    # nuovi, il modello deve continuare ad andare bene anche qui, altrimenti
    # ha "dimenticato" quello che sapeva gia' (catastrophic forgetting) - ed
    # e' proprio questo confronto a decidere se il modello va pubblicato o
    # scartato (vedi in fondo alla funzione). > pipeline prende DataFrame pandas
    print(f"Carico un campione di {ORIGINAL_BENCHMARK_DATASET} (controllo di regressione sul benchmark originale)...")
    original_dataset = load_dataset(ORIGINAL_BENCHMARK_DATASET, "sentiment")
    original_test_df = sample_df(original_dataset["test"], n_eval, seed)

    # 4. Campiono n esempi bilanciati per classe dal dataset di retraining
    n_per_class = max(1, n_train // len(LABEL_MAP))
    retraining_df = (
        retrain_train_df.groupby("sentiment", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), n_per_class), random_state=seed))
        [["text", "sentiment"]]
        .reset_index(drop=True)
    )
    print(f"Dataset di retraining: {len(retraining_df)} esempi ({n_per_class} per classe circa), da {RETRAIN_DATASET}.")

    # ------------------------------------------------------------
    # 5. VALUTAZIONE "PRIMA" DEL RETRAINING
    # ------------------------------------------------------------
    base_pipeline = pipeline(
        "sentiment-analysis",
        model=base_model_name,
        tokenizer=tokenizer,
        truncation=True,
        max_length=128,
    )

    before_new = evaluate(
        base_pipeline, 
        retrain_test_df["text"].tolist(), 
        retrain_test_df["sentiment"].tolist()
    )

    before_original = evaluate(
        base_pipeline, 
        original_test_df["text"].tolist(), 
        original_test_df["sentiment"].tolist()
    )
    print("Metriche PRIMA del retraining, su dati nuovi:", before_new)
    print("Metriche PRIMA del retraining, su benchmark originale:", before_original)

    # ------------------------------------------------------------
    # 6. PREPARAZIONE DEL DATASET PER IL TRAINER
    # ------------------------------------------------------------
    # Trainer di transformers lavora su un datasets.Dataset (non un DataFrame
    # pandas), con: label numerica (non stringa), testo gia' tokenizzato, e
    # nessuna colonna superflua (altrimenti il collator andrebbe in errore).
    # 1. Conversione in datasets.Dataset
    hf_train_dataset = HFDataset.from_pandas(retraining_df)
    # 2. Conversione in label numerica nella colonna "label"
    hf_train_dataset = hf_train_dataset.map(
        lambda batch: {"label": [LABEL2ID[s] for s in batch["sentiment"]]}, batched=True
    )
    # 3. tokenizzazione con il tokenizer
    # tokenizer(testo, ...) chiama tokenizer.__call__(), il modo standard per tokenizzare
    # OUTPUT = dizionario con input_ids/attention_mask
    hf_train_dataset = hf_train_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=128), batched=True
    )
    # 4. Remove_columns per tenere solo le tre colonne che Trainer si aspetta.
    hf_train_dataset = hf_train_dataset.remove_columns(
        [c for c in hf_train_dataset.column_names if c not in ("input_ids", "attention_mask", "label")]
    )
    print(hf_train_dataset)

    # ------------------------------------------------------------
    # 7. FINE-TUNING 
    # ------------------------------------------------------------
    # Nuova istanza del modello
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    # config dei parametri di training
    training_args = TrainingArguments(
        output_dir="./retraining_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="no",  # dimostrativo: non serve salvare checkpoint su disco/CI
        report_to=[],  # niente integrazioni di logging esterne (es. wandb)
    )
    # Trainer utilizza le impostazioni di training_args per eseguire e gestire 
    # il ciclo di training, occupandosi automaticamente di operazioni come 
    # forward pass, calcolo della loss, backward pass, aggiornamento dei pesi,
    # gestione del device e passaggio tra modalità train() ed eval().
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    print(f"\nRetraining su {len(hf_train_dataset)} esempi, {epochs} epoca/e...")
    trainer.train()

    # ------------------------------------------------------------
    # 8. VALUTAZIONE "DOPO" IL RETRAINING E CONFRONTO
    # ------------------------------------------------------------
    # Stessa identica valutazione del punto 5, ma con la pipeline costruita
    # sul modello appena riaddestrato: il confronto prima/dopo e' quindi
    # sugli stessi identici testi di test, cambia solo il modello.
    retrained_pipeline = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer, 
        truncation=True, 
        max_length=128
    )
    after_new = evaluate(
        retrained_pipeline, 
        retrain_test_df["text"].tolist(), 
        retrain_test_df["sentiment"].tolist())
    
    after_original = evaluate(
        retrained_pipeline,
        original_test_df["text"].tolist(), 
        original_test_df["sentiment"].tolist()
    )

    print_comparison(f"Confronto su {RETRAIN_DATASET} (dati nuovi, mai visti dal modello base):", before_new, after_new)
    print_comparison(
        f"Confronto su {ORIGINAL_BENCHMARK_DATASET} (benchmark originale, controllo di regressione):",
        before_original,
        after_original,
    )

    # ------------------------------------------------------------
    # 9. GATE: il modello viene salvato solo se non e' peggiorato sul benchmark
    # ------------------------------------------------------------
    # Il retraining puo' anche "funzionare" (girare senza errori) e produrre
    # comunque un modello peggiore di quello in uso: qui e' il punto in cui
    # lo decidiamo, invece di limitarci a stampare i numeri e lasciare che
    # sia una persona a doverli leggere per accorgersene.
    f1_drop = before_original["f1_macro"] - after_original["f1_macro"]
    if f1_drop > REGRESSION_TOLERANCE:
        raise SystemExit(
            f"Retraining rifiutato: F1 macro sul benchmark originale sceso di "
            f"{f1_drop:.4f} (tolleranza {REGRESSION_TOLERANCE:.4f}). "
            "Il modello riaddestrato NON viene pubblicato."
        )

    print(
        f"\nControllo di regressione superato (calo F1 macro: {f1_drop:.4f}, "
        f"tolleranza {REGRESSION_TOLERANCE:.4f}). Pubblico il modello candidato su "
        f"{RETRAINED_MODEL_REPO_ID}..."
    )
    token = os.environ["HF_TOKEN"]
    model.push_to_hub(RETRAINED_MODEL_REPO_ID, token=token)
    tokenizer.push_to_hub(RETRAINED_MODEL_REPO_ID, token=token)
    print(
        f"Modello candidato pubblicato: https://huggingface.co/{RETRAINED_MODEL_REPO_ID}\n"
        "Per usarlo davvero in produzione, aggiorna MODEL_NAME in config.py "
        f"a '{RETRAINED_MODEL_REPO_ID}' - questa e' una decisione manuale, "
        "non automatica."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=90, help="Esempi totali di retraining (divisi per classe).")
    parser.add_argument("--n-eval", type=int, default=200, help="Esempi di test per ciascun confronto prima/dopo.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_train=args.n_train, n_eval=args.n_eval, epochs=args.epochs, seed=args.seed)
