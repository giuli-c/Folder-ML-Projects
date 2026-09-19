"""
Training/retraining su dataset interno Mastodon con etichette approvate.
Default: monitoring/review_queue.json, solo righe approved e revisione reale.
Il budget viene ridotto automaticamente se sono disponibili meno esempi.
La mancanza di dati approvati interrompe la prova prima di scaricare modelli.
--data-source external abilita esplicitamente il vecchio esperimento pubblico.

Training/retraining eseguibile da riga di comando, per il job "train" della
pipeline CI/CD (.github/workflows/train.yml, trigger manuale workflow_dispatch).

A differenza della valutazione principale del notebook (sezioni 1-9, sul test
set di cardiffnlp/tweet_eval), qui il fine-tuning avviene su un dataset
DIVERSO: mteb/tweet_sentiment_extraction (fonte: competizione Kaggle 2020, non
TweetEval). Il motivo: cardiffnlp/twitter-roberta-base-sentiment-latest e'
stato originariamente fine-tuned proprio su TweetEval per il task di sentiment
(vedi model card HuggingFace) - il dataset nuovo introduce esempi diversi; la quota di replay
riutilizza invece testi originali per preservare il compito precedente. Il nuovo dataset usa lo
stesso schema di etichette (0=negative, 1=neutral, 2=positive), quindi nessun
remapping aggiuntivo e' necessario.

Il training mescola dati nuovi e un campione del train TweetEval (replay):
1500 testi totali di default, di cui 1000 nuovi e 500 originali. Il replay
serve a contenere la perdita di prestazioni precedenti, senza garanzie.
Il backbone resta congelato; si aggiorna solo la testa. A ogni epoca, due
validation separate guidano early stopping e selezione del checkpoint in RAM.
Il checkpoint deve migliorare sui nuovi dati senza regredire oltre la tolleranza
sulla validation TweetEval. In assenza di checkpoint valido mostriamo comunque il test dell'ultima epoca
a scopo diagnostico, senza autorizzare la pubblicazione.

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
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from datasets import Dataset as HFDataset, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    set_seed,
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

def evaluate(model_pipeline, texts, true_labels, prediction_sink=None) -> dict:
    """
    Valuta `model_pipeline` su `texts`, confrontando le predizioni con
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
    if prediction_sink is not None:
        prediction_sink.extend(predictions)
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
    """
    Decide da quale modello ripartire per il retraining.

    Se RETRAINED_MODEL_REPO_ID esiste gia' su HuggingFace Hub (pubblicato da
    un'esecuzione precedente di questo script), si riparte da li' - i
    retraining si incatenano invece di ripartire sempre dal modello base
    originale. Altrimenti si usa MODEL_NAME. 
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



def split_retraining_data(hf_split, n_val, seed):
    """
    Prepara i nuovi dati da utilizzare per il retraining del modello.

    La funzione:
    1. converte il dataset Hugging Face in un DataFrame Pandas;
    2. elimina gli esempi senza testo o etichetta;
    3. elimina i testi duplicati per evitare che lo stesso testo possa
       comparire sia nel training sia nella validation;
    4. divide i dati in Training Set e Validation Set mantenendo
       la stessa proporzione delle classi (split stratificato);
    5. aggiunge la colonna 'sentiment', trasformando le label numeriche
       nei corrispondenti nomi delle classi tramite LABEL_MAP.

    La Validation viene separata prima di qualsiasi successivo
    campionamento dei dati di training, in modo che non venga utilizzata
    per addestrare il modello.

    Il Test Set rimane completamente separato e verrà utilizzato
    solamente per la valutazione finale del modello.
    """
    df = hf_split.to_pandas().dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    train_df, val_df = train_test_split(
        df, test_size=n_val, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_df.copy(), val_df.copy()
    for frame in (train_df, val_df):
        frame["sentiment"] = frame["label"].map(LABEL_MAP)
    return train_df, val_df



def build_replay_training(new_pool, original_pool, n_train, n_replay, seed, excluded_texts):
    """
    Costruisce il dataset che verrà utilizzato per il retraining del modello.

    Il dataset finale combina due fonti:
    1. DATI NUOVI: post revisionati e approvati
    2. DATI DI REPLAY: esempi provenienti dal Training Set originale di TweetEval.
       Vengono aggiunti per mantenere durante il retraining anche
       esempi appartenenti alla distribuzione originale.
    `n_train` indica il numero TOTALE di esempi desiderati.

    Entrambi i gruppi vengono campionati in modo il più possibile
    bilanciato tra le tre classi.
    Prima del campionamento vengono inoltre:
    - eliminati esempi senza testo o label;
    - eliminati testi duplicati;
    - esclusi i testi appartenenti a validation e test;
    - evitati duplicati tra dati nuovi e dati di replay.

    Alla fine i due gruppi vengono uniti, mescolati e viene restituito
    un DataFrame pronto per essere utilizzato nel retraining.
    """
    if n_train - n_replay < 3 or n_replay < 0 or (0 < n_replay < 3):
        raise ValueError("Servono almeno 3 testi nuovi e zero oppure almeno 3 testi replay.")

    def sample_balanced(pool, count, excluded):
        """
        Estrae da un dataset (`pool`) il numero richiesto di esempi,
        distribuendoli nel modo più uniforme possibile tra le classi.
        Prima del campionamento:
        - elimina righe senza testo o label;
        - elimina testi duplicati;
        - elimina i testi presenti nell'insieme `excluded`.
        Successivamente divide `count` tra le tre classi.
        Se `count` non è perfettamente divisibile per 3, gli esempi
        rimanenti vengono distribuiti alle prime classi.
        Se una classe non contiene abbastanza esempi per raggiungere
        la quantità richiesta, viene sollevato un ValueError.

        Restituisce il campione bilanciato con anche la colonna
        `sentiment`, ottenuta dalla label numerica tramite LABEL_MAP.
        """
        # -----------------------------------------------------
        # PULIZIA DEL DATASET
        # -----------------------------------------------------
        pool = pool.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
        # esclusione dal training di esempi appartenenti a validation/test.
        pool = pool.loc[~pool["text"].isin(excluded)].copy()

        # CAMPIONAMENTO BILANCIATO PER CLASSE
        parts = []
        for index, label in enumerate(sorted(LABEL_MAP)):
            # Calcola quanti esempi dobbiamo prendere
            # dalla classe corrente.
            required = count // len(LABEL_MAP) + (index < count % len(LABEL_MAP))
             # prendo da pool gli esmepi della classe in questione
            group = pool.loc[pool["label"] == label]
            if len(group) < required:
                raise ValueError(f"Classe {label}: solo {len(group)} testi disponibili, richiesti {required}.")
            parts.append(group.sample(n=required, random_state=seed))
        # Unisce i campioni delle tre classi.
        result = pd.concat(parts, ignore_index=True)
        # Converte la label numerica nel nome del sentiment.
        result["sentiment"] = result["label"].map(LABEL_MAP)
        return result

    # ---------------------------------------------------------
    # 2. CAMPIONAMENTO DEI DATI NUOVI
    # ---------------------------------------------------------
    new = sample_balanced(new_pool, n_train - n_replay, excluded_texts)
    # ---------------------------------------------------------
    # 3. CAMPIONAMENTO DEI DATI DI REPLAY
    # ---------------------------------------------------------
    original = sample_balanced(original_pool, n_replay, set(excluded_texts) | set(new["text"]))
    result = pd.concat([new, original], ignore_index=True)
    print(f"Training con replay: {len(new)} testi nuovi + {len(original)} testi TweetEval "
          f"= {len(result)} totali. Validation e test esclusi.")
    # ---------------------------------------------------------
    # 4. PREPARAZIONE DEL DATASET FINALE
    # ---------------------------------------------------------
    # Conserva solamente le colonne necessarie al training:
    # text | sentiment
    # e mescola casualmente tutte le righe.
    # non rimangono raggruppati.
    result = result[["text", "sentiment"]].sample(frac=1, random_state=seed).reset_index(drop=True)
    result.attrs["source_counts"] = {
        "Training nuovo selezionato": new["sentiment"].value_counts().to_dict(),
        "Replay selezionato": original["sentiment"].value_counts().to_dict(),
    }
    return result


def save_analysis_report(directory, metadata, distributions, datasets):
    """
    Salva predizioni appaiate e dati della prova anche quando viene rifiutata.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "distributions": distributions, "datasets": datasets}
    path = directory / "report.json"
    # Prima serializziamo: in caso di errore non scriviamo un report parziale.
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(content, encoding="utf-8")
    print(f"Analisi salvata in: {path.resolve()}")
    return path


class ValidationCheckpoint(TrainerCallback):
    """
    Gestisce la selezione del checkpoint migliore durante il retraining
    e implementa una forma di Early Stopping.
    Alla fine di ogni epoca il modello viene valutato su due Validation Set:
    - Validation nuova:
      serve a verificare se il retraining sta migliorando le prestazioni
      sui nuovi dati.
    - Validation originale:
      serve a controllare che il retraining non peggiori eccessivamente
      le prestazioni sui dati originali.

    Un checkpoint viene quindi salvato solamente se soddisfa ENTRAMBE
    le condizioni:
    1. la F1 sulla validation originale non è peggiorata oltre
       REGRESSION_TOLERANCE rispetto al modello iniziale;
    2. la F1 sulla validation nuova è migliorata di almeno min_delta
       rispetto al miglior risultato ottenuto fino a quel momento.
    Poiché durante il retraining il resto di RoBERTa è congelato,
    vengono conservati in RAM solamente i parametri addestrabili
    della testa di classificazione, evitando di salvare l'intero modello.

    Se per un numero di epoche pari a 'patience' non viene trovato
    nessun nuovo checkpoint che soddisfi entrambe le condizioni,
    il training viene interrotto tramite Early Stopping.

    NOTA:
    min_delta rappresenta una soglia pratica utilizzata per stabilire
    se il miglioramento è sufficientemente grande da essere considerato,
    ma non costituisce un test di significatività statistica.
    """

    def __init__(self, model_pipeline, new_df, original_df,
                 baseline_new_f1, baseline_original_f1, patience=2, min_delta=0.001):
        self.pipe = model_pipeline
        self.new_df = new_df
        # Validation Set originale, utilizzato per controllare che
        # il modello non perda eccessivamente le conoscenze precedenti.
        self.original_df = original_df
        self.best_f1 = baseline_new_f1
        self.original_f1 = baseline_original_f1
        self.patience = patience
        # miglioramento minimo della F1 sulla validation nuova necessario
        # per considerare il nuovo risultato realmente migliore.
        self.min_delta = min_delta
        self.stale_epochs = 0
        self.best_epoch = None
        self.best_head = None

    def consider(self, new_f1, original_f1, model, epoch):
        allowed = self.original_f1 - original_f1 <= REGRESSION_TOLERANCE
        improved = new_f1 > self.best_f1 + self.min_delta
        if allowed and improved:
            self.best_f1 = new_f1
            self.best_epoch = epoch
            self.best_head = {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters() if param.requires_grad
            }
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return allowed and improved

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        new = evaluate(self.pipe, self.new_df["text"].tolist(),
                       self.new_df["sentiment"].tolist())
        original = evaluate(self.pipe, self.original_df["text"].tolist(),
                            self.original_df["sentiment"].tolist())
        selected = self.consider(new["f1_macro"], original["f1_macro"], model, state.epoch)
        print(f"\nValidation epoca {state.epoch:.0f}: "
              f"F1 nuovo={new['f1_macro']:.4f}, F1 originale={original['f1_macro']:.4f}; "
              f"checkpoint selezionato: {'si' if selected else 'no'}.")
        if self.stale_epochs >= self.patience:
            print(f"Early stopping: {self.patience} epoche senza miglioramenti ammissibili.")
            control.should_training_stop = True
        return control

    def restore(self, model):
        if self.best_head is None:
            return False
        # Gli altri pesi sono rimasti congelati: basta ripristinare la testa.
        model.load_state_dict(self.best_head, strict=False)
        print(f"Ripristinato checkpoint dell'epoca {self.best_epoch:.0f}.")
        return True

def main(n_train: int, n_eval: int, epochs: int, seed: int, no_push: bool = False,
         n_val: int = 200, learning_rate: float = 1e-6, patience: int = 2,
         n_replay: int = 500, report_dir=None, reviewed_data=None, data_source="reviewed") -> None:
    if n_train - n_replay < 3 or n_replay < 0 or (0 < n_replay < 3) or min(n_eval, n_val, epochs, patience) <= 0 or learning_rate <= 0:
        raise ValueError("Servono almeno 3 testi nuovi, replay zero o almeno 3; gli altri parametri positivi.")
    # Il dataset interno e' il percorso principale: nessun ripiego silenzioso
    # sul dataset pubblico se mancano revisioni. Controllo PRIMA dei download.
    if data_source == "reviewed":
        from review_data import QUEUE_PATH, training_budget
        reviewed_data = reviewed_data or QUEUE_PATH
        n_train, n_replay, counts = training_budget(reviewed_data, n_train, n_replay)
        print("Esempi approvati per split e classe:", counts)
        print(f"Budget effettivo: {n_train - n_replay} interni + {n_replay} replay = {n_train}.")
    elif data_source != "external" or reviewed_data:
        raise ValueError("Scegliere reviewed oppure external; --reviewed-data vale solo per reviewed.")
    set_seed(seed)
    # 1. MODELLO BASE E TOKENIZER (vedi resolve_base_model per la logica di scelta)
    base_model_name = resolve_base_model()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2. DATASET DI RETRAINING (dati NUOVI, mai visti dal modello base)
    new_dataset_name = RETRAIN_DATASET
    if reviewed_data:
        from review_data import approved_splits, readiness
        ready, counts = readiness(reviewed_data)
        if not ready:
            raise ValueError(f"Dataset revisionato insufficiente: {counts}")
        # Usiamo solo etichette approvate, con split persistenti: i testi
        # gia' valutati non migrano nel training a ogni nuova raccolta.
        retrain_dataset = {name: HFDataset.from_list(rows)
                           for name, rows in approved_splits(reviewed_data).items()}
        new_dataset_name = "Mastodon - dataset interno con revisione umana"
        retrain_train_df = retrain_dataset["train"].to_pandas()
        retrain_train_df["sentiment"] = retrain_train_df["label"].map(LABEL_MAP)
        retrain_val_df = sample_df(retrain_dataset["validation"], n_val, seed)
        if n_train - n_replay > len(retrain_train_df):
            raise ValueError("Richiesti piu' testi nuovi di quelli approvati nel train.")
    else:
        print(f"Carico dataset di retraining: {RETRAIN_DATASET} (diverso da TweetEval)...")
        retrain_dataset = load_dataset(RETRAIN_DATASET)
        retrain_train_df, retrain_val_df = split_retraining_data(
            retrain_dataset["train"], n_val, seed
        )
    print("Fonte dei nuovi dati:", new_dataset_name)
    # Split "test" dello stesso dataset. > pipeline prende DataFrame pandas
    retrain_test_df = sample_df(retrain_dataset["test"], n_eval, seed)

    # ------------------------------------------------------------
    # 3. DATASET ORIGINALE (benchmark su cui il modello e' gia' stato addestrato)
    # ------------------------------------------------------------
    # Gli split validation/test servono come controllo di regressione: dopo il fine-tuning sui dati
    # nuovi, il modello deve continuare ad andare bene anche qui, altrimenti
    # ha "dimenticato" quello che sapeva gia' (catastrophic forgetting) - ed
    # e' proprio questo confronto a decidere se il modello va pubblicato o
    # scartato (vedi in fondo alla funzione). > pipeline prende DataFrame pandas
    print(f"Carico un campione di {ORIGINAL_BENCHMARK_DATASET} (controllo di regressione sul benchmark originale)...")
    original_dataset = load_dataset(ORIGINAL_BENCHMARK_DATASET, "sentiment")
    original_test_df = sample_df(original_dataset["test"], n_eval, seed)
    original_val_df = sample_df(original_dataset["validation"], n_val, seed)
    print(f"Validation separata: {len(retrain_val_df)} testi nuovi, "
          f"{len(original_val_df)} testi TweetEval.")

    # 4. Replay: parte dei testi proviene dal TRAIN originale per ricordare
    # il compito precedente senza aumentare il numero totale di esempi.
    # Escludiamo anche eventuali testi identici presenti negli split riservati.
    excluded_texts = set(retrain_val_df["text"])
    for held_out in (retrain_dataset["test"], original_dataset["validation"], original_dataset["test"]):
        excluded_texts.update(held_out["text"])
    if "validation" in retrain_dataset:
        excluded_texts.update(retrain_dataset["validation"]["text"])
    retraining_df = build_replay_training(
        retrain_train_df, original_dataset["train"].to_pandas(),
        n_train, n_replay, seed, excluded_texts,
    )

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

    predictions_before_new, predictions_before_original = [], []
    before_new = evaluate(
        base_pipeline, 
        retrain_test_df["text"].tolist(), 
        retrain_test_df["sentiment"].tolist(), prediction_sink=predictions_before_new
    )

    before_original = evaluate(
        base_pipeline, 
        original_test_df["text"].tolist(), 
        original_test_df["sentiment"].tolist(), prediction_sink=predictions_before_original
    )
    print("Metriche PRIMA del retraining, su dati nuovi:", before_new)
    print("Metriche PRIMA del retraining, su benchmark originale:", before_original)

    validation_new = evaluate(base_pipeline, retrain_val_df["text"].tolist(),
                              retrain_val_df["sentiment"].tolist())
    validation_original = evaluate(base_pipeline, original_val_df["text"].tolist(),
                                   original_val_df["sentiment"].tolist())
    print("F1 validation iniziale:", validation_new["f1_macro"],
          "(nuovo),", validation_original["f1_macro"], "(TweetEval)")

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
    # Riutilizziamo il modello appena valutato: una sola copia in RAM/VRAM.
    # Le metriche PRIMA sono gia' state calcolate e conservate.
    model = base_pipeline.model
    # RoBERTa ha gia' appreso rappresentazioni utili dei tweet: manteniamo
    # fissi i suoi pesi e aggiorniamo solo la testa che decide il sentiment.
    # Questo riduce il lavoro di training, ma non garantisce un miglioramento:
    # anche la nuova testa deve superare il controllo di regressione finale.
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Backbone congelato: {trainable:,} parametri allenabili su {total:,} (solo testa).")
    # config dei parametri di training
    training_args = TrainingArguments(
        output_dir="./retraining_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=learning_rate,
        seed=seed,
        dataloader_pin_memory=False,  # adatto anche al training su CPU
        logging_steps=10,
        save_strategy="no",  # il callback conserva in RAM solo la testa migliore
        report_to=[],  # niente integrazioni di logging esterne (es. wandb)
    )
    # Trainer utilizza le impostazioni di training_args per eseguire e gestire 
    # il ciclo di training, occupandosi automaticamente di operazioni come 
    # forward pass, calcolo della loss, backward pass, aggiornamento dei pesi,
    # gestione del device e passaggio tra modalità train() ed eval().
    checkpoint = ValidationCheckpoint(
        base_pipeline, retrain_val_df, original_val_df,
        validation_new["f1_macro"], validation_original["f1_macro"],
        patience=patience,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[checkpoint],
    )
    print(f"\nRetraining su {len(hf_train_dataset)} esempi, {epochs} epoca/e...")
    trainer.train()
    validation_passed = checkpoint.restore(model)
    if not validation_passed:
        # Non interrompiamo qui: mostriamo comunque le metriche dell'ultima epoca.
        # Questo confronto diagnostico non rende pubblicabile il candidato.
        print("\033[31mATTENZIONE: nessun checkpoint ha superato la validation. "
              "Mostro il confronto dell'ULTIMA EPOCA solo a scopo diagnostico. "
              "Il candidato resta rifiutato.\033[0m", flush=True)
    else:
        print(f"Confronto del checkpoint selezionato: epoca {checkpoint.best_epoch:.0f}.")

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
    predictions_after_new, predictions_after_original = [], []
    after_new = evaluate(
        retrained_pipeline, 
        retrain_test_df["text"].tolist(), 
        retrain_test_df["sentiment"].tolist(), prediction_sink=predictions_after_new)
    
    after_original = evaluate(
        retrained_pipeline,
        original_test_df["text"].tolist(), 
        original_test_df["sentiment"].tolist(), prediction_sink=predictions_after_original
    )

    print_comparison(f"Confronto su {new_dataset_name} (test separato dal retraining):", before_new, after_new)
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
    f1_drop = float(before_original["f1_macro"] - after_original["f1_macro"])
    # Report persistente: il notebook puo' analizzare anche una prova rifiutata.
    # Ogni avvio del notebook usa una cartella diversa, evitando risultati obsoleti.
    distributions = dict(retraining_df.attrs["source_counts"])
    for name, frame in {
        "Training totale": retraining_df,
        "Validation nuovo": retrain_val_df, "Validation TweetEval": original_val_df,
        "Test nuovo": retrain_test_df, "Test TweetEval": original_test_df,
        "Pool train nuovo": retrain_train_df,
    }.items():
        distributions[name] = frame["sentiment"].value_counts().to_dict()
    distributions["Pool train TweetEval"] = (
        original_dataset["train"].to_pandas()["label"].map(LABEL_MAP).value_counts().to_dict()
    )
    diagnostic = {}
    for name, frame, before, after, preds_before, preds_after in [
        (new_dataset_name, retrain_test_df, before_new, after_new, predictions_before_new, predictions_after_new),
        (ORIGINAL_BENCHMARK_DATASET, original_test_df, before_original, after_original,
         predictions_before_original, predictions_after_original),
    ]:
        diagnostic[name] = {
            "metrics_before": before, "metrics_after": after,
            "rows": [
                {"text": text, "true": label, "before": first, "after": last}
                for text, label, first, last in zip(
                    frame["text"].tolist(), frame["sentiment"].tolist(), preds_before, preds_after)
            ],
        }
    save_analysis_report(
        report_dir or Path("retraining_output") / ("analysis_" + uuid4().hex[:12]),
        {"base_model": base_model_name, "new_dataset": new_dataset_name, "n_train": len(retraining_df), "n_replay": n_replay,
         "n_eval": n_eval, "n_val": n_val, "seed": seed, "learning_rate": learning_rate,
         "max_epochs": epochs, "completed_epochs": trainer.state.epoch,
         "evaluated_epoch": checkpoint.best_epoch if validation_passed else trainer.state.epoch,
         "validation_passed": validation_passed,
         "test_gate_passed": f1_drop <= REGRESSION_TOLERANCE,
         "candidate_accepted": validation_passed and f1_drop <= REGRESSION_TOLERANCE,
         "comparison": "checkpoint selezionato" if validation_passed else "ultima epoca diagnostica"},
        distributions, diagnostic,
    )
    # Il rifiuto viene comunicato DOPO aver stampato entrambi i confronti.
    # La validation resta vincolante anche se il test diagnostico e' buono.
    if not validation_passed:
        raise SystemExit(
            "\033[31mRetraining rifiutato sulla validation: nessuna epoca migliora "
            "la F1 nuova di oltre 0.001 rispettando la tolleranza sul benchmark. "
            "Il confronto sopra riguarda l'ultima epoca, non un checkpoint approvato. "
            "Il modello NON viene pubblicato.\033[0m"
        )
    if f1_drop > REGRESSION_TOLERANCE:
        raise SystemExit(
            f"\033[31mRetraining rifiutato: F1 macro sul benchmark originale sceso di "
            f"{f1_drop:.4f} (tolleranza {REGRESSION_TOLERANCE:.4f}). "
            "Il modello riaddestrato NON viene pubblicato.\033[0m"
        )

    if no_push:
        print(f"Controllo di regressione superato (calo F1 macro: {f1_drop:.4f}). "
              "Prova conclusa: --no-push esclude la pubblicazione su Hugging Face.")
        return

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
    parser.add_argument("--n-train", type=int, default=1500, help="Esempi totali di retraining, inclusa la quota replay; esclusa la validation.")
    parser.add_argument("--n-replay", type=int, default=500, help="Quota del totale presa dal train TweetEval; 0 disattiva il replay.")
    parser.add_argument("--n-eval", type=int, default=200, help="Esempi di test per ciascun confronto prima/dopo.")
    parser.add_argument("--epochs", type=int, default=3, help="Massimo di epoche; early stopping attivo.")
    parser.add_argument("--n-val", type=int, default=200, help="Esempi per ciascuna validation.")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=2, help="Epoche senza miglioramenti ammissibili prima dello stop.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", choices=["reviewed", "external"], default="reviewed", help="Default: dataset interno approvato; external ripete il vecchio esperimento.")
    parser.add_argument("--reviewed-data", default=None, help="JSON della coda: usa solo esempi approvati e split persistenti.")
    parser.add_argument("--report-dir", default=None, help="Cartella per il report di analisi; default: cartella univoca in retraining_output.")
    parser.add_argument("--no-push", action="store_true", help="Esegue training e valutazione senza pubblicare il modello.")
    args = parser.parse_args()
    main(n_train=args.n_train, n_eval=args.n_eval, epochs=args.epochs, seed=args.seed, no_push=args.no_push,
         n_val=args.n_val, learning_rate=args.learning_rate, patience=args.patience, n_replay=args.n_replay, report_dir=args.report_dir, reviewed_data=args.reviewed_data, data_source=args.data_source)
