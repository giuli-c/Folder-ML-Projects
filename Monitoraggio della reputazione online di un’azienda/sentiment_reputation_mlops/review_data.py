"""
Coda di revisione umana condivisa da monitor.py, train.py e human_retrain.py.

Il flusso: monitor.py accoda i post con confidence bassa in 
monitoring/review_queue.json; una persona apre quel file e scrive a mano 
validated_label/review_status/reviewer/reviewed_at per ogni riga da 
approvare o escludere - questo modulo non assegna mai un'etichetta da solo. 
Le funzioni qui sotto leggono quella coda, verificano che le approvazioni
siano complete e coerenti, e preparano i tre split (train/validation/test)
che train.py userà per il retraining.

Per velocizzare l'approvazione: dopo aver scritto a mano solo
validated_label su una riga "pending", si può lanciare approve_reviewed.py,
che completa da solo review_status/reviewer/reviewed_at - senza decidere
mai un'etichetta al posto della persona.
"""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("negative", "neutral", "positive")
QUEUE_PATH = Path(__file__).parent / "monitoring" / "review_queue.json"
# Numero minimo di esempi approvati per OGNI classe in ciascuno split.
# Sono soglie per la dimostrazione, non una garanzia di accuratezza delle metriche.
MIN_COUNTS = {"train": 20, "validation": 5, "test": 5}


def text_key(text):
    """
    Crea una chiave identificativa stabile a partire dal contenuto di un testo.

    Prima normalizza il testo:
    - ignora le differenze tra maiuscole e minuscole;
    - elimina gli spazi multipli.

    Per esempio:
        "Hello  World"
        "hello world"
        "HELLO     WORLD"
    vengono tutti trasformati nella stessa forma:
        "hello world"

    Sul testo normalizzato viene poi calcolato un hash SHA-256,
    che funziona come una "impronta digitale" del contenuto.
    Lo stesso testo normalizzato produce quindi sempre la stessa chiave.
    Questa chiave viene utilizzata successivamente dal programma per:
    - riconoscere testi duplicati;
    - prendere decisioni riproducibili sul campionamento;
    - assegnare in modo stabile i testi agli split
      train / validation / test.

    IMPORTANTE:
    questa funzione crea solamente la chiave.
    Le decisioni sul campionamento e sugli split vengono effettuate
    successivamente utilizzando questa chiave.
    """
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def load_queue(path=QUEUE_PATH):
    """
    Legge la coda così com'è nel JSON, senza validare nulla.

    File assente = lista vuota (prima raccolta, non c'è ancora niente da
    leggere). 
    File presente ma JSON non valido = errore esplicito: non viene
    trattato come vuoto, per non perdere in silenzio revisioni già fatte se
    il file risulta corrotto o scritto solo a metà.

    Restituisce righe pending, approved ed excluded tutte mescolate insieme:
    è approved_splits (più sotto) a filtrarle e validarle prima del training.
    """
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_queue(rows, path=QUEUE_PATH):
    """
    Salva su file lo stato completo della coda di revisione.
    Parametri:
    - rows: contiene tutte le righe che devono essere presenti nella coda;
    - path: indica il file in cui salvare la coda.

    La funzione:
    1. individua il percorso in cui deve essere salvato il file;
    2. crea automaticamente la cartella di destinazione se non esiste;
    3. converte l'intera coda `rows` in formato JSON;
    4. salva prima i dati in un file temporaneo;
    5. solo quando la scrittura è terminata, sostituisce il vecchio
       file della coda con quello appena creato.

    L'utilizzo di un file temporaneo rende il salvataggio più sicuro:
    se il programma si interrompesse mentre sta scrivendo i dati,
    il file originale della coda rimarrebbe intatto invece di rischiare
    di essere salvato solo parzialmente o corrotto.

    IMPORTANTE:
    questa funzione salva sempre l'INTERA coda ricevuta in `rows`.
    Inoltre si occupa solamente del salvataggio dei dati:
    non modifica lo stato delle righe..
    """
    # Converte il percorso in un oggetto Path per facilitarne la gestione.
    path = Path(path)
    # Crea la cartella, se non esiste già.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Crea il percorso di un file temporaneo.
    temp = path.with_suffix(".tmp")
    # Converte tutta la coda in JSON e la salva nel file temporaneo.
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Sostituzione del file al termine della scrittura
    temp.replace(path)


def enqueue(posts, predictions, model_name, scope, threshold=0.75, audit_rate=0.10, path=QUEUE_PATH):
    """
    Aggiunge alla coda i post che meritano una revisione umana.

    La funzione riceve:
    - `posts`: i post appena raccolti;
    - `predictions`: le predizioni del modello sugli stessi post;
    - `model_name`: nome del modello che ha prodotto le predizioni;
    - `scope`: contesto/ambito del monitoraggio;
    - `threshold`: soglia di default (0.75);
    - `audit_rate`: percentuale di predizioni ad alta confidence che
      vogliamo comunque controllare manualmente (default 10%);
    - `path`: file in cui è salvata la coda.

    I post possono entrare nella coda per DUE motivi:
    1. LOW CONFIDENCE
       Se confidence <= threshold, il modello non è abbastanza sicuro
       della propria predizione e il post viene mandato alla revisione.
    2. RANDOM AUDIT
       Anche alcune predizioni con confidence alta vengono selezionate
       come controllo. Con audit_rate=0.10 viene selezionato circa il 10%.
       La selezione non utilizza un numero casuale generato ogni volta:
       viene ricavata dalla chiave stabile prodotta da text_key().

    Prima di aggiungere un post, la funzione controlla inoltre che non
    sia già presente nella coda:
    - con lo stesso post_id;
    - oppure con lo stesso contenuto testuale normalizzato.

    Ogni nuovo post viene inserito con:
    - la predizione del modello e la relativa confidence;
    - il motivo per cui è stato selezionato;
    - review_status="pending";
    - nessuna validated_label, perché la vera etichetta dovrà essere
      assegnata successivamente da un revisore umano;
    - uno split stabile: train, validation oppure test.

    Lo split viene ricavato dalla chiave del testo con proporzioni
    approssimative:
        60% train
        20% validation
        20% test
    Alla fine viene salvata l'intera coda aggiornata.
    La funzione restituisce solamente il numero di NUOVI post aggiunti
    durante questa chiamata.
    """

    # I post e le predizioni devono corrispondere uno a uno.    
    # Esempio:
    # posts[0]       -> primo post
    # predictions[0] -> predizione del primo post   
    # Se le due liste hanno lunghezze diverse non possiamo sapere
    # correttamente quale predizione appartiene a quale post.
    if len(posts) != len(predictions):
        raise ValueError("Post e predizioni devono essere allineati.")
    rows = load_queue(path)
    known_ids = {row["post_id"] for row in rows}
    known_texts = {text_key(row["text"]) for row in rows}

    added = 0
    for post, prediction in zip(posts, predictions):
        # Crea l'impronta stabile del testo.
        key = text_key(post["text"])
        confidence = float(prediction["confidence"])
        label = prediction["sentiment"].lower()
        if label not in LABELS or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Predizione non valida.")
        # ---------------------------------------------------------
        # MOTIVO 1: LOW CONFIDENCE
        # ---------------------------------------------------------
        # Se la confidence è minore o uguale alla soglia,
        # la predizione viene considerata poco sicura e quindi
        # deve essere controllata da una persona.
        uncertain = confidence <= threshold

        # ---------------------------------------------------------
        # MOTIVO 2: RANDOM AUDIT
        # ---------------------------------------------------------
        # VogliO controllare manualmente anche una piccola parte
        # delle predizioni che il modello considera sicure.
        # PrendO i primi 8 caratteri della chiave SHA-256 e li
        # trasformO in un numero compreso circa tra 0 e 1.
        # Se questo numero è inferiore ad audit_rate, il post viene
        # selezionato per il controllo.
        #
        # Con audit_rate=0.10 viene selezionato circa il 10% dei testi.
        sampled = int(key[:8], 16) / 2**32 < audit_rate

        if not (uncertain or sampled) or key in known_texts or post["post_id"] in known_ids:
            continue
        rows.append({
            **post, "predicted_label": label, "confidence": confidence,
            "model_name": model_name, "scope": scope,
            "selection_reason": "low_confidence" if uncertain else "random_audit",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "validated_label": None, "review_status": "pending",
            "reviewer": None, "reviewed_at": None, "review_is_simulated": False,
            # Un altro segmento della chiave assegna 6 valori su 10 al train,
            # 2 alla validation e 2 al test. Le proporzioni sono approssimative.
            # La correzione dell'etichetta non sposta il testo tra gli split.
            "split": "train" if int(key[8:16], 16) % 10 < 6 else (
                "validation" if int(key[8:16], 16) % 10 < 8 else "test"),
        })
        known_texts.add(key)
        known_ids.add(post["post_id"])
        added += 1
    save_queue(rows, path)
    return added


def approved_splits(path=QUEUE_PATH):
    """
    Prepara i dati approvati dalla revisione umana per il futuro retraining.

    La funzione legge tutti i post presenti nella coda di revisione,
    ma considera solamente quelli con:
        review_status = "approved"
    I post ancora "pending" oppure "excluded" vengono ignorati.

    Prima di utilizzare un post approvato, controlla che sia completo
    e valido. In particolare verifica che:
    - la revisione non sia simulata (`review_is_simulated=False`);
    - `validated_label` contenga una delle classi previste;
    - il post appartenga a uno split valido: train, validation o test;
    - siano presenti il nome del revisore e la data della revisione;
    - il testo non sia vuoto;
    - lo stesso testo non compaia più volte tra i dati approvati.

    Se anche un solo post approvato non supera questi controlli,
    viene sollevato un ValueError: la funzione non restituisce quindi
    un dataset parziale contenente solo gli esempi validi.

    I post validi vengono organizzati nei tre split:
        train
        validation
        test

    Per ogni post vengono conservati solamente:
    - il testo;
    - la label numerica corrispondente alla validated_label.

    Ad esempio, se LABELS è:
        ["negative" > 0, "neutral" > 1, "positive" > 2]

    La funzione restituisce quindi una struttura del tipo:
    {
        "train": [...],
        "validation": [...],
        "test": [...]
    }
    contenente solamente dati approvati e pronti per essere
    utilizzati nelle fasi successive del retraining.
    """

    # Prepara un dizionario con una lista vuota per ogni split.
    # MIN_COUNTS ad esempio contiene:
    # {"train": 20, "validation": 5, "test": 5}
    result = {split: [] for split in MIN_COUNTS}
    seen = set()
    for row in load_queue(path):
        if row.get("review_status") != "approved":
            continue
        if row.get("review_is_simulated") is not False:
            raise ValueError("Gli esempi approvati devono avere revisione reale esplicita.")
        if (row.get("validated_label") not in LABELS or row.get("split") not in result
                or not row.get("reviewer") or not row.get("reviewed_at")
                or not row.get("text", "").strip()):
            raise ValueError("Esempio approvato incompleto o non valido.")

        # Creo chiave
        key = text_key(row["text"])
        # se la chiave è in seen allora esiste un duplicato
        if key in seen:
            raise ValueError("Testo duplicato tra gli esempi approvati.")
        seen.add(key)
        # validated_label contiene una stringa: "negative" / "neutral" / "positive"
        # Per il training vogliamo invece la label numerica.
        # uso LABELS.index(...) per la conversione.
        result[row["split"]].append({"text": row["text"],
                                     "label": LABELS.index(row["validated_label"])})
    return result


def readiness(path=QUEUE_PATH):
    """
    Controlla se sono stati raccolti abbastanza esempi approvati
    per poter avviare il retraining.

    Prima di tutto usa approved_splits() per recuperare i dati approvati.
    Successivamente conta quanti esempi di ogni classe sono disponibili
    in ciascuno split. (MIN_COUNTS argginto per ogni split)
    Per poter considerare i dati "pronti", OGNI classe deve raggiungere
    il numero minimo richiesto dal proprio split.

    La funzione restituisce due informazioni:
    1. `ready`
       True se TUTTE le classi di TUTTI gli split raggiungono
       il numero minimo richiesto, False altrimenti.
    2. i conteggi effettivi per ogni split e per ogni classe,
       utili per capire quali dati mancano ancora.
    """
    splits = approved_splits(path)
    counts = {name: Counter(r["label"] for r in rows) for name, rows in splits.items()}
    ready = all(counts[split][label] >= minimum
                for split, minimum in MIN_COUNTS.items() for label in range(3))
    return ready, {split: {LABELS[i]: counts[split][i] for i in range(3)} for split in counts}


def dataset_fingerprint(path=QUEUE_PATH):
    """
    Crea una "firma" del dataset attualmente approvato per il retraining.
    La firma è un hash SHA-256 calcolato utilizzando solamente i dati
    che entrerebbero realmente nel training:
    - testo;
    - etichetta validata;
    - split (train / validation / test).

    Lo scopo è capire se il dataset è cambiato dall'ultimo tentativo
    di retraining.
    Se la firma attuale è uguale a quella salvata in precedenza,
    significa che i dati utilizzabili per il training sono gli stessi
    e quindi non è necessario ripetere lo stesso retraining.
    """
    splits = approved_splits(path)
    for rows in splits.values():
        rows.sort(key=lambda row: text_key(row["text"]))
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def training_budget(path=QUEUE_PATH, max_total=1500, max_replay=500):
    """
    Calcola quanti esempi utilizzare nel prossimo retraining.
    Il retraining utilizza due gruppi di dati:
    1. DATI NUOVI
       Post revisionati e approvati dagli utenti/revisori.
    2. DATI DI REPLAY
       Esempi provenienti da TweetEval, utilizzati insieme ai nuovi dati.
    Con i valori predefiniti l'obiettivo massimo è:
        1000 nuovi + 500 replay = 1500 esempi totali
    Tuttavia, la funzione adatta automaticamente questo budget alla
    quantità di dati nuovi realmente disponibili nel train.
    Per mantenere un dataset bilanciato tra le tre classi, guarda
    quanti esempi sono disponibili nella classe meno numerosa, 
    adattando a quel numero anche le altre classi.
    Esempio:
        negative = 100 |     neutral  = 80 |     positive = 20
    Viene quindi costruito un insieme bilanciato contenente al massimo:
        negative = 20  |     neutral  = 20 |     positive = 20

    Prima di calcolare il budget, la funzione controlla inoltre che:
    - i parametri max_total e max_replay siano validi;
    - il file contenente i dati revisionati esista;
    - readiness() confermi che siano già disponibili almeno
      i dati minimi richiesti per ogni classe e ogni split.

    IMPORTANTE:
    questa funzione calcola solamente quanti esempi nuovi e di replay
    train.py dovrà successivamente richiedere.

    Restituisce:
    - numero totale di esempi da utilizzare;
    - numero di esempi di replay;
    - conteggi degli esempi approvati per split e classe.
    """
    # 1. CONTROLLO DEL BUDGET RICHIESTO
    # max_total - max_replay rappresenta quanti dati NUOVI vogliamo utilizzare.
    if max_total - max_replay < 3 or max_replay < 0:
        raise ValueError("Budget non valido: servono almeno 3 testi nuovi.")
    # 2. CONTROLLO DELL'ESISTENZA DEI DATI
    if not Path(path).is_file():
        raise ValueError(f"Dataset interno assente: {path}. Raccogli e revisiona i post prima del training.")
    # 3. CONTROLLO DELLA QUANTITÀ MINIMA DI DATI
    ready, counts = readiness(path)
    if not ready:
        raise ValueError(f"Etichette approvate insufficienti. Minimi per classe: {MIN_COUNTS}. Disponibili: {counts}")
    # 4. BUDGET IDEALE DEI DATI NUOVI
    requested_new = max_total - max_replay
    # 5. ADATTA IL BUDGET AI DATI REALMENTE DISPONIBILI
    # new_count numero di esempi tot (min(requested_new) * 3 (train, test, split))
    new_count = min(requested_new, min(counts["train"].values()) * 3)
    # 6. CALCOLA LA QUOTA DI REPLAY
    # Se abbiamo dovuto ridurre i dati nuovi, riduciamo
    # proporzionalmente anche i dati di replay.
    # nuovi : replay =  1000 : 500 = cioè circa 2 : 1.
    replay_count = round(new_count * max_replay / requested_new)
    # Se il calcolo produce una quota di replay maggiore di zero
    # ma inferiore a 3, viene considerata troppo piccola.
    if 0 < replay_count < 3:
        raise ValueError("Quota replay troppo piccola: aumentare il budget oppure usare zero.")
    return new_count + replay_count, replay_count, counts
