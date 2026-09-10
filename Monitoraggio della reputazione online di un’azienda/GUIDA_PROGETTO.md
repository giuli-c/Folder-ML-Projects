# Guida Progetto: Monitoraggio della Reputazione Online di un'Azienda

> **Azienda**: MachineInnovators Inc. — sviluppo di applicazioni di machine learning scalabili e pronte per la produzione
> **Problema**: monitorare manualmente il sentiment degli utenti sui social media è inefficiente, soggetto a errori umani e troppo lento per intervenire prima che un calo di reputazione diventi un problema pubblico
> **Modello richiesto**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — usato in inferenza diretta, **senza fine-tuning**
> **Dataset pubblico**: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), subtask `sentiment`
> **File notebook**: `Monitoraggio_Reputazione_Online_MLOps.ipynb`
> **Repository CI/CD**: [`sentiment_reputation_mlops/`](sentiment_reputation_mlops/)

## Indice

1. [Obiettivo e contesto](#1-obiettivo-e-contesto)
2. [Setup ambiente e installazione librerie](#2-setup-ambiente-e-installazione-librerie)
3. [Configurazione centralizzata (`Config`)](#3-configurazione-centralizzata-config)
4. [Dataset: caricamento e controllo qualità](#4-dataset-caricamento-e-controllo-qualità)
5. [Analisi esplorativa e distribuzione delle classi](#5-analisi-esplorativa-e-distribuzione-delle-classi)
6. [Caricamento del modello e funzioni modulari](#6-caricamento-del-modello-e-funzioni-modulari)
7. [Inferenza e valutazione delle performance](#7-inferenza-e-valutazione-delle-performance)
8. [Instradamento a revisione umana (human-in-the-loop)](#8-instradamento-a-revisione-umana-human-in-the-loop)
9. [Monitoraggio continuo della reputazione](#9-monitoraggio-continuo-della-reputazione)
10. [Monitoraggio del modello e regole di retraining](#10-monitoraggio-del-modello-e-regole-di-retraining)
11. [Pipeline CI/CD e repository `sentiment_reputation_mlops/`](#11-pipeline-cicd-e-repository-sentiment_reputation_mlops)
12. [Limiti e sviluppi futuri](#12-limiti-e-sviluppi-futuri)

## 1. Obiettivo e contesto

La traccia chiede di **usare** un modello di sentiment analysis già pronto (non di addestrarne uno), valutarlo su un dataset pubblico, e costruire attorno ad esso l'infrastruttura MLOps che serve a un'azienda per monitorare la propria reputazione online nel tempo: qualità del dato, valutazione, instradamento a revisione umana, monitoraggio continuo, regole di retraining, pipeline CI/CD.

Il notebook non addestra nulla da zero: il valore aggiunto è nel valutare criticamente un modello pre-addestrato e progettare il "contorno" operativo che lo rende utilizzabile in produzione in modo responsabile.

## 2. Setup ambiente e installazione librerie

Il notebook è pensato per Google Colab, ma gira anche in un ambiente Jupyter locale con le stesse librerie. La prima cella installa tutto il necessario:

```python
!pip install -q transformers datasets evaluate accelerate scikit-learn matplotlib seaborn pandas numpy gradio huggingface_hub
```

## 3. Configurazione centralizzata (`Config`)

Una `dataclass Config` raccoglie in un solo punto **tutti** i parametri del progetto — non solo seed/modello/dataset/batch size, ma anche le soglie operative usate più avanti (F1 minimo, confidence media minima, soglia di revisione umana, incremento massimo di sentiment negativo, numero di settimane "correnti" per il monitoraggio):

```python
@dataclass
class Config:
    seed: int = 42
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    dataset_name: str = "cardiffnlp/tweet_eval"
    dataset_task: str = "sentiment"
    max_eval_samples: int = 1000
    batch_size: int = 32
    label_map: Dict[int, str] = None

    min_macro_f1: float = 0.70
    min_average_confidence: float = 0.60
    max_negative_share_increase: float = 0.15
    low_confidence_review_threshold: float = 0.60
    n_current_weeks_for_monitoring: int = 1
```

Cambiare modello, campione di valutazione o una qualunque soglia operativa richiede di toccare un solo punto del notebook, non di rincorrere costanti sparse in celle diverse.

## 4. Dataset: caricamento e controllo qualità

Il dataset è `cardiffnlp/tweet_eval`, subtask `sentiment` — **non** `tweet_eval` senza namespace, che è stato dismesso su HuggingFace Hub (le versioni recenti di `huggingface_hub`/`datasets` richiedono un repository id nel formato `namespace/name`; caricare `tweet_eval` restituisce un `HfUriError`). Il dataset è rimasto disponibile sotto il namespace `cardiffnlp`, la stessa organizzazione che mantiene anche il modello richiesto dalla traccia.

Split caricati e dimensioni reali osservate: **Train** 45.615 esempi, **Validation** 2.000, **Test** 12.284. Il controllo qualità (`data_quality_report`) non trova valori mancanti in nessuno split; 29 testi duplicati nel solo train (~0,06%, ininfluenti dato che il train non viene mai usato per addestrare nulla in questo notebook).

## 5. Analisi esplorativa e distribuzione delle classi

Tre osservazioni concrete emerse dai grafici:

- **Distribution shift tra split**: Train e Validation hanno una distribuzione simile (`negative` ~15%, `neutral` ~44%, `positive` ~40%), ma il **Test** è composto diversamente — `negative` sale al 32,3%, `positive` scende al 19,3%, `neutral` resta la classe più frequente (48,3%). È una caratteristica reale dello split ufficiale del dataset, non un artefatto del campionamento fatto nel notebook, e va tenuta presente interpretando le metriche finali (calcolate sul test).
- **Lunghezza dei testi**: mediana simile tra le classi (~19-21 parole) — non è una feature che distingue il sentiment.
- **Esempi testuali**: linguaggio tipico di Twitter/X (informale, hashtag, menzioni, abbreviazioni); il sentiment dipende dal significato complessivo della frase, non da singole parole — coerente con la scelta di non fare preprocessing "classico" (niente stopword removal, niente TF-IDF: un Transformer pre-addestrato lavora meglio su testo vicino alla sua forma naturale di pre-training).

## 6. Caricamento del modello e funzioni modulari

Il modello viene caricato con `AutoTokenizer`/`AutoModelForSequenceClassification` e incapsulato in una `pipeline("sentiment-analysis", ...)`, con `truncation=True, max_length=128` (i tweet sono testi brevi, 128 token bastano nella stragrande maggioranza dei casi).

Le funzioni di utilità (sezione 7 del notebook) separano le responsabilità e vengono riusate più volte:

- `normalize_model_label` — normalizza eventuali etichette tipo `LABEL_0` nei nomi sentiment.
- `predict_sentiment` — inferenza in batch, restituisce predizioni e confidence.
- `evaluate_sentiment_model` — calcola accuracy, precision/recall/F1 macro e weighted.
- `sample_for_test` — campiona il test set per rendere l'esecuzione sostenibile.
- `evaluate_negative_share_alert` — la logica di alert sul sentiment negativo (dettagliata in sezione 9), riusata sia sui dati reali sia su uno scenario dimostrativo sintetico.

## 7. Inferenza e valutazione delle performance

Valutazione su un campione di 1.000 testi del test set. Risultati osservati:

| Metrica | Valore |
|---|---:|
| Accuracy | 0,70 |
| Precision macro | 0,699 |
| Recall macro | 0,710 |
| F1 macro | 0,702 |
| F1 weighted | 0,698 |

Per classe: `negative` è la più riconosciuta (F1 = 0,73, recall = 0,79), `positive` equilibrata (F1 = 0,70), `neutral` la più difficile (recall = 0,63, F1 = 0,67).

Dalla matrice di confusione: gli errori passano soprattutto attraverso `neutral` (110 `neutral`→`negative`, 63 `negative`→`neutral`, 59 `neutral`→`positive`, 53 `positive`→`neutral`); la confusione diretta `positive`↔`negative` è molto bassa (6 e 9 casi). Dal grafico di confidence: `positive` e `negative` hanno confidence generalmente alta, `neutral` ha una distribuzione molto più dispersa — il modello è meno sicuro proprio dove sbaglia di più.

## 8. Instradamento a revisione umana (human-in-the-loop)

Una confidence alta non garantisce una previsione corretta. La coda di revisione (`review_queue_df`) seleziona **tutti e tre** i sentiment predetti con confidence sotto `cfg.low_confidence_review_threshold`, non solo `negative`: limitarsi a `negative` lascerebbe fuori proprio gli errori più pericolosi per il monitoraggio reputazionale — un testo `negative` reale che il modello etichetta, con poca sicurezza, come `neutral` o `positive`, e che quindi non genererebbe mai un alert.

**Validazione della soglia** (possibile solo qui, dove il benchmark porta con sé l'etichetta vera `sentiment`, che in produzione non esisterà mai): il tasso di errore nella coda di revisione è del 48,57%, contro il 30,00% medio sull'intero campione — 1,6x più alto. Conferma che la confidence è un segnale utile, ma non un rilevatore di errori affidabile al 100%: più della metà dei testi in coda sono in realtà predizioni corrette.

**In produzione**, dove questa validazione non è possibile (nessuna etichetta vera per i tweet reali), la coda di revisione andrebbe esportata in CSV (testo, sentiment predetto, confidence, colonna vuota `human_label`) per un revisore umano — un passaggio che il notebook non può eseguire davvero (gira in batch). Le correzioni umane accumulate nel tempo costituirebbero il dataset etichettato specifico dell'azienda necessario per un eventuale fine-tuning futuro.

## 9. Monitoraggio continuo della reputazione

Il dataset non ha una dimensione temporale reale: vengono generate date fittizie (`pd.date_range`, frequenza oraria a partire dal 1° gennaio 2026) solo per simulare come si costruirebbe una dashboard con dati reali da API social o strumenti di listening. I testi vengono aggregati per settimana (~6 settimane con 1.000 campioni) e per ciascuna si calcola la quota di `negative`/`neutral`/`positive` (`unstack(fill_value=0)` garantisce che ogni settimana compaia con tutte e tre le classi, anche a 0, evitando che una settimana senza `negative` sparisca silenziosamente dalla serie).

**Alert sul sentiment negativo** (`evaluate_negative_share_alert`): la baseline (media + 1 deviazione standard) si calcola **solo sulle settimane precedenti** al periodo corrente (l'ultima settimana), non sull'intera serie — usare l'intera serie per giudicare il suo stesso ultimo punto sarebbe circolare, e renderebbe l'alert meno sensibile proprio quando servirebbe di più. Vengono restituiti due segnali indipendenti: un alert **statistico** (quota corrente > baseline + 1 std) e un alert di **business** (incremento assoluto > `cfg.max_negative_share_increase`, 15% di default).

Con il campione usato in questa esecuzione l'alert **statistico** scatta già (la quota corrente supera la baseline storica), mentre quello di **business** no (l'incremento resta sotto i 15 punti percentuali): un segnale borderline, non ancora una crisi reputazionale netta — ma non è garantito che il risultato sia lo stesso a ogni esecuzione, dato che le settimane simulate dipendono dalla data corrente (`pd.Timestamp.now()`). Per verificare che la logica riconosca anche un caso inequivocabile, lo stesso tipo di scenario **sintetico**, con un incremento netto che supera entrambe le soglie, viene riusato più avanti come terzo scenario (`REPUTATION_ALERT`) nella sezione 12.1, invece di restare una dimostrazione isolata.

## 10. Monitoraggio del modello e regole di retraining

La tabella `monitoring_summary` (costruita dalla funzione riusabile `build_monitoring_summary`, sezione 7) combina quattro controlli, tutti collegati alle soglie di `Config`: F1 macro minimo, confidence media minima, quota di sentiment negativo (soglia statistica), incremento di sentiment negativo (soglia di business). Separa concettualmente due problemi diversi: un calo di F1 macro segnala che il **modello** potrebbe non essere più affidabile (possibile retraining); un aumento del sentiment negativo con un modello che funziona bene segnala una **vera crisi reputazionale** (va avvisato il team business, non riaddestrato il modello).

**Tre scenari sintetici** (sezione 12.1) dimostrano che ciascuno dei tre stati non-`OK` (`RETRAINING_CANDIDATE`, `REVIEW_REQUIRED`, `REPUTATION_ALERT`) scatta davvero quando le condizioni si verificano, isolando una causa alla volta — lo stesso principio del già citato scenario sintetico dell'alert in sezione 9.

**Da stato a azione (sezione 12.2)**: ciascuno dei tre stati è collegato a un'azione concreta, e ognuna parte solo se lo stato **reale** (non lo scenario sintetico) calcolato in `monitoring_summary` lo richiede — un booleano dedicato per azione (`needs_retraining`, `needs_review`, `needs_reputation_check`) legge la colonna `status` della tabella reale. Con i dati di questa esecuzione solo `needs_reputation_check` risulta vero; per le altre due un flag esplicito (`FORCE_RETRAINING_DEMO`, `FORCE_REVIEW_DEMO = True`, sullo stesso principio di `RUN_GRADIO_DEMO` usato più avanti nel notebook, sezione 14) forza comunque la dimostrazione, ed è commentato come "da rimuovere in produzione". Le azioni di revisione umana e di analisi reputazionale non ricalcolano nulla: riusano rispettivamente `review_queue_df` (sezione 8) e i testi negativi di `monitor_df` nella settimana corrente (sezione 9).

**Il retraining, reso eseguibile (sezioni 12.3-12.5)**: `get_retraining_dataset()` cerca un dataset di correzioni umane (`human_corrected_df`, che in produzione arriverebbe dalla sezione 8) e, non trovandolo in questa esecuzione, usa come default un campione stratificato di `train_df`. Su quel dataset viene eseguito un **fine-tuning dimostrativo minimo** (1 epoca, batch size 8, learning rate 2e-5, tramite `transformers.Trainer`) su una **copia separata** del modello (`retrain_model`): `model`/`sentiment_pipeline`, usati nel resto del notebook, non vengono toccati. Una cella finale confronta le metriche prima/dopo sullo stesso `sample_test_df`. Resta un meccanismo dimostrativo, non un vero processo di retraining di produzione: il trigger `needs_retraining` esiste ma è coperto dal flag di demo, nessuna validazione decide se il modello riaddestrato è davvero migliore prima di sostituire quello in uso, e il miglioramento non è garantito vista la dimensione ridotta del dataset e delle epoche.

## 11. Pipeline CI/CD e repository `sentiment_reputation_mlops/`

La consegna richiede una repository GitHub pubblica con codice documentato. I file applicativi vivono come file veri nella cartella [`sentiment_reputation_mlops/`](sentiment_reputation_mlops/), non come stringhe dentro il notebook. Il workflow `ci.yml`, invece, sta alla **radice del repository** GitHub (non dentro questa cartella): GitHub Actions legge i workflow solo da `.github/workflows/` nella vera radice del repository ricevuto da un push, mai da una sottocartella — se restasse annidato qui, la pipeline non partirebbe mai.

```
<radice del repository GitHub>
├── .github/workflows/ci.yml   # radice del repo: qui GitHub Actions lo trova davvero;
│                               # un filtro `paths` lo fa scattare solo per questa cartella
└── sentiment_reputation_mlops/
    ├── requirements.txt
    ├── predictor.py          # SentimentPredictor: carica il modello una volta, espone predict()
    ├── app.py                # demo Gradio, usa SentimentPredictor
    ├── conftest.py            # vuoto: serve solo perche' pytest trovi predictor.py da tests/
    ├── .gitignore
    └── tests/
        └── test_smoke.py     # test_model_loads + test_known_examples
```

`app.py` e `tests/test_smoke.py` importano entrambi `SentimentPredictor` da `predictor.py`, invece di caricare il modello ciascuno per conto proprio — stesso principio di modularità della sezione 6/7 del notebook. `conftest.py` è vuoto ma necessario: senza di esso, pytest non aggiungerebbe la radice del repository a `sys.path`, e l'import di `predictor` da dentro `tests/` fallirebbe. Il workflow `ci.yml` installa le dipendenze (con cache pip) ed esegue `pytest` ad ogni push o pull request su `main` che tocchi `sentiment_reputation_mlops/` (`working-directory` nel workflow punta lì, così i comandi girano nella cartella giusta invece che nella radice del repo).

**Prima della consegna**: creare il repository, pushare il contenuto di `sentiment_reputation_mlops/`, e incollare il link reale nella cella `GITHUB_REPOSITORY_URL` in cima al notebook (oggi contiene ancora un placeholder).

## 12. Limiti e sviluppi futuri

- Il modello è usato così com'è, senza fine-tuning su dati reali dell'azienda — le prestazioni misurate su `tweet_eval` sono un indicatore generale, non una garanzia sul dominio specifico di MachineInnovators.
- Il monitoraggio temporale è simulato su dati statici, non su un vero flusso nel tempo.
- La regola di alert (media + 1 deviazione standard) resta un'euristica semplice, non un test statistico di drift rigoroso (es. Kolmogorov-Smirnov, Population Stability Index).
- La coda di revisione umana non ha un ciclo di chiusura reale in questo notebook (nessun export CSV effettivo, nessun merge delle correzioni) — è descritto come si farebbe in produzione, non implementato end-to-end. Di conseguenza, il fine-tuning dimostrativo di sezione 12.3 non trova mai un `human_corrected_df` reale e ricade sempre sul dataset di default.
- Il retraining (sezioni 12.3-12.5) è eseguibile e collegato a un trigger reale (`needs_retraining`, derivato da `monitoring_summary`), ma resta dimostrativo: un flag esplicito (`FORCE_RETRAINING_DEMO`) lo forza comunque quando lo stato reale non lo richiederebbe, e mancano versionamento di dati/modello, tracciamento degli esperimenti, e una validazione automatica che confronti il modello riaddestrato con quello in uso prima di sostituirlo.
