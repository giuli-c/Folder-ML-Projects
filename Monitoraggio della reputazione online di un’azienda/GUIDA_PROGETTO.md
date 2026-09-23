# Guida Progetto: Monitoraggio della Reputazione Online di un'Azienda

> **Azienda**: MachineInnovators Inc. — sviluppo di applicazioni di machine learning scalabili e pronte per la produzione
> **Problema**: monitorare manualmente il sentiment degli utenti sui social media è inefficiente, soggetto a errori umani e troppo lento per intervenire prima che un calo di reputazione diventi un problema pubblico
> **Modello richiesto**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — usato in inferenza diretta, **senza fine-tuning** nel notebook
> **Dataset pubblico**: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), subtask `sentiment`
> **File notebook**: `Monitoraggio_Reputazione_Online_MLOps.ipynb`
> **Repository CI/CD**: [`sentiment_reputation_mlops/`](sentiment_reputation_mlops/)

> **Nota**: questo documento spiega **cosa** è stato costruito e **come** funziona (guida tecnica completa, notebook + repository). Per **perché** è stato fatto così — teoria, alternative scartate, domande d'esame con risposta — vedi [`SCELTE_PROGETTUALI_ESAME.md`](SCELTE_PROGETTUALI_ESAME.md). Per **come clonare ed eseguire** il codice passo-passo, vedi [`sentiment_reputation_mlops/README.md`](sentiment_reputation_mlops/README.md) (che è anche il README pubblico del repository e della demo HuggingFace Space).

## Indice

1. [Obiettivo e contesto](#1-obiettivo-e-contesto)
2. [Setup ambiente e installazione librerie](#2-setup-ambiente-e-installazione-librerie)
3. [Configurazione centralizzata (`Config`)](#3-configurazione-centralizzata-config)
4. [Dataset: caricamento e controllo qualità](#4-dataset-caricamento-e-controllo-qualità)
5. [Analisi esplorativa e distribuzione delle classi](#5-analisi-esplorativa-e-distribuzione-delle-classi)
6. [Caricamento del modello e funzioni modulari](#6-caricamento-del-modello-e-funzioni-modulari)
7. [Inferenza e valutazione delle performance](#7-inferenza-e-valutazione-delle-performance)
8. [Pipeline CI/CD e sistema di monitoraggio](#8-pipeline-cicd-e-sistema-di-monitoraggio)
9. [Limiti e sviluppi futuri](#9-limiti-e-sviluppi-futuri)

## 1. Obiettivo e contesto

La traccia si divide in tre fasi: **Fase 1**, usare (non addestrare da zero) un modello di sentiment analysis già pronto su un dataset pubblico; **Fase 2**, costruire una pipeline CI/CD automatizzata per training, test di integrazione e deploy dell'applicazione su HuggingFace; **Fase 3**, un sistema di monitoraggio continuo delle performance del modello e del sentiment rilevato, con deploy su HuggingFace (facoltativo).

Il **notebook** copre la Fase 1 per intero (caricamento modello e dataset, valutazione, documentazione dei risultati) e contiene il link al repository GitHub richiesto dalla consegna. Le Fasi 2 e 3 **non sono simulate nel notebook**: sono implementate come codice reale nella repository — pipeline GitHub Actions, script Python eseguibili, test — descritto nella sezione 8 di questa guida e nella sezione 10 del notebook. Questa separazione è intenzionale: la consegna chiede una pipeline e un sistema di monitoraggio automatizzati, non una loro rappresentazione dentro un notebook.

## 2. Setup ambiente e installazione librerie

Il notebook è pensato per Google Colab, ma gira anche in un ambiente Jupyter locale con le stesse librerie. La prima cella installa tutto il necessario:

```python
!pip install -q transformers datasets evaluate accelerate scikit-learn matplotlib seaborn pandas numpy gradio huggingface_hub
```

## 3. Configurazione centralizzata (`Config`)

Una `dataclass Config` raccoglie in un solo punto i parametri usati dal notebook (seed, modello, dataset, dimensione del campione di valutazione, batch size):

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
```

Cambiare modello o campione di valutazione richiede di toccare un solo punto del notebook, non di rincorrere costanti sparse in celle diverse. La repository CI/CD ha una propria configurazione centralizzata equivalente (`sentiment_reputation_mlops/config.py`, sezione 8) — indipendente da questa perché il notebook gira da solo su Colab, senza accesso ai file della repository.

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

## 8. Pipeline CI/CD e sistema di monitoraggio

Questa sezione copre le Fasi 2 e 3 della consegna. Tutto il codice descritto qui è reale ed eseguibile, vive nella repository GitHub — non nel notebook — nella cartella [`sentiment_reputation_mlops/`](sentiment_reputation_mlops/) e nei workflow alla radice del repository:

```
<radice del repository GitHub>
├── .github/workflows/
│   ├── ci.yml                # job "test" (pytest) + job "deploy" (HuggingFace Space, dopo i test)
│   ├── train.yml             # job "train", trigger manuale (workflow_dispatch)
│   ├── train-reviewed.yml    # job "train-reviewed", scatta sui commit a review_queue.json + manuale
│   └── monitor.yml           # job "monitor", trigger manuale (cron giornaliero presente ma commentato)
└── sentiment_reputation_mlops/
    ├── requirements.txt      # dipendenze del repository (transformers, gradio, requests, ecc.)
    ├── config.py             # costanti centralizzate: modello, dataset, soglie, repo HuggingFace
    ├── predictor.py          # SentimentPredictor: carica il modello una volta, espone predict()
    ├── app.py                # demo Gradio, usa SentimentPredictor da predictor.py
    ├── train.py              # retraining (dataset interno approvato o pubblico) + gate a due livelli
    ├── review_data.py        # coda di revisione umana: raccolta, validazione, split, budget
    ├── human_retrain.py      # decide se/come lanciare train.py sui dati approvati (usato da train-reviewed.yml)
    ├── approve_reviewed.py   # completa review_status/reviewer/reviewed_at dopo un'approvazione a mano
    ├── monitor.py            # monitoraggio del sentiment su post reali (Mastodon), con baseline storica
    ├── deploy_to_hf.py       # pubblica questa cartella come HuggingFace Space
    ├── README.md             # frontmatter richiesto da HuggingFace Space (sdk: gradio, app_file: app.py)
    ├── conftest.py           # vuoto: serve solo perche' pytest trovi predictor.py da tests/
    ├── .gitignore
    ├── monitoring/
    │   ├── history.json          # baseline storica, aggiornata automaticamente dal job "monitor"
    │   ├── review_queue.json     # coda di revisione umana (raccolta da monitor.py, revisionata a mano)
    │   └── retraining_state.json # ultimo tentativo di training sui dati approvati (fingerprint, esito)
    └── tests/
        ├── test_app.py        # predizioni, schema di output, casi limite, interfaccia Gradio
        ├── test_data.py       # raccolta Mastodon, revisione umana, duplicati e split
        └── test_training.py   # precontrolli, replay, checkpoint, report, avvio automatico
```

Il workflow `ci.yml` sta alla **vera radice del repository** GitHub, non dentro `sentiment_reputation_mlops/`: GitHub Actions legge i workflow solo da `.github/workflows/` nella radice del repository ricevuto da un push, mai da una sottocartella — se restasse annidato nella cartella applicativa, la pipeline non partirebbe mai. Un filtro `paths` lo fa comunque scattare solo quando cambia qualcosa dentro `sentiment_reputation_mlops/`.

### `config.py` — configurazione centralizzata della repository

Stesso principio della `Config` del notebook (sezione 3), ma per il codice che vive nella repository: nome del modello, dataset di retraining, dataset di benchmark originale, repository HuggingFace di destinazione (Space e modello), soglie di tolleranza e di alert, soglie della coda di revisione umana (`REVIEW_CONFIDENCE_THRESHOLD`, `REVIEW_AUDIT_RATE`, `MONITOR_KEYWORDS`). Un solo punto da modificare invece di costanti duplicate in `predictor.py`, `train.py`, `deploy_to_hf.py`, `monitor.py`, `review_data.py`.

### Job `test` (Fase 2 — test di integrazione)

Installa le dipendenze ed esegue `pytest` ad ogni push o pull request su `main` che tocchi `sentiment_reputation_mlops/`. `app.py` e `tests/test_app.py` importano entrambi `SentimentPredictor` da `predictor.py`, che accentra il caricamento del modello in un solo posto — `conftest.py` (vuoto) è necessario perché pytest aggiunga la radice del repository a `sys.path`, altrimenti l'import di `predictor` da dentro `tests/` fallirebbe con `ModuleNotFoundError`.

`tests/test_app.py` verifica: che il modello carichi e restituisca il formato atteso; che due frasi non ambigue vengano classificate nella classe corretta (piccolo test di regressione); lo **schema di output** (etichetta tra le tre valide, confidence in `[0, 1]`) su casi limite — stringa vuota, testo molto lungo (oltre i 128 token di truncation), lingua diversa dall'inglese, emoji. Lo stesso file verifica che anche `app.py` — non solo `predictor.py` — funzioni davvero, importando il modulo e chiamando la sua funzione `predict()`.

### Job `deploy` (Fase 2/3 — deploy su HuggingFace)

`needs: test`, gira solo su push diretto a `main` (mai sulle pull request — i secret non sono comunque disponibili alle PR da fork, ed è corretto così: non si deploya codice non ancora mergiato). Usa `huggingface_hub` (`deploy_to_hf.py`) per pubblicare `app.py`/`predictor.py`/`requirements.txt`/`README.md` come HuggingFace Space, creandolo al primo deploy se non esiste ancora (`create_repo(..., exist_ok=True)`). Richiede il secret `HF_TOKEN` (un token HuggingFace con permessi di scrittura) configurato su GitHub in *Settings → Secrets and variables → Actions*.

### Job `train` (Fase 2 — training automatizzato, dataset interno o pubblico)

Trigger manuale (`workflow_dispatch`, dalla tab *Actions* di GitHub): un fine-tuning, anche piccolo, su runner CPU gratuiti può richiedere diversi minuti, e non ha senso farlo scattare per un commit qualsiasi. Esegue `train.py`, il cui default (`--data-source reviewed`) usa il **dataset interno approvato** (`monitoring/review_queue.json`, si veda "Coda di revisione umana e retraining incrementale" più sotto) — non più solo il dataset pubblico. Passando `--data-source external` si ripete il vecchio esperimento dimostrativo su **`mteb/tweet_sentiment_extraction`**, deliberatamente **diverso** dal dataset di valutazione (`tweet_eval`), perché il modello base è già stato fine-tuned proprio su TweetEval per il task di sentiment (lo dice la sua model card su HuggingFace).

Il training mescola i dati nuovi con un campione del train di `tweet_eval` (**replay**, `--n-replay`): serve a contenere la perdita di prestazioni sul compito originale, senza garanzie. Il backbone di RoBERTa resta **congelato**, si allena solo la testa di classificazione. Ad ogni epoca, `ValidationCheckpoint` valuta due validation separate (dati nuovi + TweetEval) e conserva in RAM solo la testa che (a) migliora la F1 sui dati nuovi di almeno `min_delta` e (b) non fa scendere la F1 sulla validation originale oltre `REGRESSION_TOLERANCE` — è un **gate a due livelli**: quello durante il training decide quale epoca tenere, un secondo controllo finale (sugli stessi due dataset, ma sui rispettivi *test* set) decide se pubblicare davvero. Se nessuna epoca soddisfa entrambe le condizioni, viene comunque mostrato il confronto dell'ultima epoca a scopo diagnostico (con un avviso rosso), ma il modello **non** viene pubblicato.

`resolve_base_model()` decide da quale modello ripartire ad ogni esecuzione: se esiste già un modello pubblicato su `RETRAINED_MODEL_REPO_ID` (da un run precedente), riparte da lì — i retraining si incatenano invece di ripartire sempre dal modello CardiffNLP originale; altrimenti usa `MODEL_NAME`. È una decisione distinta dalla promozione a produzione: sceglie solo il punto di partenza del *prossimo* training, non cosa serve il traffico reale. La promozione del modello candidato a "modello in produzione" (aggiornare `MODEL_NAME` in `config.py`) resta comunque una decisione manuale, non automatica.

### Job `train-reviewed` (Fase 2 — training automatico dopo un'approvazione umana)

A differenza di `train`, questo workflow **non parte solo a mano**: scatta anche automaticamente ad ogni push su `main` che modifica `monitoring/review_queue.json` (cioè dopo che qualcuno ha approvato/escluso dei post — ma anche dopo un commit del bot di `monitor.yml` che aggiunge solo nuovi post `pending`, si veda il commento in cima al file YAML). Esegue prima `python human_retrain.py --check`, un controllo economico (nessuna dipendenza pesante installata) che verifica se i dati approvati bastano (`review_data.readiness()`) e sono cambiati dall'ultimo tentativo (`review_data.dataset_fingerprint()`); solo se `ready=true` installa le dipendenze e lancia `python human_retrain.py --publish`, che a sua volta chiama `train.py` con gli iperparametri scelti per questo percorso (più conservativi del default "demo": learning rate più basso, più epoche di pazienza) e il budget calcolato da `review_data.training_budget()`. Ogni tentativo, superato o rifiutato, viene registrato in `monitoring/retraining_state.json` (ricommittato nel repository) per non ripetere lo stesso identico training sugli stessi dati.

### Job `monitor` (Fase 3 — monitoraggio continuo)

Parte solo a mano per ora (`workflow_dispatch`, tab Actions) — il cron
giornaliero è nel file ma commentato, pronto da riattivare togliendo il
commento. Esegue `monitor.py`, che scarica testi pubblici reali dalla timeline pubblica di un'istanza Mastodon (nessuna autenticazione richiesta) — a differenza del retraining, qui **non serve nessuna etichetta**: il monitoraggio del drift di sentiment si basa solo sulle predizioni del modello su testo fresco, non sull'accuratezza rispetto a una verità nota. I testi vengono classificati con il modello attuale (`predictor.SentimentPredictor`), si calcola la quota di sentiment negativo del batch e la si confronta con una baseline storica (media + 1 deviazione standard delle esecuzioni precedenti — stessa logica statistica di una baseline classica, ma su dati reali accumulati nel tempo). Il risultato di ogni esecuzione viene salvato in `monitoring/history.json`, che il workflow **ricommitta nel repository** ad ogni run: è così che la baseline cresce davvero nel tempo, invece di ripartire da zero ad ogni esecuzione. Se la quota supera la soglia statistica o l'incremento supera la soglia di business, il job stampa un'annotazione `::warning::` visibile nella pagina del job GitHub Actions.

Oltre a `history.json`, ogni esecuzione alimenta anche `monitoring/review_queue.json` tramite `review_data.enqueue()`: i post con confidence bassa (o un piccolo campione casuale di controllo) vengono accodati per una revisione umana — si veda "Coda di revisione umana e retraining incrementale" più sotto.

### Coda di revisione umana e retraining incrementale

Il monitoraggio non si limita a misurare un drift aggregato: alimenta anche un ciclo reale di **human-in-the-loop**. `monitor.py` accoda in `monitoring/review_queue.json` i post con confidence <= `REVIEW_CONFIDENCE_THRESHOLD` (0,75 di default) più un campione casuale ma deterministico (`REVIEW_AUDIT_RATE`, 10%) degli altri, per controllare anche predizioni che il modello dichiara sicure. Una persona apre quel file e scrive a mano `validated_label`/`review_status` (`approved`/`excluded`) per ogni riga da revisionare — `review_data.py` non assegna mai un'etichetta da solo. Per velocizzare la bookkeeping (senza mai decidere un'etichetta al posto della persona), `approve_reviewed.py` completa `review_status`/`reviewer`/`reviewed_at` per le righe a cui è stato scritto solo `validated_label`.

Solo le righe `approved` (con `review_is_simulated=False`, revisore e data compilati) entrano nel training, tramite `review_data.approved_splits()`; `review_data.readiness()` verifica che ci siano abbastanza esempi per classe in ogni split prima di procedere. Questo intero ciclo è quello descritto nel job `train-reviewed` sopra.

**Prima della consegna**: creare il repository, pushare tutto il contenuto (inclusi i workflow alla radice), configurare il secret `HF_TOKEN`, e incollare il link reale nella cella `GITHUB_REPOSITORY_URL` in cima al notebook (oggi contiene ancora un placeholder).

## 9. Limiti e sviluppi futuri

- Il modello è usato così com'è, senza fine-tuning su dati reali dell'azienda nella valutazione principale — le prestazioni misurate su `tweet_eval` sono un indicatore generale, non una garanzia sul dominio specifico di MachineInnovators.
- Il dataset interno approvato (via `review_data.py`) resta comunque piccolo rispetto a un uso in produzione reale: anche dopo diversi round di raccolta e revisione, si parla di poche decine/centinaia di esempi per classe, non migliaia — sufficiente per dimostrare il meccanismo, non per garantire stime stabili. Con campioni di valutazione troppo piccoli (`--n-eval`/`--n-val` bassi), il calo di F1 misurato su validation e sul test finale può differire abbastanza da cambiare l'esito vicino alla soglia di tolleranza — è per questo che la configurazione di default di `train-reviewed.yml`/`train.yml` usa `--n-eval`/`--n-val` più alti (500) rispetto al minimo tecnicamente sufficiente (200).
- Il job `monitor` osserva un campione generico della timeline pubblica di Mastodon, non menzioni reali dell'azienda (che richiederebbero API social con accesso a pagamento, es. Twitter/X, o filtri per hashtag/keyword specifici — facilmente aggiungibili in futuro).
- La regola di alert (media + 1 deviazione standard) resta un'euristica semplice, non un test statistico di drift rigoroso (es. Kolmogorov-Smirnov, Population Stability Index).
- Il job `deploy` pubblica `app.py` su HuggingFace Space ad ogni push su `main` che superi i test, ma senza canary/rollback: se una modifica passa i test ma si comporta male in produzione, non c'è un meccanismo automatico per tornare alla versione precedente.
- Manca versionamento sistematico di dati/modello e tracciamento degli esperimenti (es. MLflow, Weights & Biases) per confrontare più run di training nel tempo.
