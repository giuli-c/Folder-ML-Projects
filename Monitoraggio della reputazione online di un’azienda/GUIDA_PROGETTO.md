# Guida Progetto: Monitoraggio della Reputazione Online di un'Azienda

> **Azienda**: MachineInnovators Inc. — sviluppo di applicazioni di machine learning scalabili e pronte per la produzione
> **Problema**: monitorare manualmente il sentiment degli utenti sui social media è inefficiente, soggetto a errori umani e troppo lento per intervenire prima che un calo di reputazione diventi un problema pubblico
> **Modello richiesto**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — usato in inferenza diretta, **senza fine-tuning** nel notebook
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
│   └── monitor.yml           # job "monitor", schedulato (cron) + trigger manuale
└── sentiment_reputation_mlops/
    ├── requirements.txt      # dipendenze del repository (transformers, gradio, requests, ecc.)
    ├── config.py             # costanti centralizzate: modello, dataset, soglie, repo HuggingFace
    ├── predictor.py          # SentimentPredictor: carica il modello una volta, espone predict()
    ├── app.py                # demo Gradio, usa SentimentPredictor da predictor.py
    ├── train.py              # retraining su dati mai visti dal modello base + gate di promozione
    ├── monitor.py            # monitoraggio del sentiment su post reali (Mastodon), con baseline storica
    ├── deploy_to_hf.py       # pubblica questa cartella come HuggingFace Space
    ├── README.md             # frontmatter richiesto da HuggingFace Space (sdk: gradio, app_file: app.py)
    ├── conftest.py           # vuoto: serve solo perche' pytest trovi predictor.py da tests/
    ├── .gitignore
    ├── monitoring/history.json  # baseline storica, aggiornata automaticamente dal job "monitor"
    └── tests/
        ├── test_smoke.py     # test_model_loads, test_known_examples, casi limite, schema di output
        └── test_app.py       # verifica che app.py (Gradio) funzioni, non solo predictor.py
```

Il workflow `ci.yml` sta alla **vera radice del repository** GitHub, non dentro `sentiment_reputation_mlops/`: GitHub Actions legge i workflow solo da `.github/workflows/` nella radice del repository ricevuto da un push, mai da una sottocartella — se restasse annidato nella cartella applicativa, la pipeline non partirebbe mai. Un filtro `paths` lo fa comunque scattare solo quando cambia qualcosa dentro `sentiment_reputation_mlops/`.

### `config.py` — configurazione centralizzata della repository

Stesso principio della `Config` del notebook (sezione 3), ma per il codice che vive nella repository: nome del modello, dataset di retraining, dataset di benchmark originale, repository HuggingFace di destinazione (Space e modello), soglie di tolleranza e di alert. Un solo punto da modificare invece di costanti duplicate in `predictor.py`, `train.py`, `deploy_to_hf.py`, `monitor.py`.

### Job `test` (Fase 2 — test di integrazione)

Installa le dipendenze ed esegue `pytest` ad ogni push o pull request su `main` che tocchi `sentiment_reputation_mlops/`. `app.py` e `tests/test_smoke.py` importano entrambi `SentimentPredictor` da `predictor.py`, che accentra il caricamento del modello in un solo posto — `conftest.py` (vuoto) è necessario perché pytest aggiunga la radice del repository a `sys.path`, altrimenti l'import di `predictor` da dentro `tests/` fallirebbe con `ModuleNotFoundError`.

`tests/test_smoke.py` verifica: che il modello carichi e restituisca il formato atteso; che due frasi non ambigue vengano classificate nella classe corretta (piccolo test di regressione); lo **schema di output** (etichetta tra le tre valide, confidence in `[0, 1]`) su casi limite — stringa vuota, testo molto lungo (oltre i 128 token di truncation), lingua diversa dall'inglese, emoji. `tests/test_app.py` verifica che anche `app.py` — non solo `predictor.py` — funzioni davvero, importando il modulo e chiamando la sua funzione `predict()`.

### Job `deploy` (Fase 2/3 — deploy su HuggingFace)

`needs: test`, gira solo su push diretto a `main` (mai sulle pull request — i secret non sono comunque disponibili alle PR da fork, ed è corretto così: non si deploya codice non ancora mergiato). Usa `huggingface_hub` (`deploy_to_hf.py`) per pubblicare `app.py`/`predictor.py`/`requirements.txt`/`README.md` come HuggingFace Space, creandolo al primo deploy se non esiste ancora (`create_repo(..., exist_ok=True)`). Richiede il secret `HF_TOKEN` (un token HuggingFace con permessi di scrittura) configurato su GitHub in *Settings → Secrets and variables → Actions*.

### Job `train` (Fase 2 — training automatizzato)

Trigger manuale (`workflow_dispatch`, dalla tab *Actions* di GitHub): un fine-tuning, anche piccolo, su runner CPU gratuiti può richiedere diversi minuti, e non ha senso farlo scattare per un commit qualsiasi. Esegue `train.py`, che riallena il modello su **`mteb/tweet_sentiment_extraction`** — deliberatamente **diverso** dal dataset di valutazione (`tweet_eval`), perché il modello base è già stato fine-tuned proprio su TweetEval per il task di sentiment (lo dice la sua model card su HuggingFace): riallenarlo sugli stessi dati non introdurrebbe nessuna informazione nuova. Il nuovo dataset usa lo stesso schema di etichette (0=negative, 1=neutral, 2=positive), quindi nessun remapping aggiuntivo.

Il confronto prima/dopo viene fatto su **due** test set: quello del dataset nuovo (misura se il fine-tuning aiuta davvero su dati mai visti) e un campione di `tweet_eval` (controllo di regressione, per verificare che il modello non abbia "dimenticato" quello che sapeva già fare bene — *catastrophic forgetting*). Il modello riaddestrato viene pubblicato su un repository HuggingFace dedicato (`RETRAINED_MODEL_REPO_ID`) **solo se** il calo di F1 macro sul benchmark originale resta entro `REGRESSION_TOLERANCE`: altrimenti lo script si interrompe con un errore esplicito e non pubblica nulla. La promozione del modello candidato a "modello in produzione" (aggiornare `MODEL_NAME` in `config.py`) resta comunque una decisione manuale, non automatica.

### Job `monitor` (Fase 3 — monitoraggio continuo)

Parte solo a mano per ora (`workflow_dispatch`, tab Actions) — il cron
giornaliero è nel file ma commentato, pronto da riattivare togliendo il
commento. Esegue `monitor.py`, che scarica testi pubblici reali dalla timeline pubblica di un'istanza Mastodon (nessuna autenticazione richiesta) — a differenza del retraining, qui **non serve nessuna etichetta**: il monitoraggio del drift di sentiment si basa solo sulle predizioni del modello su testo fresco, non sull'accuratezza rispetto a una verità nota. I testi vengono classificati con il modello attuale (`predictor.SentimentPredictor`), si calcola la quota di sentiment negativo del batch e la si confronta con una baseline storica (media + 1 deviazione standard delle esecuzioni precedenti — stessa logica statistica di una baseline classica, ma su dati reali accumulati nel tempo). Il risultato di ogni esecuzione viene salvato in `monitoring/history.json`, che il workflow **ricommitta nel repository** ad ogni run: è così che la baseline cresce davvero nel tempo, invece di ripartire da zero ad ogni esecuzione. Se la quota supera la soglia statistica o l'incremento supera la soglia di business, il job stampa un'annotazione `::warning::` visibile nella pagina del job GitHub Actions.

**Prima della consegna**: creare il repository, pushare tutto il contenuto (inclusi i workflow alla radice), configurare il secret `HF_TOKEN`, e incollare il link reale nella cella `GITHUB_REPOSITORY_URL` in cima al notebook (oggi contiene ancora un placeholder).

## 9. Limiti e sviluppi futuri

- Il modello è usato così com'è, senza fine-tuning su dati reali dell'azienda nella valutazione principale — le prestazioni misurate su `tweet_eval` sono un indicatore generale, non una garanzia sul dominio specifico di MachineInnovators.
- Il job `train` allena su un dataset pubblico generico (`tweet_sentiment_extraction`), non su menzioni reali dell'azienda: risolve il problema di riallenare su dati già visti dal modello, ma non quello di specializzarlo sul dominio specifico di MachineInnovators — servirebbero correzioni umane reali o dati aziendali etichettati, che oggi non esistono.
- Il job `monitor` osserva un campione generico della timeline pubblica di Mastodon, non menzioni reali dell'azienda (che richiederebbero API social con accesso a pagamento, es. Twitter/X, o filtri per hashtag/keyword specifici — facilmente aggiungibili in futuro).
- La regola di alert (media + 1 deviazione standard) resta un'euristica semplice, non un test statistico di drift rigoroso (es. Kolmogorov-Smirnov, Population Stability Index).
- Il job `deploy` pubblica `app.py` su HuggingFace Space ad ogni push su `main` che superi i test, ma senza canary/rollback: se una modifica passa i test ma si comporta male in produzione, non c'è un meccanismo automatico per tornare alla versione precedente.
- Manca versionamento sistematico di dati/modello e tracciamento degli esperimenti (es. MLflow, Weights & Biases) per confrontare più run di training nel tempo.
