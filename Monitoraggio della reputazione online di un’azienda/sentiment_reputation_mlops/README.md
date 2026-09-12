---
title: Sentiment Reputation Monitor
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Sentiment Reputation Monitor

Demo del modello [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) per la classificazione del sentiment (`negative`/`neutral`/`positive`) di testi social, nell'ambito del progetto "Monitoraggio della reputazione online di un'azienda".

Questo Space viene pubblicato automaticamente dal job `deploy` della pipeline CI/CD (`.github/workflows/ci.yml` nella radice del repository GitHub), dopo che i test in `tests/test_smoke.py` sono passati — non va aggiornato a mano.

---

## Come replicare e usare questo progetto

Il codice vive dentro un monorepo con tutti i progetti d'esame ([`giuli-c/Folder-progetti-ProfessionAI`](https://github.com/giuli-c/Folder-progetti-ProfessionAI)), in questa cartella (`Monitoraggio della reputazione online di un'azienda/sentiment_reputation_mlops/`).
I tre workflow GitHub Actions vivono invece alla **radice** del repository (`.github/workflows/`), ma sono limitati a questa cartella tramite `paths:` e `working-directory:` — se sposti/copi solo questa cartella altrove, dovrai portarti dietro anche quei tre file e aggiustare i percorsi al loro interno.

### 1. Setup locale

```bash
git clone https://github.com/giuli-c/Folder-progetti-ProfessionAI.git
cd "Folder-progetti-ProfessionAI/Monitoraggio della reputazione online di un’azienda/sentiment_reputation_mlops"
```

**Ambiente virtuale** (consigliato: evita di installare le dipendenze di questo progetto nell'ambiente Python globale, dove potrebbero scontrarsi con quelle degli altri progetti d'esame). Va creato **una sola volta**.

Su **Windows**, crealo in un percorso **corto, fuori da questa cartella** (non `.\venv`): il percorso completo di questo progetto è già lungo di suo, e i file interni di `torch`/`transformers` sono annidati molto in profondità (es. `transformers\models\audio_spectrogram_transformer\configuration_audio_spectrogram_transformer.py`) — sommati superano facilmente il limite storico di Windows sui percorsi (260 caratteri, `MAX_PATH`), e l'installazione (o anche solo l'avvio di `app.py` in un secondo momento) fallisce con `FileNotFoundError`/`No such file or directory` su file che in realtà esistono. Un venv fuori da questa cartella, con un nome corto, evita il problema alla radice:

```powershell
python -m venv C:\venvs\sentiment-reputation-monitor
```

Poi va **attivato**:

```powershell
C:\venvs\sentiment-reputation-monitor\Scripts\Activate.ps1
```

Se PowerShell rifiuta di eseguire lo script di attivazione ("l'esecuzione di script è disabilitata su questo sistema"):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

(poi ripetere l'attivazione). Infine, con il venv attivo (prompt con `(sentiment-reputation-monitor)` o `(venv)` davanti):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se `pip install -r requirements.txt` da solo dà "pip non riconosciuto", usare `python -m pip install -r requirements.txt` — più affidabile su Windows perché non dipende dal PATH di `pip` separatamente da quello di `python`.

**Alternativa, se si preferisce comunque un venv dentro la cartella del progetto** su Windows: abilitare il supporto ai percorsi lunghi a livello di sistema (fix permanente, utile anche per gli altri progetti in questa stessa cartella). PowerShell **come Amministratore**, poi **riavviare il PC**:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 2. Provare il modello in locale

```bash
python app.py
```

Apre una pagina Gradio locale dove scrivere un testo e vedere subito `sentiment`/`confidence` — usa `predictor.SentimentPredictor`, lo stesso modulo condiviso da tutto il resto del progetto (vedi `config.MODEL_NAME`).

### 3. Eseguire i test

`pytest` non è in `requirements.txt` (di proposito: serve solo per sviluppare/testare, non per far girare l'app in produzione — anche la CI lo installa a parte). Va installato una volta nel venv:

```bash
pip install pytest
pytest -q
```

Esegue `tests/test_smoke.py` (carica il modello, verifica il formato dell'output, controlla due esempi non ambigui) e `tests/test_app.py`(verifica lo schema di `app.predict()` e che `gr.Interface` si costruisca
senza errori). `conftest.py` (vuoto) è necessario perché pytest trovi `predictor.py` dalla cartella `tests/` — non rimuoverlo.

### 4. Cosa succede automaticamente (CI)

Ad ogni **push o pull request su `main`** che tocca qualcosa dentro questa cartella, `.github/workflows/ci.yml` fa partire due job in sequenza:

1. **`test`**: installa le dipendenze ed esegue `pytest`.
2. **`deploy`** (solo su push diretto a `main`, mai sulle PR): se i test sono passati, esegue `deploy_to_hf.py`, che pubblica `app.py`, `predictor.py`, `requirements.txt` e questo stesso `README.md` come HuggingFace Space (`config.SPACE_REPO_ID`) — creandolo al primo deploy se non esiste ancora.

Non serve fare nulla a mano: basta pushare su `main`.

### 5. Lanciare un retraining (manuale)

Da GitHub → tab **Actions** → workflow **"Train - Monitoraggio reputazione online"** → **Run workflow**, impostando (opzionali, hanno un default):
- `n_train`: numero di esempi di retraining (default 90, divisi per classe);
- `epochs`: epoche di fine-tuning (default 1).

`train.py` riaddestra il modello su `mteb/tweet_sentiment_extraction` (dati mai visti dal modello base, diverso da TweetEval — vedi il docstring in cima al file per il perché), confronta le metriche prima/dopo su **due** test set (quello nuovo + un campione di `tweet_eval` come controllo di regressione), e pubblica il modello candidato su `config.RETRAINED_MODEL_REPO_ID` **solo se** il calo di F1 macro sul benchmark originale resta entro `REGRESSION_TOLERANCE` — altrimenti il job fallisce esplicitamente e non pubblica nulla.

Le esecuzioni successive di `train.py` ripartono automaticamente dall'ultimo modello già pubblicato su `RETRAINED_MODEL_REPO_ID`, se esiste (vedi `resolve_base_model()` in `train.py`) — i retraining si incatenano.

**Promuovere il modello candidato in produzione è una decisione manuale separata**: aggiornare `MODEL_NAME` in `config.py` con il valore di `RETRAINED_MODEL_REPO_ID`. Solo dopo questo edit (commit + push) `predictor.py`
— e quindi `app.py`/`monitor.py`/i test — inizieranno a usare il modello nuovo. Nessun trigger automatico lo fa da solo.

### 6. Monitoraggio continuo (schedulato)

`.github/workflows/monitor.yml` parte **solo a mano** per ora, dalla tab Actions (`workflow_dispatch`) — il cron giornaliero (`0 8 * * *` UTC) è presente nel file ma commentato; basta togliere il commento per farlo
scattare da solo ogni giorno. Ogni esecuzione di `monitor.py`:

1. scarica testi pubblici reali dalla timeline federata di un'istanza Mastodon (`config.MASTODON_INSTANCE`, di default `mastodon.social`) — non serve nessun token, l'accesso è anonimo;
2. li classifica con il modello attuale (`predictor.SentimentPredictor`);
3. calcola la quota di sentiment negativo del batch e la confronta con la baseline delle esecuzioni precedenti (baseline calcolata solo sulle esecuzioni passate, mai includendo quella corrente — vedi il commento nel codice di `monitor.py`);
4. se supera la soglia statistica o quella di business (`MAX_NEGATIVE_SHARE_INCREASE`), stampa un warning visibile nella pagina del job GitHub Actions;
5. salva il risultato in `monitoring/history.json` e lo ricommitta nel repository — senza questo passaggio ogni esecuzione ripartirebbe da zero, senza nessuna storia su cui costruire una baseline.

### 7. Secret richiesti

Da impostare in GitHub → Settings del repository → **Secrets and variables → Actions**:

| Secret | Serve a | Usato da |
|---|---|---|
| `HF_TOKEN` | Token HuggingFace con permessi di **scrittura** sia su Space sia su repository modello (serve un token "Write", non "Read") | job `deploy` di `ci.yml` (pubblica lo Space); `train.yml` (pubblica il modello candidato) |

`monitor.yml` non richiede nessun secret HuggingFace (l'accesso alla timeline pubblica di Mastodon è anonimo); usa invece il `GITHUB_TOKEN` automatico di Actions (già disponibile, non va creato) per ricommittare `monitoring/history.json` — per questo il workflow dichiara `permissions: contents: write`.

### 8. Configurazione centralizzata

Tutte le costanti sopra (`MODEL_NAME`, `RETRAIN_DATASET`,`RETRAINED_MODEL_REPO_ID`, `REGRESSION_TOLERANCE`, `MASTODON_INSTANCE`, `MAX_NEGATIVE_SHARE_INCREASE`, `SPACE_REPO_ID`, ecc.) vivono in un solo punto: `config.py`. Per cambiare modello, dataset di retraining, istanza Mastodon o una qualunque soglia, si modifica solo lì.
