# Guida Progetto: Data Augmentation per la Sicurezza delle Centrali Elettriche

> **Cliente**: CyberEye Solutions — sicurezza cibernetica per infrastrutture critiche
> **Problema**: il sistema di riconoscimento immagini della sorveglianza è addestrato su un dataset troppo piccolo e poco vario
> **Dataset usato come base**: `OxfordIIITPet` (torchvision) — 37 razze di cani e gatti, l'intero split `trainval` (~3.680 immagini) più un test set ufficiale separato (~3.669 immagini)
> **Pipeline**: Image captioning (BLIP) → generazione testuale di varianti (Qwen2.5-1.5B-Instruct) → generazione di immagini sintetiche (sdxl-turbo) → training e confronto di un classificatore su dati reali vs. dati reali + sintetici
> **File notebook**: `Data_Augmentation_Sicurezza_Centrali_Elettriche.ipynb`

> Questa è una guida **tecnica** al codice, sezione per sezione. Per la teoria e le motivazioni da usare in sede d'esame, vedi `SCELTE_PROGETTUALI_ESAME.md`.

## Indice

1. [Obiettivo e perché il dataset dei pet](#1-obiettivo-e-perché-il-dataset-dei-pet)
2. [Struttura del notebook](#2-struttura-del-notebook)
3. [Setup ambiente Colab](#3-setup-ambiente-colab)
4. [Configurazione centralizzata](#4-configurazione-centralizzata)
5. [Dataset e split](#5-dataset-e-split)
6. [Trasformazioni e DataLoader](#6-trasformazioni-e-dataloader)
7. [Classificatore e classi di supporto](#7-classificatore-e-classi-di-supporto)
8. [Esperimento A — Baseline](#8-esperimento-a--baseline)
9. [Pipeline di Data Augmentation generativa](#9-pipeline-di-data-augmentation-generativa)
10. [Valutazione della qualità dei sintetici](#10-valutazione-della-qualità-dei-sintetici)
11. [Esperimento B — Augmented](#11-esperimento-b--augmented)
12. [Metriche di valutazione: quelle richieste e quelle aggiunte](#12-metriche-di-valutazione-quelle-richieste-e-quelle-aggiunte)
13. [Error Analysis](#13-error-analysis)
14. [Come leggere i risultati e completare i commenti](#14-come-leggere-i-risultati-e-completare-i-commenti)
15. [Limiti e sviluppi futuri](#15-limiti-e-sviluppi-futuri)

## 1. Obiettivo e perché il dataset dei pet

La traccia chiede di lavorare su `torchvision.datasets.OxfordIIITPet`, non su immagini reali di centrali elettriche. È una scelta didattica della traccia stessa, e va capita per quello che è: il dataset dei pet riproduce la **struttura** del problema di CyberEye, non il suo contenuto.

| Problema di CyberEye | Equivalente in Oxford-IIIT Pet |
|---|---|
| Poche immagini per ogni tipo di anomalia/comportamento critico | Poche immagini per razza (in media ~80-100 nel training) |
| Classi visivamente simili (es. "quasi normale" vs "sospetto") | Razze di cani/gatti molto simili tra loro (fine-grained classification) |
| Necessità di generalizzare da un dataset scarso | Stesso identico problema, solo con soggetti diversi |

Il progetto verifica una tecnica (Data Augmentation generativa), non un caso d'uso specifico. Le conclusioni quantitative valgono per il dataset dei pet; il metodo, se funziona qui, è il candidato da testare poi sui dati reali di CyberEye.

## 2. Struttura del notebook

Il notebook è organizzato in 16 sezioni numerate (corrispondenti agli header `## N.` effettivamente presenti nel notebook):

1. Setup dell'ambiente Colab e installazione librerie
2. Import delle librerie, seed e device
3. Configurazione centralizzata (`Config`)
4. Caricamento ed esplorazione di Oxford-IIIT Pet
5. Costruzione degli split: training, validation, test
6. Trasformazioni e DataLoader
7. Architettura del classificatore e classi di supporto
8. Esperimento A — Baseline (solo dati reali)
9. Valutazione dell'esperimento Baseline sul test set
10. Pipeline di Data Augmentation generativa
11. Valutazione della qualità dei dati sintetici
12. Esperimento B — Augmented (dati reali + sintetici)
13. Valutazione dell'esperimento Augmented sul test set
14. Confronto finale tra Baseline e Augmented (metriche, grafici, bootstrap)
15. Error Analysis
16. Conclusioni e limiti

**Nota**: le sezioni di questa guida (indice in cima al documento) hanno una numerazione propria e indipendente da quella del notebook appena elencata — questa guida raggruppa insieme, per leggibilità, il training di un esperimento e la sua valutazione sul test set (che nel notebook sono due sezioni separate, es. 8+9 per il Baseline, 12+13 per l'Augmented).

## 3. Setup ambiente Colab

Il notebook installa solo ciò che Colab non ha già preinstallato:

```python
!pip install -q diffusers accelerate transformers timm sentencepiece
```

`torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn` sono già nell'immagine standard di Colab.

Il modello di generazione immagini, `stabilityai/sdxl-turbo`, è pubblico e non richiede token o autenticazione: non è necessaria nessuna configurazione manuale per eseguire il notebook.

## 4. Configurazione centralizzata

Tutti i parametri vivono in una dataclass `Config`: un solo posto dove cercare/cambiare un numero, nessun valore sparso nel codice.

```python
@dataclass
class Config:
    # --- Dati ---
    DATA_ROOT: str = "./data"
    SYNTH_DIR: str = "./data/synthetic"
    IMG_SIZE: int = 224

    # --- Split train/validation sull'intero trainval ---
    TRAIN_SIZE: float = 0.8        # quota (stratificata) del trainval usata per il training

    # --- Pipeline di Data Augmentation generativa ---
    AUGMENT_FRACTION: float = 0.3  # quota (stratificata) del training set da aumentare
    N_TEXT_VARIANTS: int = 1       # varianti testuali generate per ogni immagine campionata
    SYNTH_STEPS: int = 2           # step di diffusione se si usa il modello "turbo"

    # --- Training del classificatore ---
    BATCH_SIZE: int = 32
    EPOCHS: int = 80
    LR_BACKBONE: float = 1e-5      # LR basso: il backbone è già pretrained
    LR_HEAD: float = 1e-3          # LR più alto: la testa parte da pesi casuali
    WEIGHT_DECAY: float = 5e-4
    DROPOUT: float = 0.3           # dropout nella testa (via drop_rate di timm)
    LABEL_SMOOTHING: float = 0.1
    PATIENCE: int = 4              # epoche di pazienza per l'early stopping
    MIN_DELTA: float = 1e-3        # miglioramento minimo di val_loss per non essere rumore

    # --- Modelli pretrained usati (Hugging Face Hub) ---
    CLASSIFIER_MODEL: str = "efficientnet_b0"
    CAPTION_MODEL: str = "Salesforce/blip-image-captioning-base"
    TEXT_GEN_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    IMAGE_GEN_MODEL: str = "stabilityai/sdxl-turbo"
    QUALITY_CHECK_MODEL: str = "openai/clip-vit-base-patch32"
```

I cinque parametri con `_MODEL` nel nome sono "specialisti" che preparano il materiale di training (descrivono foto, riscrivono descrizioni, disegnano immagini, ne giudicano la qualità); `CLASSIFIER_MODEL` è l'unico che viene effettivamente allenato e valutato — è la distinzione più importante da tenere a mente leggendo il resto della guida.

**Nota sui parametri di training**: `EPOCHS=80` non significa che il training dura sempre 80 epoche — è solo il tetto massimo. Nella pratica, l'early stopping (`PATIENCE`, `MIN_DELTA`) interrompe il training molto prima, quando la validation loss smette di migliorare in modo significativo. Il tetto è stato alzato progressivamente nel corso del progetto perché con valori più bassi il training veniva interrotto prima di arrivare a una vera convergenza (vedi `SCELTE_PROGETTUALI_ESAME.md`, sezione 6).

### Tempi indicativi (con i valori di default, GPU T4 su Colab)

| Fase | Tempo stimato |
|---|---|
| Download dataset (~800 MB) | 1-2 minuti |
| Download modelli pretrained (BLIP, Qwen2.5-1.5B, sdxl-turbo, CLIP) | 4-7 minuti |
| Training Baseline (fino a 80 epoche, early stopping) | 15-35 minuti a seconda di quando scatta l'early stopping |
| Captioning del campione per l'augmentation (BLIP) | 1-3 minuti |
| Generazione delle varianti testuali (Qwen2.5-1.5B-Instruct) | 3-6 minuti |
| Generazione delle immagini sintetiche (~880 immagini, 30% del training) | 5-10 minuti con `sdxl-turbo` |
| Training Augmented | 15-35 minuti, stesso ordine di grandezza del Baseline |
| Valutazioni e grafici | 2-3 minuti |

Numeri indicativi, non garantiti: dipendono dalla GPU assegnata dalla sessione Colab gratuita in quel momento e da quando scatta l'early stopping.

## 5. Dataset e split

`OxfordIIITPet(root=..., split="trainval"/"test", target_types="category", download=True)` scarica il dataset ed espone `classes` (37 nomi di razza) e le etichette di ogni immagine.

Tre insiemi disgiunti, costruiti con `train_test_split` di scikit-learn (stratificato, seed fisso) applicato agli **indici** delle immagini, non alle immagini stesse — perché il dataset le carica pigramente da disco e train/validation devono avere trasformazioni diverse (vedi sezione 6 e `SCELTE_PROGETTUALI_ESAME.md`):

- **Training** (`TRAIN_SIZE=0.8`, l'80% dell'intero trainval): il training set di partenza, identico nei due esperimenti principali.
- **Validation** (il restante 20% del trainval, stratificato, esclude per costruzione gli indici di training): usato per l'early stopping, sempre reale, mai aumentato.
- **Test** (split ufficiale `test` di Oxford-IIIT Pet, ~3.669 immagini): mai toccato prima della valutazione finale, usato una sola volta per ciascun modello.

## 6. Trasformazioni e DataLoader

Due pipeline di trasformazione, entrambe definite una sola volta e riusate:

- **`train_transform`**: `Resize(256)` → `RandomResizedCrop(224, scale=(0.65, 1.0))` → `RandomHorizontalFlip` → `RandomRotation(10°)` → `ToTensor` → `Normalize` (statistiche ImageNet). Applicata a tutte le immagini di training, reali e sintetiche, in entrambi gli esperimenti.
- **`eval_transform`**: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`. Nessuna augmentation: usata per validation e test, dove serve una misura deterministica e riproducibile.

Per applicare trasformazioni diverse a train e validation pur partendo dallo stesso pool di immagini (`trainval`), il notebook crea **due dataset separati** (`train_dataset` con `train_transform`, `val_dataset` con `eval_transform`) e li restringe con `Subset` agli indici calcolati in sezione 5.

Una versione di `ColorJitter` era stata aggiunta e poi rimossa: rendeva ogni epoca sensibilmente più lenta su Colab, senza un beneficio misurabile oltre a quello già ottenuto con dropout/weight decay/label smoothing (vedi sezione 7).

## 7. Classificatore e classi di supporto

### EfficientNet-B0, backbone sbloccato, `timm`

Backbone costruito con `timm.create_model(cfg.CLASSIFIER_MODEL, pretrained=True, num_classes=NUM_CLASSES, drop_rate=cfg.DROPOUT)`. Il backbone **non è congelato**: si allena tutto il modello, ma con due learning rate diversi nello stesso ottimizzatore (`LR_BACKBONE` basso, `LR_HEAD` alto) — vedi `SCELTE_PROGETTUALI_ESAME.md` per il perché di questa scelta rispetto a backbone congelato o sblocco progressivo.

`timm` (non `torchvision.models`) perché `num_classes` sostituisce automaticamente la testa finale con un'interfaccia identica per qualunque backbone della libreria, e `drop_rate` espone il dropout come parametro pronto all'uso — cambiare architettura richiede solo di cambiare `CLASSIFIER_MODEL`.

### `Trainer`

Ciclo di training/validazione scritto a mano (non una libreria), con:
- ottimizzatore `AdamW` con due gruppi di parametri (learning rate differenziati per backbone e testa) e `WEIGHT_DECAY`;
- loss `CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)`;
- early stopping basato su `PATIENCE` (epoche di pazienza) e `MIN_DELTA` (soglia minima di miglioramento della validation loss per non essere considerato rumore);
- ripristino automatico, a fine training, dei pesi del checkpoint con la validation loss migliore.

Usato identico per Baseline e Augmented: cambia solo il `DataLoader` di training passato in ingresso.

### `Evaluator`

Calcola su un qualunque `DataLoader`: accuracy, top-3 accuracy, precision/recall/F1 (macro e weighted), `classification_report` per classe (`hardest_classes()`, `most_confused_classes()`), matrice di confusione. Produce anche i grafici (curve di training, matrice di confusione normalizzata). Condivisa tra tutti gli esperimenti, così le metriche sono calcolate esattamente allo stesso modo per ciascuno.

### `Blip`, `Qwen`, `ImageGenerator`, `ClipScorer`

Le quattro classi che incapsulano la pipeline generativa e la valutazione CLIP (sezioni 9-10), ciascuna con lo stesso schema: `load_model()` → un metodo di generazione/valutazione → `clear_gpu()`. Vedi sezione 9 per i dettagli.

### `build_comparison_table()`

Funzione che costruisce la tabella di confronto delle metriche tra Baseline e Augmented, a partire da un dizionario `{nome_modello: risultati_evaluate()}`.

## 8. Esperimento A — Baseline

Training del classificatore **solo** sulle immagini reali di training. Early stopping sul validation set reale. È il punto di riferimento: rappresenta la situazione di partenza di CyberEye, prima di qualunque intervento di Data Augmentation.

## 9. Pipeline di Data Augmentation generativa

Prima si campiona, in modo stratificato, il sottoinsieme del training da aumentare (`AUGMENT_FRACTION=0.3`, cioè il 30% del training, sopra la soglia minima del 20% richiesta dalla traccia). Poi tre passaggi, nell'ordine richiesto dalla traccia, ciascuno incapsulato in una classe dedicata con lo schema carica → usa → libera dalla memoria GPU:

### 9.1 Image captioning — classe `Blip`

`blip = Blip(cfg.CAPTION_MODEL); blip.load_model(); caption_records = blip.run_captioning(augment_idx); blip.clear_gpu()`. Genera una didascalia in inglese per ogni immagine campionata (es. *"a cat sitting on a wooden floor"*). BLIP non conosce le razze del dataset: descrive la scena (posa, contesto, sfondo), non la razza specifica.

### 9.2 Generazione testuale di varianti — classe `Qwen`

`qwen = Qwen(cfg.TEXT_GEN_MODEL); qwen.load_model(); synth_prompts = qwen.run_variant_generation(caption_records, n_variants=cfg.N_TEXT_VARIANTS); qwen.clear_gpu()`. Genera `N_TEXT_VARIANTS` parafrasi della **sola parte descrittiva** prodotta da BLIP, tramite il chat template del modello e campionamento (`temperature=0.9`, `top_k=50`).

**Scelta di design importante**: non si parafrasa l'intera frase "a photo of a [razza], [didascalia]". Si parafrasa solo la didascalia, e il nome della razza — noto con certezza dal dataset — viene riattaccato dopo, nella costruzione del prompt finale. Tenendo il nome della razza completamente fuori dal testo che il modello riceve in input, il rischio di corruzione dell'etichetta non viene ridotto: viene eliminato per costruzione.

### 9.3 Generazione di immagini sintetiche — classe `ImageGenerator`

`image_gen = ImageGenerator(cfg.IMAGE_GEN_MODEL); image_gen.load_model(); synth_records = image_gen.run_image_generation(synth_prompts, cfg.SYNTH_DIR); image_gen.clear_gpu()`. Per ogni variante testuale, il prompt finale (`"a photo of a {razza}, {variante}"`) viene passato a una pipeline di diffusione (`AutoPipelineForText2Image` di `diffusers`) per generare un'immagine a 512×512, poi ridimensionata a 224×224.

`ImageGenerator.load_model()` carica un unico modello, `IMAGE_GEN_MODEL` (`sdxl-turbo`) — nessun modello di riserva: è pubblico e non richiede token, quindi non c'è un caricamento che possa fallire per motivi di licenza. `generate()` usa sempre gli stessi parametri, pensati per un modello "turbo":

| Parametro | Valore | Perché |
|---|---|---|
| Step di diffusione | `cfg.SYNTH_STEPS` (2) | Modello "turbo", distillato per generare in pochi step |
| Guidance scale | 0.0 (nessun classifier-free guidance) | I modelli turbo sono addestrati per funzionare senza guidance: la fedeltà al prompt è già "incorporata" nella distillazione |

`sdxl-turbo` è stato preferito a Stable Diffusion XL "pieno" per un motivo pratico: SDXL Base richiede tipicamente 25-40 step per immagine, mentre generare centinaia di immagini sintetiche in queste condizioni richiederebbe ore su un T4 di Colab, non minuti.

Ogni immagine sintetica viene salvata su disco in `data/synthetic/<label>_<razza>/`, insieme a un file `synthetic_metadata.csv` con path, classe, caption originale, variante testuale e prompt completo.

## 10. Valutazione della qualità dei sintetici

La traccia chiede esplicitamente di "valutare la qualità dei dati prodotti". Oltre all'ispezione visiva (una griglia che affianca immagini reali e sintetiche per alcune razze, più una griglia dedicata alle sintetiche con CLIP score più basso), il notebook calcola il **CLIP score** tramite la classe `ClipScorer` (`clip_scorer = ClipScorer(cfg.QUALITY_CHECK_MODEL); clip_scorer.load_model(); ...; clip_scorer.clear_gpu()`): la similarità coseno, moltiplicata per 100, tra l'embedding dell'immagine sintetica e l'embedding del prompt che l'ha generata, usando `openai/clip-vit-base-patch32`.

Il CLIP score è usato **solo in modo descrittivo**, non come filtro automatico che scarta immagini sotto una soglia: è una scelta deliberata, discussa in `SCELTE_PROGETTUALI_ESAME.md`.

## 11. Esperimento B — Augmented

Il training set aumentato è l'unione (`ConcatDataset`) del training reale (invariato) e di un `SyntheticImageDataset` che carica le immagini sintetiche generate al passo precedente, con la stessa trasformazione di training del Baseline. Stessa architettura (`build_model()`), stessa `Config` di training (learning rate, weight decay, dropout, label smoothing, patience, min_delta), stesso validation set reale: l'unica variabile che cambia rispetto al Baseline è la composizione del training set.

## 12. Metriche di valutazione: quelle richieste e quelle aggiunte

La traccia chiede di misurare "accuracy, precision, recall e altre metriche". Il notebook usa:

| Metrica | Cosa cattura | Perché è utile qui |
|---|---|---|
| **Accuracy** | Quota di predizioni corrette sul totale | Richiesta esplicitamente dalla traccia; semplice da interpretare |
| **Precision / Recall / F1 (macro)** | Media semplice sulle 37 classi, senza pesare per frequenza | Richiesta dalla traccia; con 37 classi il macro-F1 è più informativo dell'accuracy da solo |
| **Precision / Recall / F1 (weighted)** | Come sopra ma pesata per il numero di esempi per classe nel test | Confrontata con la versione macro, aiuta a capire se un eventuale miglioramento è uniforme o concentrato sulle classi più rappresentate |
| **Top-3 accuracy** | Il modello include la classe corretta tra le sue 3 predizioni più probabili | Aggiunta. Con 37 classi fine-grained è una metrica più tollerante e realistica |
| **Matrice di confusione (normalizzata)** | Dove esattamente si concentrano gli errori | Richiesta implicitamente; permette confronti qualitativi tra Baseline e Augmented |
| **Delta di recall per classe** | Quali classi migliorano/peggiorano con l'augmentation | Aggiunta. Mostra se il beneficio della Data Augmentation è diffuso o concentrato su poche razze |
| **CLIP score sui sintetici** | Coerenza semantica immagine-testo dei dati generati | Aggiunta, per "valutare la qualità dei dati prodotti" come richiesto dalla traccia |
| **Intervallo di confidenza bootstrap (95%) sulla differenza di accuracy** | Se la differenza osservata è solida o compatibile col rumore campionario | Aggiunta. Un confronto tra due numeri singoli, senza incertezza, può portare a conclusioni premature |

## 13. Error Analysis

Confronta le predizioni di Baseline e Augmented **sulle stesse immagini del test set**, indice per indice (le due chiamate a `evaluate(test_loader)` restituiscono etichette nello stesso ordine). Individua quattro categorie: corretti dall'Augmented (Baseline sbagliava), rotti dall'Augmented (Baseline indovinava), sbagliati da entrambi, e il bilancio netto tra i primi due. Una funzione (`show_error_examples`) mostra alcuni esempi visivi per ciascuna categoria, con l'etichetta vera e le predizioni di entrambi i modelli nel titolo.

## 14. Come leggere i risultati e completare i commenti

Le celle di commento dopo ogni grafico contengono segnaposto tra parentesi quadre (es. `[Da completare dopo l'esecuzione]`), oppure sono già state completate a mano con i numeri osservati in un'esecuzione reale su Colab.

Il procedimento consigliato per le parti ancora segnaposto:

1. Eseguire il notebook su Colab per intero, dall'alto verso il basso.
2. Per ogni cella di commento con un segnaposto, guardare il grafico/la tabella appena prodotta e sostituire il testo tra parentesi quadre con l'osservazione reale, mantenendo il resto della frase.
3. Verificare che la sezione "Risultati e considerazioni finali" nelle conclusioni sia ancorata ai numeri effettivamente ottenuti.

## 15. Limiti e sviluppi futuri

- **Un solo seed**: sia il training dei classificatori sia la generazione dei dati sintetici derivano da un'unica esecuzione a seed fisso. Ripetere con più seed e riportare media/deviazione standard darebbe una conclusione più solida.
- **Iperparametri di regolarizzazione scelti empiricamente**: dropout, weight decay e label smoothing sono stati verificati con un solo giro di prove, non con una ricerca sistematica.
- **Qualità dei sintetici non filtrata**: le immagini sintetiche entrano tutte nel training, indipendentemente dal loro CLIP score. Un filtro di qualità (soglia minima) è un'estensione naturale non implementata.
- **Scala contenuta per motivi di tempo**: i numeri di default sono pensati per un tempo di esecuzione ragionevole su Colab gratuito; sono tutti parametri esposti in `Config`.
- **Trasferibilità limitata al caso reale**: le conclusioni quantitative valgono per Oxford-IIIT Pet, non per immagini di sorveglianza di una centrale elettrica; è il metodo che si intende trasferibile, non i numeri.

Sviluppi naturali: più seed con intervalli di confidenza anche sul training, un filtro di qualità automatico sui sintetici basato sul CLIP score, un confronto diretto con augmentation "classica" più aggressiva (Mixup/CutMix) sullo stesso training set, uno sblocco progressivo del backbone al posto dei learning rate differenziati, altri backbone pretrained (`timm` rende il confronto immediato), una ricerca sistematica degli iperparametri.

---

*Guida basata sulla struttura del notebook `Data_Augmentation_Sicurezza_Centrali_Elettriche.ipynb`. Da usare insieme a `SCELTE_PROGETTUALI_ESAME.md` per la teoria e le motivazioni.*
