# Guida Completa: Classificatore di Fiori con Transfer Learning (EfficientNet-B0)

> **Progetto**: GreenTech Solutions Ltd. — Riconoscimento automatico di fiori per AgriTech  
> **Algoritmo principale**: Transfer Learning con EfficientNet-B0 (timm + PyTorch)  
> **Dataset**: Daisy vs Dandelion (immagini fotografiche reali)

---

## Indice

1. [Obiettivo e Scelta della Metrica](#1-obiettivo-e-scelta-della-metrica)
2. [Librerie Utilizzate](#2-librerie-utilizzate)
3. [Seed e Riproducibilità](#3-seed-e-riproducibilità)
4. [Il Dataset e la struttura delle directory](#4-il-dataset-e-la-struttura-delle-directory)
5. [Classe Config: configurazione centralizzata](#5-classe-config-configurazione-centralizzata)
6. [Transfer Learning: cos'è e perché usarlo](#6-transfer-learning-cosè-e-perché-usarlo)
7. [EfficientNet-B0: il backbone scelto](#7-efficientnet-b0-il-backbone-scelto)
8. [Trasformazioni e Data Augmentation con timm](#8-trasformazioni-e-data-augmentation-con-timm)
9. [DataLoader e ImageFolder](#9-dataloader-e-imagefolder)
10. [Automatic Mixed Precision (AMP)](#10-automatic-mixed-precision-amp)
11. [Loss Functions](#11-loss-functions)
12. [Ottimizzatore: AdamW](#12-ottimizzatore-adamw)
13. [Scheduler del Learning Rate](#13-scheduler-del-learning-rate)
14. [Early Stopping](#14-early-stopping)
15. [Esperimento 1: Baseline](#15-esperimento-1-baseline)
16. [Esperimento 2: Fine-Tuning Progressivo](#16-esperimento-2-fine-tuning-progressivo)
17. [Esperimento 3: Fine-Tuning Completo](#17-esperimento-3-fine-tuning-completo)
18. [Esperimento 4: Augmentation Forte + Label Smoothing](#18-esperimento-4-augmentation-forte--label-smoothing)
19. [Esperimento 5: Mixup](#19-esperimento-5-mixup)
20. [Confronto Finale e Selezione del Modello](#20-confronto-finale-e-selezione-del-modello)
21. [Conclusioni](#21-conclusioni)

---

## 1. Obiettivo e Scelta della Metrica

Il progetto risolve un problema di **classificazione binaria su immagini fotografiche**: dato un'immagine di un fiore, il sistema deve determinare automaticamente se appartiene alla classe **daisy** (margherita) o **dandelion** (tarassaco).

### Perché F1-score macro come metrica principale?

L'**accuracy** misura semplicemente la percentuale di predizioni corrette. In problemi bilanciati è sufficiente, ma non cattura la qualità per ogni singola classe.

L'**F1-score macro** è la media del F1-score calcolato separatamente per ogni classe, senza ponderazione per frequenza:

```
F1_classe = 2 × (Precision × Recall) / (Precision + Recall)

F1_macro = (F1_daisy + F1_dandelion) / 2
```

**Vantaggi del F1 macro:**
- Valuta il modello in modo bilanciato su tutte le classi
- Penalizza i modelli che ignorano una classe
- Non si lascia ingannare dall'accuratezza alta su una sola classe dominante
- Standard accettato per dataset con possibile sbilanciamento

**In questo progetto:** le due classi non sono perfettamente bilanciate, quindi il F1 macro è più rappresentativo dell'accuracy per valutare se il sistema funziona bene su entrambi i tipi di fiori.

---

## 2. Librerie Utilizzate

```python
import torch, torch.nn as nn, torch.optim as optim
from torchvision.datasets import ImageFolder
import timm
from timm.data import create_transform, resolve_data_config
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.metrics import accuracy_score, classification_report, f1_score
```

| Libreria | Funzione | Perché |
|---------|---------|--------|
| `torch` + `torch.nn` | Framework Deep Learning | Ecosistema principale per reti neurali in Python |
| `torchvision.ImageFolder` | Caricamento dataset strutturato per directory | Legge automaticamente le classi dai nomi delle cartelle |
| `timm` (PyTorch Image Models) | Modelli pretrained e utility | Accesso diretto a 600+ architetture pretrained su ImageNet |
| `timm.create_transform` | Pipeline di trasformazioni coerente con il modello | Usa le stesse statistiche di normalizzazione usate nel pretraining |
| `timm.Mixup` | Data augmentation Mixup | Implementazione ottimizzata con supporto a soft labels |
| `LabelSmoothingCrossEntropy` | Loss con label smoothing | Già integrata in timm, compatibile con le trasformazioni |
| `sklearn.metrics` | Metriche di valutazione | Calcolo di accuracy, F1, classification report, confusion matrix |
| `dataclasses.dataclass` | Configurazione strutturata | Genera `__init__` e `__repr__` automaticamente dai campi annotati |

---

## 3. Seed e Riproducibilità

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Perché impostare seed su tutti questi generatori?**

PyTorch usa più sorgenti di casualità indipendenti che devono essere controllate tutte:

| Sorgente | Cosa influenza |
|---------|---------------|
| `random.seed` | Operazioni Python standard |
| `np.random.seed` | Operazioni NumPy (ordine dei batch, augmentation) |
| `torch.manual_seed` | Inizializzazione pesi, Dropout, shuffle interno |
| `torch.cuda.manual_seed_all` | Operazioni casuali su tutte le GPU disponibili |
| `cudnn.deterministic=True` | Forza CUDA a usare algoritmi deterministici |
| `cudnn.benchmark=False` | Disabilita la selezione automatica dell'algoritmo (non deterministica) |

**Trade-off:** `deterministic=True` rallenta leggermente il training (~10-15%) ma garantisce che rieseguire il notebook due volte produca esattamente gli stessi risultati — fondamentale per la validità degli esperimenti.

---

## 4. Il Dataset e la Struttura delle Directory

Il dataset è organizzato in tre split (train, valid, test) con la struttura di directory standard per `ImageFolder`:

```
datasets/progetto-finale-flowes/
├── train/
│   ├── daisy/       ← immagini di training della classe daisy
│   └── dandelion/   ← immagini di training della classe dandelion
├── valid/
│   ├── daisy/
│   └── dandelion/
└── test/
    ├── daisy/
    └── dandelion/
```

**`ImageFolder` di torchvision** legge questa struttura automaticamente:
- Ogni sottocartella diventa una classe
- Le classi vengono ordinate alfabeticamente: `daisy=0`, `dandelion=1`
- L'ordine è garantito essere consistente tra i tre split

### Download del dataset

```python
!wget -O progetto-finale-flowes.tar.gz {DATASET_URL}
!tar -xzf progetto-finale-flowes.tar.gz
```

Il blocco `if not os.path.isdir(DATASET_FOLDER)` evita di scaricare il dataset ad ogni esecuzione, controllando se la directory esiste già.

### Filtraggio dei file non validi

```python
def _is_valid_image_file(self, filename):
    return (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
            and not os.path.basename(filename).startswith('._'))
```

macOS e alcuni sistemi di file S3 creano file di metadati nascosti che iniziano con `._`. Senza questo filtro, `ImageFolder` proverebbe a caricarli come immagini causando errori.

---

## 5. Classe Config: Configurazione Centralizzata

```python
@dataclass
class Config:
    experiment_name: str = "baseline"
    model_name: str = "efficientnet_b0"
    freeze_backbone: bool = True
    ...
```

**`@dataclass`** è un decoratore Python che genera automaticamente `__init__`, `__repr__`, e `__eq__` dai campi annotati. Evita di scrivere manualmente il costruttore.

**Perché centralizzare la configurazione?**

1. **Confronto trasparente:** `progressive_cfg` differisce da `baseline_cfg` per soli 3 parametri → la differenza è esplicita
2. **Evita errori:** i parametri non sono sparsi in decine di celle del notebook
3. **`copy.deepcopy(cfg)`:** permette di clonare una configurazione e modificare un solo parametro senza toccare l'originale

**Parametri chiave:**

| Parametro | Valori possibili | Effetto |
|-----------|-----------------|---------|
| `freeze_backbone` | True/False | Se True: allena solo la testa, backbone congelato |
| `progressive_unfreeze` | True/False | Se True: sblocca backbone a `unfreeze_epoch` |
| `augmentation` | "basic"/"moderate"/"strong" | Intensità delle trasformazioni di training |
| `label_smoothing` | 0.0 — 0.2 | Regolarizzazione via target morbidi |
| `mixup_alpha` | 0.0 — 0.4 | Intensità del Mixup (0=disattivato) |
| `use_cosine_scheduler` | True/False | Tipo di scheduler del learning rate |

---

## 6. Transfer Learning: cos'è e perché usarlo

Il **Transfer Learning** consiste nel riutilizzare un modello già allenato su un task diverso (ma correlato) come punto di partenza per il task specifico.

### Senza Transfer Learning (CNN from scratch)

```
Parametri EfficientNet-B0: ~5.3 milioni
→ Richiedono milioni di immagini per convergere
→ GPU ad alto costo
→ Settimane di training
```

### Con Transfer Learning

```
EfficientNet-B0 pretrained su ImageNet (1.2M immagini, 1000 classi)
→ Il backbone ha già imparato:
  - Layer bassi: bordi, angoli, texture
  - Layer medi: pattern di colore, forme
  - Layer alti: parti di oggetti (petali, foglie)
→ Adattiamo solo la testa finale al nostro task (daisy vs dandelion)
→ Convergenza in poche epoche
→ Performance eccellenti anche con pochi dati
```

### Perché funziona per i fiori?

I fiori sono oggetti visivi simili alle immagini di ImageNet (sono fotografie naturali). Le feature imparate su ImageNet — colori, texture, forme — sono direttamente utili per distinguere margherite da tarassachi.

**Regola pratica:** Transfer Learning è particolarmente efficace quando il dataset di destinazione è:
- Piccolo (< 10.000 immagini)
- Simile al dataset di pre-training (fotografie naturali)

---

## 7. EfficientNet-B0: il Backbone Scelto

**EfficientNet** è una famiglia di reti convoluzionali sviluppata da Google (2019) che ottimizza contemporaneamente tre dimensioni: profondità, larghezza, e risoluzione dell'input.

```python
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
```

**`num_classes=2`**: sostituisce automaticamente la testa finale originale (1000 classi per ImageNet) con un layer `Linear(1280, 2)`.

### Perché EfficientNet-B0?

| Modello | Parametri | Top-1 ImageNet | Velocità |
|---------|-----------|----------------|---------|
| ResNet-18 | 11.7M | 69.8% | Molto veloce |
| EfficientNet-B0 | 5.3M | 77.1% | Veloce |
| EfficientNet-B4 | 19M | 82.9% | Medio |
| ResNet-50 | 25.6M | 76.1% | Medio |

EfficientNet-B0 offre il miglior compromesso tra accuratezza su ImageNet e dimensione del modello. È particolarmente adatto per dataset piccoli su Google Colab (basso costo VRAM, training veloce).

### Congelamento del backbone

```python
if self.config.freeze_backbone:
    for param in self.model.parameters():
        param.requires_grad = False  # congela tutto

    for name, param in self.model.named_parameters():
        if any(key in name for key in ["classifier", "fc", "head"]):
            param.requires_grad = True  # sblocca solo la testa
```

**`requires_grad=False`**: PyTorch non calcolerà il gradiente per questi parametri durante la backpropagation → non vengono aggiornati dall'ottimizzatore.

---

## 8. Trasformazioni e Data Augmentation con timm

### `resolve_data_config`

```python
data_cfg = resolve_data_config({}, model=model_for_cfg)
# → {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
#    "input_size": (3, 224, 224), "interpolation": "bicubic", "crop_pct": 0.875}
```

Recupera automaticamente le statistiche usate durante il pretraining di ImageNet. **È fondamentale usare le stesse statistiche:** il backbone è stato ottimizzato per ricevere input normalizzati con questi valori. Usare normalizzazioni diverse riduce significativamente le performance del transfer learning.

### Livelli di augmentation

| Livello | Trasformazioni | Quando usarlo |
|---------|---------------|---------------|
| `basic` | HFlip(50%) | Baseline: minima distorsione |
| `moderate` | + ColorJitter(0.2) + RandAugment(m=7) + RandomErasing(10%) | Fine-tuning con augmentation moderata |
| `strong` | + VFlip(10%) + ColorJitter(0.3) + RandAugment(m=9) + RandomErasing(20%) | Fine-tuning con augmentation intensa |

**Significato fisico delle trasformazioni:**

| Trasformazione | Simula |
|---------------|--------|
| `HorizontalFlip` | Fiore fotografato da destra o sinistra |
| `VerticalFlip` | Fiore fotografato dall'alto (macrofotografia) |
| `ColorJitter` | Diverse condizioni di illuminazione (sole, nuvole, ombra) |
| `RandAugment` | Policy automatica di distorsioni geometriche e cromatiche |
| `RandomErasing` | Occlusioni parziali del fiore (foglie, altri oggetti in primo piano) |

### Augmentation SOLO sul training set

```python
# Training: trasformazioni con augmentation
train_tfms = create_transform(is_training=True, hflip=0.5, ...)

# Validation e test: solo resize e normalizzazione
eval_tfms = create_transform(is_training=False, ...)
```

L'augmentation serve ad aumentare la variabilità dei dati di training per migliorare la generalizzazione. Il test set deve rappresentare le **condizioni reali di deployment**: applicare augmentation al test set modificherebbe le immagini reali, rendendo le metriche non rappresentative delle performance reali.

---

## 9. DataLoader e ImageFolder

```python
self.train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)
```

| Parametro | Training | Val/Test | Motivazione |
|-----------|----------|----------|-------------|
| `batch_size=32` | 32 | 32 | Compromesso RAM GPU / stabilità del gradiente |
| `shuffle=True` | ✓ | ✗ | Mescola i campioni → il modello non impara l'ordine dei dati |
| `num_workers=2` | 2 | 2 | Carica i batch in parallelo mentre la GPU calcola (elimina I/O bottleneck) |
| `pin_memory=True` | ✓ (se GPU) | ✓ | Alloca i tensori in memoria "pinned" → trasferimento CPU→GPU più veloce |

**`batch_size=32`:** Con immagini 224×224×3 su EfficientNet, 32 campioni occupano ~450MB di VRAM. Su Colab T4 (16GB) questo lascia abbondante margine per i gradienti e i buffer del modello.

---

## 10. Automatic Mixed Precision (AMP)

```python
self.scaler = torch.cuda.amp.GradScaler(enabled=(config.use_amp and device.type == "cuda"))

with torch.cuda.amp.autocast(enabled=...):
    outputs = self.model(images)
    loss = self.criterion(outputs, labels)

self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

**Cos'è AMP?** PyTorch usa normalmente `float32` (32 bit) per tutti i calcoli. AMP usa `float16` (16 bit) dove è sicuro farlo (Conv2d, Linear, BatchNorm) e `float32` dove la precisione è critica (accumulatori di gradiente, operazioni numericamente sensibili).

**Vantaggi:**
- Riduce il consumo di VRAM di circa il 50%
- Accelera il training del 20-50% su GPU moderne
- Permette batch_size più grandi

**`GradScaler`:** Con `float16` i gradienti molto piccoli possono diventare zero (underflow numerico). Lo scaler moltiplica la loss per un fattore grande prima della backpropagation, de-scala i gradienti prima dell'aggiornamento dei pesi, e adatta automaticamente il fattore di scala.

---

## 11. Loss Functions

### CrossEntropyLoss (standard)

```python
nn.CrossEntropyLoss()
```

Per classificazione multiclasse, combina internamente `LogSoftmax + NLLLoss`. Con target one-hot `[0, 1]`:

```
CE(logit, target) = -Σ target[c] × log(softmax(logit)[c])
```

Usata in:
- Training della baseline (exp 1) e fine-tuning semplice (exp 2)
- Sempre per la **valutazione** (validation e test), indipendentemente dalla configurazione di training

### LabelSmoothingCrossEntropy

```python
from timm.loss import LabelSmoothingCrossEntropy
LabelSmoothingCrossEntropy(smoothing=0.1)
```

Invece di target "hard" `[0.0, 1.0]`, usa target "soft":
```
target_smooth[classe_corretta] = 1 - α + α/K
target_smooth[classi_errate]   = α/K

Con α=0.1 e K=2 classi:
target_smooth = [0.05, 0.95]  invece di  [0.0, 1.0]
```

**Motivazione:** un modello troppo sicuro di sé (logit molto alti) diventa rigido e non generalizza. Label smoothing "ammorbidisce" i target, forzando il modello a mantenere un certo grado di incertezza → migliore calibrazione e generalizzazione.

### SoftTargetCrossEntropy (per Mixup)

```python
from timm.loss import SoftTargetCrossEntropy
SoftTargetCrossEntropy()
```

Quando Mixup è attivo, le etichette non sono più interi `{0, 1}` ma vettori continui `[0.7, 0.3]`. La CrossEntropyLoss standard non supporta questo formato. `SoftTargetCrossEntropy` accetta etichette come distribuzioni di probabilità.

---

## 12. Ottimizzatore: AdamW

```python
optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
```

**Adam** adatta il learning rate per ogni singolo parametro basandosi sulla storia dei suoi gradienti (media e varianza dei gradienti passati).

**AdamW vs Adam:** in Adam, `weight_decay` viene applicato in modo non corretto (decoupled dalla normalizzazione del gradiente). AdamW corregge questo problema, applicando il weight decay direttamente ai pesi prima dell'aggiornamento.

```
# Adam: weight_decay accoppiato al gradiente
w_new = w - lr × (gradiente + weight_decay × w) / sqrt(varianza)

# AdamW: weight_decay separato (corretto)
w_new = w - lr × gradiente/sqrt(varianza) - lr × weight_decay × w
```

**Perché AdamW per il fine-tuning?** SGD richiede un tuning attento del learning rate e del momentum. AdamW converge più velocemente con meno tuning degli iperparametri, ed è particolarmente robusto al fine-tuning di modelli pretrained.

**`lr=1e-3` per backbone congelato, `lr=5e-4` per backbone sbloccato:**
Con il backbone sbloccato, i pesi pretrained di ImageNet potrebbero essere distrutti da un lr troppo alto. Il lr dimezzato garantisce aggiornamenti più piccoli e controllati.

---

## 13. Scheduler del Learning Rate

### ReduceLROnPlateau (default)

```python
optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
```

Riduce il learning rate quando la metrica monitorata (F1 macro) non migliora per `patience=2` epoche consecutive:

```
Epoch 1-5:  lr = 1e-3  (F1 migliora)
Epoch 6-7:  lr = 1e-3  (F1 non migliora: counter=1, counter=2)
Epoch 8:    lr = 5e-4  (trigger: counter >= patience → lr × 0.5)
```

**Vantaggio:** adattivo — riduce il lr solo quando serve, indipendentemente dal numero totale di epoche.

### CosineAnnealingLR (alternativa)

```python
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

Riduce il lr seguendo una curva cosinusoidale da `lr_max` a ~0:

```
lr(t) = lr_min + (lr_max - lr_min) × (1 + cos(π × t / T_max)) / 2
```

**Quando usarlo:** quando si conosce il numero esatto di epoche in anticipo e si vuole un decadimento smooth e continuo.

---

## 14. Early Stopping

Il codice non ha una classe EarlyStopping separata ma la logica è integrata nel loop principale di `ExperimentRunner.run()`:

```python
best_val_f1 = -1.0
best_state = None
early_stop_counter = 0

for epoch in range(1, epochs + 1):
    ...
    if val_metrics["f1_macro"] > best_val_f1:
        best_val_f1 = val_metrics["f1_macro"]
        best_state = copy.deepcopy(model.state_dict())  # salva checkpoint
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= config.patience:
        break  # stop anticipato

# Ripristina i pesi del miglior checkpoint
model.load_state_dict(best_state)
```

**Perché monitorare F1 e non la loss?**

La validation loss può continuare a scendere anche quando il modello sta overfittando (memorizzando il training set). Il F1 macro è più direttamente legato alla qualità della classificazione sulle classi di interesse.

**`copy.deepcopy(model.state_dict())`:** salva i pesi del modello in un momento specifico (il miglior checkpoint). Senza `deepcopy`, all'aggiornamento successivo dei pesi il checkpoint verrebbe sovrascritto. Il `state_dict()` contiene solo i tensori numerici (pesi e bias), senza dipendenze dal codice sorgente.

---

## 15. Esperimento 1: Baseline

```python
Config(
    freeze_backbone=True,
    augmentation="basic",
    label_smoothing=0.0,
    lr=1e-3
)
```

**Scopo:** stabilire il punto di partenza più semplice possibile.

**Struttura trainable:**
```
EfficientNet-B0 backbone (5.3M parametri, CONGELATI)
    ↓ feature vector [1280]
Linear(1280, 2)  ← UNICA PARTE TRAINABLE (~2.560 parametri)
    ↓
logit [2]  →  softmax  →  [P(daisy), P(dandelion)]
```

Con soli 2.560 parametri allenabili, il training è estremamente veloce e il rischio di overfitting è minimo. Questo è anche il motivo per cui si usa `lr=1e-3` più alto: la testa è inizializzata casualmente e ha bisogno di aggiornamenti più grandi per convergere rapidamente.

**Risultato atteso:** buone performance grazie al transfer learning, ma con margine di miglioramento tramite fine-tuning del backbone.

---

## 16. Esperimento 2: Fine-Tuning Progressivo

```python
Config(
    freeze_backbone=True,
    progressive_unfreeze=True,
    unfreeze_epoch=4,
    augmentation="moderate"
)
```

**Strategia a 2 fasi:**

```
Epoche 1-3:   backbone CONGELATO   →  allena solo la testa (fast convergence)
Epoca 4+:     backbone SBLOCCATO   →  adatta anche il backbone al dominio fiori
```

**Perché non sbloccare subito tutto?**

Con `lr=1e-3` e backbone sbloccato fin dall'inizio, i gradienti della testa (ancora non convergente) verrebbero propagati in tutto il backbone, potenzialmente "distruggendo" i pesi pretrained di ImageNet in poche iterazioni. Il warm-up iniziale stabilizza la testa, poi il backbone si adatta gradualmente.

**`unfreeze_last_layers()`:** sblocca i parametri dei blocchi finali del backbone (quelli più specifici per il task) mantenendo congelati i layer bassi (che contengono feature più generiche come bordi e texture, già utili così).

**Risultato atteso:** miglior F1 rispetto alla baseline grazie all'adattamento del backbone, con convergenza più stabile rispetto al fine-tuning completo immediato.

---

## 17. Esperimento 3: Fine-Tuning Completo

```python
Config(
    freeze_backbone=False,     # tutto allenabile fin dall'inizio
    augmentation="moderate",
    label_smoothing=0.1,
    lr=5e-4                    # dimezzato per proteggere i pesi pretrained
)
```

**Differenza chiave:** con `freeze_backbone=False` tutti i ~5.3M parametri del backbone sono allenabili dall'inizio.

**`lr=5e-4` (dimezzato):** con il backbone sbloccato, un lr troppo alto potrebbe sovrascrivere rapidamente le feature utili imparate su ImageNet. Il lr dimezzato garantisce aggiornamenti più piccoli e controllati.

**`label_smoothing=0.1`:** aggiunto perché con più parametri allenabili il rischio di overfitting aumenta. Il label smoothing agisce come regolarizzazione aggiuntiva.

**Confronto con exp 2:** se il fine-tuning progressivo ottiene F1 simile o migliore, è preferibile perché più stabile e meno sensibile al lr iniziale.

---

## 18. Esperimento 4: Augmentation Forte + Label Smoothing

```python
Config(
    freeze_backbone=False,
    augmentation="strong",     # vflip + color jitter forte + RandAugment m=9 + erasing 20%
    label_smoothing=0.1,
    lr=5e-4
)
```

**Augmentation "strong" vs "moderate":**

| Parametro | moderate | strong |
|-----------|---------|--------|
| `vflip` | 0.0 | 0.1 |
| `color_jitter` | 0.2 | 0.3 |
| `RandAugment` | m=7 | m=9 |
| `RandomErasing` | 10% | 20% |

**Motivazione:** con un dataset di piccole dimensioni, aumentare artificialmente la variabilità del training set è uno dei metodi più efficaci per ridurre l'overfitting. L'augmentation forte simula condizioni fotografiche molto diverse (illuminazione, prospettiva, occlusioni).

**Risultato atteso:** convergenza rapida (poche epoche) e buona generalizzazione grazie alla forte regolarizzazione combinata.

---

## 19. Esperimento 5: Mixup

```python
Config(
    freeze_backbone=False,
    augmentation="strong",
    label_smoothing=0.1,
    mixup_alpha=0.2           # aggiunge Mixup all'exp 4
)
```

**Mixup** è una tecnica di data augmentation che crea campioni sintetici combinando due immagini e le loro etichette:

```
λ ~ Beta(0.2, 0.2)          (quasi sempre vicino a 0 o 1)

img_mix   = λ·img_a + (1-λ)·img_b
label_mix = λ·label_a + (1-λ)·label_b

Esempio con λ=0.8:
img_mix   = 0.8·[img_daisy] + 0.2·[img_dandelion]  (blend leggero)
label_mix = [0.8, 0.2]                              (soft label)
```

**Perché funziona?** Il modello impara a classificare non solo le immagini "pure" ma anche transizioni graduali tra classi. Questo forza l'apprendimento di feature discriminative più robuste e confini decisionali più smooth.

**`SoftTargetCrossEntropy`:** le etichette soft `[0.8, 0.2]` non sono compatibili con `CrossEntropyLoss` standard (che si aspetta interi `{0, 1}`). `SoftTargetCrossEntropy` calcola:
```
Loss = -Σ label_mix[c] × log(softmax(logit)[c])
```

---

## 20. Confronto Finale e Selezione del Modello

### Regola fondamentale: separazione val set e test set

```python
# FASE DI SVILUPPO: confronto su validation set
summary_df = run_experiments(light_experiments)  # usa val_loader

# FASE DI VALUTAZIONE FINALE: una sola volta sul test set
test_metrics = test_evaluator.full_report(...)   # usa test_loader
```

**Perché non usare il test set per scegliere il modello?**

Se usassimo il test set per confrontare gli esperimenti e scegliere il migliore, faremmo **data leakage**: le nostre scelte di configurazione sarebbero inconsciamente ottimizzate per quei dati specifici. La stima finale delle performance sarebbe troppo ottimistica e non rappresentativa del comportamento reale del sistema su nuovi dati.

Il validation set è la "cartina di tornasole" durante lo sviluppo. Il test set è il "giudice finale" — usato una sola volta alla fine.

### Selezione tramite DataFrame

```python
summary_df = pd.DataFrame(all_results).sort_values(
    by="best_val_f1_during_training",
    ascending=False
)
```

Raccogliere i risultati in un DataFrame pandas permette di:
- Confrontare tutti gli esperimenti side-by-side
- Ordinare automaticamente dal migliore al peggiore
- Esportare i risultati per la documentazione

### Riepilogo risultati

| Esperimento | Tecnica principale | Best Val F1 |
|-------------|-------------------|-------------|
| `baseline_basic` | Backbone congelato | ~0.947 |
| `progressive_moderate` | Fine-tuning progressivo | ~0.975 |
| `full_ft_moderate` | Fine-tuning completo | ~0.970 |
| `strong_aug_label_smoothing` | Augmentation forte | ~0.972 |
| `strong_aug_mixup` | + Mixup | variabile |

---

## 21. Conclusioni

### Flusso completo del progetto

```
Dataset (photos: daisy, dandelion)
    ↓  ImageFolder(train/, valid/, test/)
    ↓  create_transform(is_training=True/False)
    ↓  DataLoader(batch=32, shuffle=True/False)
    ↓
EfficientNet-B0 (pretrained ImageNet)
    ↓  freeze_backbone / progressive_unfreeze / full fine-tuning
    ↓  CrossEntropyLoss / LabelSmoothing / SoftTargetCE
    ↓  AdamW(lr=1e-3 o 5e-4, weight_decay=1e-4)
    ↓  ReduceLROnPlateau(mode='max', factor=0.5)
    ↓  Early Stopping (patience=4, monitora F1 macro)
    ↓
Confronto su validation set → scelta del modello migliore
    ↓
Valutazione finale su test set → stima delle performance reali
```

### Lezioni chiave del progetto

1. **Il transfer learning è straordinariamente efficace per dataset piccoli:** anche la baseline (solo testa allenata) raggiunge ~0.947 di F1 macro. Questo sarebbe impossibile con una CNN allenata da zero.

2. **Il fine-tuning progressivo supera il fine-tuning immediato:** sbloccare il backbone gradualmente preserva meglio le feature di ImageNet e garantisce una convergenza più stabile.

3. **L'augmentation è il fattore più impattante dopo la strategia di fine-tuning:** RandAugment + random erasing + label smoothing migliorano sistematicamente la generalizzazione su dataset fotografici.

4. **La metrica deve rispecchiare l'obiettivo reale:** usare F1 macro invece di accuracy garantisce che il modello performi bene su entrambe le classi, non solo su quella più frequente.

5. **AMP è quasi sempre un vantaggio su GPU moderne:** dimezza la VRAM e accelera il training senza impatto sulla qualità finale.

6. **Il test set è sacro:** usarlo per selezionare il modello introduce data leakage. La valutazione finale deve avvenire una sola volta sul modello già scelto tramite validation.

---

*Guida realizzata per il progetto GreenTech Solutions Ltd. — Riconoscimento di fiori (Daisy vs Dandelion)*
