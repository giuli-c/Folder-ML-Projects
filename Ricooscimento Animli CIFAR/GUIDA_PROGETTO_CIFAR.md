# Guida Completa: Classificatore CNN Animali vs Veicoli (CIFAR-10)

> **Progetto**: VisionTech Solutions — Riconoscimento di animali per auto a guida autonoma  
> **Algoritmo principale**: CNN (Convolutional Neural Network) con PyTorch  
> **Dataset**: CIFAR-10 (filtrato a classificazione binaria: animali vs veicoli)

---

## Indice

1. [Obiettivo e Scelta della Metrica](#1-obiettivo-e-scelta-della-metrica)
2. [Librerie Utilizzate](#2-librerie-utilizzate)
3. [Seed e Riproducibilità](#3-seed-e-riproducibilità)
4. [Il Dataset CIFAR-10](#4-il-dataset-cifar-10)
5. [Dataset Personalizzato: BinaryCIFAR10](#5-dataset-personalizzato-binarycifar10)
6. [Suddivisione Stratificata Train/Validation](#6-suddivisione-stratificata-trainvalidation)
7. [DataLoader](#7-dataloader)
8. [Architettura CNN Configurabile](#8-architettura-cnn-configurabile)
9. [Classi di Supporto](#9-classi-di-supporto)
10. [Esperimento 1: Baseline](#10-esperimento-1-baseline)
11. [Esperimento 2: Dropout](#11-esperimento-2-dropout)
12. [Esperimento 3: Batch Normalization](#12-esperimento-3-batch-normalization)
13. [Esperimento 4: Dropout + BatchNorm](#13-esperimento-4-dropout--batchnorm)
14. [Esperimento 5: Data Augmentation](#14-esperimento-5-data-augmentation)
15. [Esperimento 6: Loss Pesata + LR Scheduler](#15-esperimento-6-loss-pesata--lr-scheduler)
16. [Confronto Finale e Selezione del Modello](#16-confronto-finale-e-selezione-del-modello)
17. [Conclusioni](#17-conclusioni)

---

## 1. Obiettivo e Scelta della Metrica

Il progetto risolve un problema di **classificazione binaria su immagini**: dato un frame video catturato da un'auto a guida autonoma, il sistema deve distinguere automaticamente tra **animali** e **veicoli stradali**.

### Perché non usare l'Accuracy come metrica principale?

In questo problema il **costo degli errori non è simmetrico**:

| Tipo di errore | Conseguenza reale |
|----------------|-------------------|
| **Falso Negativo** (animale non rilevato) | L'auto non frena → rischio incidente |
| **Falso Positivo** (veicolo visto come animale) | L'auto rallenta inutilmente → solo fastidio |

Per questo motivo la **metrica di riferimento è il Recall della classe 'animale'**:

```
Recall = TP / (TP + FN)

dove:
  TP = animali classificati correttamente
  FN = animali classificati come veicoli (falsi negativi - i più pericolosi)
```

Vogliamo **Recall → 1.0** e **Falsi Negativi → 0**, anche accettando qualche falso positivo in cambio.

---

## 2. Librerie Utilizzate

```python
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns, numpy as np
import gc
```

| Libreria | Funzione | Perché |
|---------|---------|--------|
| `torch` + `torch.nn` | Framework Deep Learning | Ecosistema principale per CNN in Python |
| `torchvision` | Dataset e trasformazioni immagini | Accesso diretto a CIFAR-10 con `download=True` |
| `albumentations` | Data augmentation avanzata | Più flessibile di torchvision.transforms; trasformazioni fotograficamente realistiche |
| `ToTensorV2` | Conversione albumentations → tensore | Compatibilità tra albumentations (NumPy) e PyTorch (tensore) |
| `train_test_split` | Split stratificato | Garantisce la stessa proporzione di classi in train e val |
| `classification_report` | Report metriche per classe | Precision, recall, F1 per ogni classe |
| `confusion_matrix` | Matrice di confusione | Analisi dettagliata degli errori TP/TN/FP/FN |
| `gc` | Garbage collector | Libera la RAM GPU dopo ogni esperimento su Colab |

---

## 3. Seed e Riproducibilità

```python
SEED = 42

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Perché impostare seed su tutti questi generatori?

PyTorch usa **più sorgenti di casualità indipendenti**:

| Sorgente | Cosa influenza |
|---------|---------------|
| `random.seed` | Operazioni di sampling Python standard |
| `PYTHONHASHSEED` | Hashing di dict e set (ordine di iterazione) |
| `np.random.seed` | Operazioni NumPy (es. generazione indici in train_test_split) |
| `torch.manual_seed` | Inizializzazione pesi, operazioni casuali su CPU |
| `torch.cuda.manual_seed_all` | Operazioni casuali su GPU (tutte le GPU disponibili) |
| `cudnn.deterministic=True` | Forza CUDA a usare algoritmi deterministici |
| `cudnn.benchmark=False` | Disabilita la selezione automatica dell'algoritmo (non deterministica) |

**Trade-off**: `deterministic=True` + `benchmark=False` rendono il training leggermente più lento ma garantiscono che eseguendo il notebook due volte si ottengano esattamente gli stessi risultati.

### Selezione del device

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Le CNN richiedono molte operazioni matriciali in parallelo. La **GPU** le esegue su migliaia di core contemporaneamente, accelerando il training di **10-100x** rispetto alla CPU. Su Google Colab è disponibile una GPU NVIDIA T4 o simile gratuitamente.

---

## 4. Il Dataset CIFAR-10

**CIFAR-10** è un dataset standard di Computer Vision contenente:
- 60.000 immagini RGB di dimensione **32×32 pixel**
- 10 classi bilanciate (6.000 immagini per classe)
- 50.000 immagini di training + 10.000 di test

```python
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_base)
testset_full  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_base)
```

`download=True` scarica automaticamente il dataset alla prima esecuzione; nelle esecuzioni successive lo carica dalla cache locale (`./data`). Funziona sia in locale che su Colab.

### Filtraggio delle classi

| ID CIFAR-10 | Classe | Usata nel progetto |
|-------------|--------|--------------------|
| 0 | airplane | ✗ (non rilevante per strade) |
| 1 | automobile | ✓ → Veicolo (label=1) |
| 2 | bird | ✓ → Animale (label=0) |
| 3 | cat | ✓ → Animale (label=0) |
| 4 | deer | ✓ → Animale (label=0) |
| 5 | dog | ✓ → Animale (label=0) |
| 6 | frog | ✓ → Animale (label=0) |
| 7 | horse | ✓ → Animale (label=0) |
| 8 | ship | ✗ (non rilevante per strade) |
| 9 | truck | ✓ → Veicolo (label=1) |

**Perché scartare airplane e ship?**  
Il contesto applicativo è la sicurezza stradale urbana. Aerei e navi non si incontrano su strade; includerli introdurrebbe rumore e allontanerebbe il modello dal problema reale.

### Trasformazione base

```python
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

**`ToTensor()`**: converte l'immagine PIL con valori [0, 255] in tensore PyTorch con valori [0.0, 1.0].

**`Normalize((0.5,), (0.5,))`**: porta i valori in [-1, 1] con la formula:
```
pixel_normalizzato = (pixel - 0.5) / 0.5
```

**Perché normalizzare in [-1, 1]?**  
I pesi iniziali delle reti neurali (inizializzati con Kaiming o Xavier) sono centrati intorno allo zero. Avere input nello stesso range [-1, 1] accelera la convergenza e migliora la stabilità numerica durante la backpropagation.

---

## 5. Dataset Personalizzato: BinaryCIFAR10

```python
class BinaryCIFAR10(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset):
        self.data, self.targets = [], []
        for img, label in cifar_dataset:
            if label in animal_classes + vehicle_road_classes:
                self.data.append(img)
                self.targets.append(binary_label(label))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)
```

### Perché una classe Dataset personalizzata?

PyTorch richiede che ogni dataset implementi due metodi fondamentali:
- **`__getitem__(index)`**: restituisce il campione all'indice specificato
- **`__len__()`**: restituisce la dimensione totale del dataset

Ereditare da `torch.utils.data.Dataset` garantisce compatibilità automatica con:
- `DataLoader` (batching, shuffle, multi-process loading)
- `Subset` (selezione di sottoinsiemi per indici)
- `random_split` e `train_test_split`

### `binary_label()`

```python
def binary_label(label):
    return 0 if label in animal_classes else 1
```

Rimappa le etichette originali CIFAR-10 (0-9) in etichette binarie:
- 0 = animale
- 1 = veicolo

---

## 6. Suddivisione Stratificata Train/Validation

```python
train_idx, val_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=SEED
)
train_dataset = Subset(binary_train_dataset, train_idx)
val_dataset   = Subset(binary_train_dataset, val_idx)
```

### Perché `train_test_split` con `stratify` invece di `random_split`?

| Metodo | Comportamento | Quando usarlo |
|--------|---------------|---------------|
| `random_split` | Suddivisione casuale senza garantire proporzione classi | Dataset perfettamente bilanciati |
| `train_test_split(stratify=...)` | Garantisce stessa proporzione di classi in train e val | Dataset con classi sbilanciate (**questo caso**) |

Nel dataset binario ci sono **6 classi animali** vs **2 classi veicoli** → 3:1 di sbilanciamento. Senza stratificazione, per pura casualità il validation set potrebbe avere una proporzione diversa, rendendo le metriche di validazione poco rappresentative.

### `Subset`

`Subset(dataset, indices)` crea una vista del dataset originale selezionando solo i campioni agli indici specificati, **senza copiare i dati in memoria**. Questo è fondamentale per efficienza con dataset grandi.

---

## 7. DataLoader

```python
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
valloader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)
testloader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)
```

### Parametri e motivazioni

| Parametro | Training | Validation/Test | Motivazione |
|-----------|----------|-----------------|-------------|
| `batch_size=64` | 64 | 64 | Compromesso: abbastanza grande per sfruttare il parallelismo GPU (batch grandi → gradiente stabile), abbastanza piccolo da stare in VRAM |
| `shuffle=True` | ✓ | ✗ | Mescola i campioni ad ogni epoca → il modello non impara l'ordine dei dati; disabilitato in val/test per confronti riproducibili |
| `num_workers=2` | 2 | 2 | Carica i batch in parallelo su thread separati mentre la GPU calcola → elimina il bottleneck di I/O |

### Trade-off batch_size

| Batch piccolo (16-32) | Batch grande (64-128) |
|----------------------|----------------------|
| Gradiente più rumoroso (può aiutare a uscire da minimi locali) | Gradiente più stabile (media su più campioni) |
| Più aggiornamenti per epoca | Meno aggiornamenti ma più veloci |
| Meno RAM GPU | Più RAM GPU |

---

## 8. Architettura CNN Configurabile

### Perché una CNN e non una rete Fully Connected?

Una rete FC standard tratterebbe ogni pixel come feature indipendente:
- Immagine 32×32×3 = **3.072 feature in input**
- Non considera le relazioni spaziali tra pixel vicini (un bordo è un pattern locale, non dipende da pixel lontani)

La **CNN** usa **filtri convoluzionali** che scorrono sull'immagine, imparando a rilevare pattern locali (bordi, texture, forme). Questo la rende:
- Molto più efficiente in parametri
- Capace di catturare feature gerarchiche (bordi → forme → oggetti)
- **Invariante alla traslazione**: rileva lo stesso pattern ovunque nell'immagine

### I layer della CNN

```python
class ExperimentCNN(nn.Module):
    def __init__(self, conv_layers_config, fc_layers_config, ...):
```

**`nn.Conv2d(in_channels, out_channels, kernel_size)`**  
Applica `out_channels` filtri di dimensione `kernel_size × kernel_size` all'immagine in input.  
- Ogni filtro apprende a rilevare un pattern diverso (bordi verticali, orizzontali, texture...)
- Parametri: `(in_channels × kernel_size² + 1) × out_channels`

Esempio con Conv2d(3, 32, 3):
```
Parametri = (3 × 3² + 1) × 32 = 896
Output shape: [batch, 32, 30, 30]  (32−3+1=30 per ogni dimensione spaziale)
```

**`nn.BatchNorm2d(num_features)`** (opzionale)  
Normalizza le attivazioni per canale (media≈0, std≈1) all'interno di ogni batch.  
Vantaggi:
- Stabilizza il training (riduce gradient vanishing/exploding)
- Permette learning rate più alti
- Leggero effetto regolarizzante

**`F.relu(x)`** — Rectified Linear Unit  
Funzione di attivazione non lineare: `relu(x) = max(0, x)`.  
Introduce la non-linearità necessaria perché la rete possa apprendere funzioni complesse.  
Vantaggi rispetto a sigmoid/tanh: non soffre di gradient vanishing per valori positivi.

**`nn.MaxPool2d(kernel_size=2, stride=2)`**  
Riduce la dimensione spaziale di fattore 2 prendendo il valore massimo in ogni finestra 2×2.  
Vantaggi:
- Riduce il numero di parametri nei layer successivi
- Introduce invarianza alle piccole traslazioni
- Riduce il rischio di overfitting

**`torch.flatten(x, 1)`**  
Appiattisce tutte le dimensioni tranne il batch: `[batch, canali, H, W]` → `[batch, canali×H×W]`.  
Necessario per passare dall'output dei conv layer ai layer FC.

**`nn.Linear(in_features, out_features)`** — Fully Connected  
Ogni neurone è connesso a tutti i neuroni del layer precedente.  
Nei layer FC la rete combina le feature estratte dai conv layer per prendere la decisione finale di classificazione.

**`nn.Dropout(p=0.5)`** (opzionale)  
Durante il training, azzera casualmente il p% dei neuroni ad ogni forward pass.  
Motivazione: forza la rete a non dipendere da singoli neuroni (co-adattamento), distribuendo l'apprendimento → migliore generalizzazione.  
**Importante**: disattivato automaticamente in `model.eval()`.

### `nn.ModuleList` vs lista Python

```python
self.convs = nn.ModuleList()  # ✓ corretto
self.convs = []               # ✗ sbagliato
```

`nn.ModuleList` registra automaticamente i layer come parametri del modello. Una lista Python normale non viene tracciata da PyTorch, quindi i parametri non verrebbero inclusi nell'ottimizzazione e i layer non verrebbero spostati su GPU con `.to(device)`.

### Output shape — calcolo manuale

Per l'architettura standard [(32,3), (64,3)] su input [3,32,32]:

```
Input:               [3, 32, 32]
Conv2d(3→32, k=3):   [32, 30, 30]   (32 − 3 + 1 = 30)
MaxPool2d(2):        [32, 15, 15]   (30 // 2 = 15)
Conv2d(32→64, k=3):  [64, 13, 13]   (15 − 3 + 1 = 13)
MaxPool2d(2):        [64,  6,  6]   (13 // 2 = 6)
Flatten:             [2304]          (64 × 6 × 6 = 2304)
Linear(2304→128):    [128]
Dropout(0.5):        [128]
Linear(128→2):       [2]  ← logit per Animale e Veicolo
```

---

## 9. Classi di Supporto

### `Experiment` — Contenitore di configurazione

```python
class Experiment:
    def __init__(self, name, transform, conv_config, fc_config,
                 use_dropout, use_batchnorm, use_adam, dataloaders, use_weighted_loss):
```

Centralizza tutti i parametri di un esperimento in un unico oggetto. Questo permette di:
- Mantenere una lista di esperimenti e confrontarli facilmente
- Riusare la configurazione del miglior esperimento in quelli successivi (`best_model.conv_config`, ecc.)
- Evitare errori da parametri duplicati o dimenticati

### `EarlyStopping` — Stop anticipato del training

```python
class EarlyStopping:
    def __init__(self, save_path, patience=5, min_delta=0.0):
```

**Funzionamento**: monitora la validation loss ad ogni epoca. Se non migliora per `patience` epoche consecutive, interrompe il training.

**Perché è fondamentale?**  
Senza early stopping il modello continua ad allenarsi fino all'ultima epoca, ma dopo un certo punto fa **overfitting**: memorizza il training set invece di generalizzare. L'early stopping ferma il training nel momento di miglior generalizzazione e salva quel modello.

```python
def __call__(self, val_loss, model):
    if migliora:
        self._save(model)   # salva il checkpoint migliore
        self.counter = 0
    else:
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
```

**`patience=5`**: accetta fino a 5 epoche senza miglioramento. Troppo basso → stop prematuro; troppo alto → non serve a nulla.

**Perché salvare `state_dict()` e non il modello intero?**  
```python
torch.save(model.state_dict(), path)  # ✓ salva solo i pesi
torch.save(model, path)               # ✗ dipende dalla struttura della classe
```
`state_dict()` salva solo i pesi (tensori numerici), indipendente dal codice sorgente. È il metodo raccomandato da PyTorch per la portabilità.

### `ModelTrain` — Ciclo di training

```python
class ModelTrain:
    def train_epoch(self):   # una epoca di training
    def test_epoch(self):    # una epoca di validation
```

**`train_epoch()`** — i 4 passi fondamentali della backpropagation:
```python
self.optimizer.zero_grad()   # 1. azzera i gradienti accumulati
outputs = self.model(inputs) # 2. forward pass
loss = self.criterion(...)   # 3. calcolo della loss
loss.backward()              # 4. backpropagation (calcola gradiente)
self.optimizer.step()        # 5. aggiorna i pesi
```

**`optimizer.zero_grad()`**: PyTorch **accumula** i gradienti per default (utile per gradienti di più batch). Bisogna azzerarli manualmente prima di ogni batch, altrimenti i gradienti del batch precedente si sommerebbero a quelli del batch corrente.

**`test_epoch()`** con `torch.no_grad()`:  
Disabilita il calcolo del gradiente durante la validation. Risparmia circa il 30-50% di memoria e velocizza l'esecuzione, poiché non abbiamo bisogno di fare backpropagation.

**`model.train()` vs `model.eval()`**:

| Modalità | Dropout | BatchNorm |
|----------|---------|-----------|
| `model.train()` | Attivo (azzera neuroni casualmente) | Usa statistiche del batch corrente |
| `model.eval()` | Disattivato (tutti i neuroni attivi) | Usa statistiche globali accumulate nel training |

### `ModelEvaluator` — Valutazione e visualizzazione

```python
class ModelEvaluator:
    def evaluate(self, model, dataloader, device)   # → labels, preds, images
    def show_results(self, labels, preds)            # classification_report + confusion matrix
    def show_animal_errors(self, images, preds, labels)  # visualizza falsi negativi
```

**`@torch.no_grad()`** come decoratore: equivalente a `with torch.no_grad():` ma più pulito sintatticamente. Disabilita il calcolo del gradiente per tutta la funzione.

**`torch.max(outputs, dim=1)`**: restituisce `(valori_max, indici)`. Gli indici sono le classi predette (0=animale, 1=veicolo). Equivale ad applicare `argmax` lungo la dimensione delle classi.

### `ModelCreator` — Coordinatore dell'esperimento

```python
class ModelCreator:
    def __init__(self, exp, trainloader, valloader, testloader, epochs=50):
```

Crea e configura:
1. Il modello (`ExperimentCNN`)
2. La loss function (`CrossEntropyLoss` o pesata)
3. L'ottimizzatore (`SGD` o `Adam`)
4. Lo scheduler del learning rate (opzionale)
5. L'early stopper

---

## 10. Esperimento 1: Baseline

```python
exp1 = Experiment(
    name="base",
    conv_config=[(32, 3), (64, 3)],
    fc_config=[128],
    use_dropout=False, use_batchnorm=False, use_adam=False
)
```

**Scopo**: stabilire un punto di riferimento (baseline) con la CNN più semplice possibile.

**Ottimizzatore: SGD (Stochastic Gradient Descent)**
```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
- `lr=0.01`: learning rate standard per SGD con CNN su CIFAR
- `momentum=0.9`: accumula il gradiente delle epoche precedenti per accelerare la convergenza nelle direzioni persistenti e smorzare le oscillazioni

**Loss: CrossEntropyLoss standard**  
Combina internamente `LogSoftmax` + `NLLLoss`. Per classificazione multiclasse/binaria è la scelta standard quando le classi sono bilanciate.

**Risultato atteso**: overfitting dopo 4-5 epoche (training loss scende, val loss risale).

---

## 11. Esperimento 2: Dropout

```python
exp2 = Experiment(..., use_dropout=True)
# → ExperimentCNN(..., dropout=0.5)
# → nn.Dropout(0.5) dopo ogni FC layer nascosto
```

**Dropout**: durante il training azzera casualmente il 50% dei neuroni ad ogni forward pass.

**Meccanismo:**
```
Input FC: [x₁, x₂, x₃, x₄, x₅, x₆]
Maschera: [ 1,  0,  1,  0,  1,  1 ]  (casuale ogni forward pass)
Output:   [x₁,  0, x₃,  0, x₅, x₆]
```

**Perché aiuta?** I neuroni non possono "contare" sui loro vicini → devono imparare feature indipendenti e robuste. Questo riduce il co-adattamento e migliora la generalizzazione.

**Nota importante**: `p=0.5` è il valore standard proposto nell'articolo originale di Hinton (2014). Per dropout sui layer convoluzionali si usano valori più bassi (0.1-0.2).

---

## 12. Esperimento 3: Batch Normalization

```python
exp3 = Experiment(..., use_batchnorm=True)
# → nn.BatchNorm2d(out_channels) dopo ogni Conv2d
```

**BatchNorm2d**: normalizza le attivazioni per canale all'interno di ogni batch:
```
BN(x) = γ × (x - μ_batch) / √(σ²_batch + ε) + β

dove:
  μ_batch = media del batch per canale
  σ²_batch = varianza del batch per canale
  γ, β = parametri apprendibili (scala e shift)
  ε = piccola costante per stabilità numerica
```

**Vantaggi:**
1. **Stabilità**: riduce il problema dell'Internal Covariate Shift (le distribuzioni interne cambiano durante il training)
2. **Learning rate più alto**: la normalizzazione rende il training meno sensibile al lr
3. **Regolarizzazione implicita**: l'aggiunta di rumore (statistiche del batch) ha un effetto simile al dropout
4. **Convergenza più veloce**: tipicamente converge in meno epoche

**Posizione**: BN viene applicata DOPO la convoluzione e PRIMA della ReLU.

---

## 13. Esperimento 4: Dropout + BatchNorm

```python
exp4 = Experiment(..., use_dropout=True, use_batchnorm=True)
```

**Combinazione complementare:**
- **BatchNorm** agisce sui **layer convoluzionali**: stabilizza le feature maps
- **Dropout** agisce sui **layer fully connected**: riduce il co-adattamento tra neuroni

Le due tecniche si complementano perché agiscono su parti diverse della rete.

**Risultato atteso**: miglior bilanciamento tra regolarizzazione e capacità del modello. Tipicamente questa combinazione supera ciascuna tecnica presa singolarmente.

---

## 14. Esperimento 5: Data Augmentation

```python
augment_pipeline = A.Compose([
    A.Resize(32, 32),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(blur_limit=(3,3), p=0.1),
    A.Affine(translate_percent=0.05, scale=(0.95,1.05), rotate=(-15,15), p=0.3),
    A.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ToTensorV2()
])
```

### Perché albumentations invece di torchvision.transforms?

| Caratteristica | torchvision.transforms | albumentations |
|---------------|----------------------|----------------|
| Performance | Standard | ~2-3x più veloce |
| Varietà trasformazioni | Base | Molto ricca |
| Realismo | Limitato | Trasformazioni fotografiche realistiche |
| Compatibilità | PIL/Tensore | NumPy → richiede wrapper |

### Le trasformazioni scelte e il loro significato fisico

| Trasformazione | Probabilità | Cosa simula |
|---------------|------------|-------------|
| `HorizontalFlip` | 50% | Animale/veicolo che si muove da destra o sinistra |
| `RandomBrightnessContrast` | 20% | Variazioni di illuminazione (sole, nuvole, ombra) |
| `GaussianBlur` | 10% | Sfocatura da movimento o messa a fuoco imperfetta |
| `Affine` (shift+scale+rotate) | 30% | Prospettiva diversa, posizione variabile nell'inquadratura |

### Perché augmentare SOLO il training set?

L'augmentation serve a **aumentare la variabilità del training set** per migliorare la generalizzazione. Il test set deve rappresentare le condizioni reali di deployment: applicare augmentation al test set altererebbe le immagini reali e renderebbe le metriche non realistiche.

### Wrapper `Transforms`

```python
class Transforms:
    def __call__(self, img):
        return self.aug(image=np.array(img))['image']
```

Albumentations riceve array NumPy e restituisce un dizionario. Il wrapper converte automaticamente da PIL (torchvision) a NumPy e viceversa.

---

## 15. Esperimento 6: Loss Pesata + LR Scheduler

### CrossEntropyLoss pesata

```python
weight = torch.tensor([2.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
```

**Motivazione**: penalizza il doppio un errore sulla classe 0 (animale) rispetto alla classe 1 (veicolo). Questo dovrebbe spingere il modello a essere più "conservativo" → meno falsi negativi sugli animali.

**Formula**:
```
Loss_pesata = -Σ weight[c] × y[c] × log(p[c])

dove weight[0]=2.0 (animale) e weight[1]=1.0 (veicolo)
```

### StepLR Scheduler

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

Riduce il learning rate di un fattore `gamma=0.5` ogni `step_size=10` epoche:
```
Epoch 1-10:  lr = 0.001
Epoch 11-20: lr = 0.001 × 0.5 = 0.0005
Epoch 21-30: lr = 0.0005 × 0.5 = 0.00025
...
```

**Motivazione**: con lr alto il modello converge velocemente ma può oscillare attorno al minimo senza raggiungerlo. Ridurre progressivamente il lr permette un affinamento più preciso dei pesi nelle epoche finali.

---

## 16. Confronto Finale e Selezione del Modello

```python
# Metrica di selezione: prima il recall, poi i falsi negativi come tiebreaker
best_model = max(experiments, key=lambda x: (x.recall_animale, -x.false_negatives_animale))
```

**Perché questa funzione di ordinamento?**  
Un recall di 1.0 con 0 falsi negativi è la situazione ideale. La tupla `(recall, -FN)` ordina prima per recall (più è alto meglio è) e, a parità di recall, per falsi negativi (meno è meglio è, da cui il segno `-`).

### Tabella riepilogativa dei risultati

| Esperimento | Tecnica principale | Recall Animali | FN |
|-------------|-------------------|----------------|-----|
| base | Nessuna regolarizzazione | ~0.80 | ~1199 |
| Dropout | Dropout(0.5) | ~0.00 | ~6000 |
| Batch_norm | BatchNorm2d | ~0.46 | ~3226 |
| Drop_Norm | Dropout + BatchNorm | ~0.99 | ~46 |
| Augmentation | Drop_Norm + albumentations | ~1.00 | ~0 |
| Custom | Augmentation + loss pesata | variabile | variabile |

**Osservazioni chiave:**
- Il **Dropout da solo** ha peggiorato drasticamente le performance: con p=0.5 ha ridotto troppo la capacità del modello
- La **BatchNorm da sola** ha migliorato ma non abbastanza
- **Dropout + BatchNorm** si sono complementati bene: recall quasi perfetto
- La **Data Augmentation** ha portato il modello al massimo: recall=1.0, FN=0
- La **Loss pesata** non ha aiutato: l'alterazione del gradiente ha destabilizzato il training

---

## 17. Conclusioni

### Architettura del flusso di lavoro

```
CIFAR-10 (60k immagini, 10 classi)
    ↓  filtraggio BinaryCIFAR10
Binario (40k animali + veicoli)
    ↓  train_test_split(stratify, 80/20)
Train (32k) + Val (8k)     Test (8k) [non toccato durante training]
    ↓  DataLoader(batch=64, shuffle=True)
    ↓  ExperimentCNN (Conv → BN → ReLU → Pool → FC → Dropout → Output)
    ↓  CrossEntropyLoss + SGD/Adam
    ↓  EarlyStopping (patience=5)
    ↓  ModelEvaluator → Recall, Precision, FN
    ↓  Confronto tra 6 esperimenti
Modello migliore → Deployment
```

### Lezioni chiave del progetto

1. **La scelta della metrica è cruciale**: ottimizzare l'accuracy avrebbe dato risultati molto diversi da ottimizzare il recall. La metrica deve riflettere il costo reale degli errori nell'applicazione target.

2. **L'augmentation è il fattore più impattante**: per immagini 32×32 di bassa risoluzione, aumentare la variabilità del training set ha avuto più impatto di qualsiasi tecnica di regolarizzazione.

3. **Dropout e BatchNorm si complementano**: agiscono su parti diverse della rete (FC vs Conv) e la loro combinazione supera ciascuna tecnica singolarmente.

4. **La loss pesata non è sempre la soluzione**: alterare i pesi della loss può destabilizzare il training, specialmente se il modello ha già convergito a buone performance con altri metodi.

5. **Early stopping è indispensabile**: senza di esso, il modello farebbe overfitting perdendo la capacità di generalizzare su dati nuovi.

6. **Il test set è sacro**: non deve mai essere usato durante lo sviluppo. La sua valutazione finale rappresenta la stima più onesta delle performance reali del sistema.

---

*Guida realizzata per il progetto VisionTech Solutions — Classificazione CNN CIFAR-10 (Animali vs Veicoli)*
