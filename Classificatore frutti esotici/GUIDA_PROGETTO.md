# Guida Completa: Classificatore di Frutti Esotici con KNN

> **Progetto**: TropicTaste Inc. — Classificazione automatica di frutti esotici tramite Machine Learning  
> **Algoritmo principale**: K-Nearest Neighbors (KNN)  
> **Dataset**: 500 campioni, 5 frutti (Arancia, Banana, Kiwi, Mela, Uva), 5 feature numeriche

---

## Indice

1. [Obiettivo del Progetto](#1-obiettivo-del-progetto)
2. [Librerie Utilizzate](#2-librerie-utilizzate)
3. [Caricamento del Dataset](#3-caricamento-del-dataset)
4. [Esplorazione dei Dati (EDA)](#4-esplorazione-dei-dati-eda)
5. [Analisi degli Outlier](#5-analisi-degli-outlier)
6. [Encoding delle Variabili Categoriche](#6-encoding-delle-variabili-categoriche)
7. [Matrice di Correlazione](#7-matrice-di-correlazione)
8. [Preparazione dei Dati per il Modello](#8-preparazione-dei-dati-per-il-modello)
9. [Test 1-4: Feature Selection](#9-test-1-4-feature-selection)
10. [Test 5: Metriche di Distanza KNN](#10-test-5-metriche-di-distanza-knn)
11. [Test 6: Pesi nella Votazione KNN](#11-test-6-pesi-nella-votazione-knn)
12. [Test 7: Confronto tra Algoritmi](#12-test-7-confronto-tra-algoritmi)
13. [Test 8: Cross-Validation con Diversi K-Fold](#13-test-8-cross-validation-con-diversi-k-fold)
14. [Modello Finale e Valutazione](#14-modello-finale-e-valutazione)
15. [Matrice di Confusione](#15-matrice-di-confusione)
16. [Analisi degli Errori](#16-analisi-degli-errori)
17. [Conclusioni](#17-conclusioni)

---

## 1. Obiettivo del Progetto

Il progetto nasce da un'esigenza aziendale reale: classificare automaticamente i frutti esotici in base a misurazioni fisiche e organolettiche, eliminando la classificazione manuale soggetta a errori umani.

**Problema**: dato un frutto di cui conosciamo peso, diametro, lunghezza, durezza della buccia e dolcezza, vogliamo predire automaticamente a quale specie appartiene.

**Tipo di problema**: classificazione multiclasse (5 classi: Arancia, Banana, Kiwi, Mela, Uva).

### Perché K-Nearest Neighbors (KNN)?

Il KNN è stato scelto come algoritmo principale per diversi motivi:

| Caratteristica | Motivazione |
|---------------|-------------|
| **Non parametrico** | Non assume alcuna distribuzione nei dati: funziona anche con dati che non seguono una distribuzione gaussiana |
| **Intuitivo** | Classifica un campione in base ai suoi K "vicini" più simili nel dataset: lo stesso ragionamento che farebbe un operatore umano esperto |
| **Lazy learner** | Non ha una vera fase di addestramento: il modello si aggiorna automaticamente aggiungendo nuovi campioni senza ri-addestrare da zero |
| **Adatto a feature numeriche** | Le misurazioni fisiche dei frutti sono tutte numeriche continue: perfette per calcolare distanze euclidee |
| **Dataset di dimensioni moderate** | Con 500 campioni il costo computazionale del KNN è trascurabile |

---

## 2. Librerie Utilizzate

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.stats import zscore
```

### Spiegazione delle librerie

| Libreria | Funzione | Perché |
|---------|---------|--------|
| `pandas` | Gestione del DataFrame | Lettura CSV, selezione colonne, statistiche descrittive |
| `numpy` | Operazioni numeriche | Array, calcoli statistici, generazione di range |
| `matplotlib.pyplot` | Grafici base | Curve accuracy/errore al variare di K, grafici a barre |
| `seaborn` | Grafici statistici avanzati | Heatmap della correlazione, boxplot, scatter plot con stile |
| `google.colab.files` | Upload file in Colab | Permette di caricare il CSV dalla macchina locale in Google Colab |
| `StandardScaler` | Normalizzazione feature | Fondamentale per KNN: porta tutte le feature alla stessa scala |
| `LabelEncoder` | Encoding target | Converte nomi di frutti in numeri interi per il modello |
| `train_test_split` | Suddivisione dataset | Divide i dati in training (80%) e test (20%) |
| `GridSearchCV` | Ricerca iperparametri | Trova automaticamente il K ottimale tramite cross-validation |
| `cross_val_score` | Validazione incrociata | Calcola score medio su più fold per stima più affidabile |
| `KNeighborsClassifier` | Modello KNN | Algoritmo principale del progetto |
| `DecisionTreeClassifier` | Albero decisionale | Confronto nel Test 7 |
| `RandomForestClassifier` | Ensemble di alberi | Confronto nel Test 7 |
| `SVC` | Support Vector Machine | Confronto nel Test 7 |
| `GaussianNB` | Naive Bayes | Confronto nel Test 7 |
| `classification_report` | Report metriche | Precision, recall, F1-score per ogni classe |
| `confusion_matrix` | Matrice di confusione | Analisi dettagliata degli errori per classe |
| `zscore` (scipy) | Calcolo Z-score | Identificazione quantitativa degli outlier |

---

## 3. Caricamento del Dataset

```python
# In Google Colab: carica il file dalla macchina locale
upload_file = files.upload()
file_name = next(iter(upload_file))
df = pd.read_csv(file_name)
```

### Perché `files.upload()` e `next(iter(...))`?

- **`files.upload()`**: funzione specifica di Google Colab che apre un dialog di upload. Restituisce un dizionario `{nome_file: contenuto_bytes}`.
- **`next(iter(upload_file))`**: estrae automaticamente il nome del file caricato dal dizionario, senza doverlo hardcodare. Il codice funziona indipendentemente dal nome del file CSV.

### Struttura del dataset

| Colonna | Tipo | Descrizione |
|--------|------|-------------|
| `Frutto` | Categorica | Classe target (Arancia, Banana, Kiwi, Mela, Uva) |
| `Peso (g)` | Numerica | Peso del frutto in grammi |
| `Diametro medio (mm)` | Numerica | Diametro medio in millimetri |
| `Lunghezza media (mm)` | Numerica | Lunghezza media in millimetri |
| `Durezza buccia (1-10)` | Numerica | Scala 1-10 della durezza della buccia |
| `Dolcezza (1-10)` | Numerica | Scala 1-10 della dolcezza |

---

## 4. Esplorazione dei Dati (EDA)

L'**Exploratory Data Analysis** è il passo fondamentale prima di qualsiasi modellazione. Serve a capire i dati, identificare problemi e guidare le scelte successive.

### 4.1 Dimensioni e struttura

```python
df.shape    # (500, 6) → 500 righe, 6 colonne
df.info()   # Tipi di dato, conteggio non-null per colonna
df.head()   # Prime 5 righe per controllo visivo
```

**`df.shape`**: restituisce una tupla `(n_righe, n_colonne)`. Permette di verificare che il dataset abbia le dimensioni attese.

**`df.info()`**: mostra il tipo di dato di ogni colonna (`float64`, `object`, ecc.) e il numero di valori non-null. Evidenzia immediatamente se ci sono valori mancanti (colonne con meno di 500 non-null in un dataset da 500 righe).

**`df.head(n)`**: mostra le prime `n` righe. Utile per un controllo visivo rapido del formato dei dati.

### 4.2 Valori unici nel target

```python
df["Frutto"].unique()
# → ['Mela', 'Banana', 'Arancia', 'Uva', 'Kiwi']
```

Fondamentale per sapere quante e quali classi ha il problema di classificazione.

### 4.3 Controllo qualità dei dati

```python
df.isnull().sum()     # Conta i valori null per colonna
df.duplicated().sum() # Conta le righe duplicate
```

**Valori null**: i valori mancanti richiedono gestione (rimozione, imputation con media/mediana, forward-fill, ecc.). In questo dataset non ce ne sono.

**Duplicati**: righe identiche possono causare **data leakage** se un campione finisce sia nel training set che nel test set, gonfiando artificialmente le metriche di valutazione. In questo dataset non ce ne sono.

### 4.4 Statistiche descrittive

```python
df.describe()
```

Restituisce per ogni colonna numerica: count, mean, std, min, 25° percentile, mediana (50°), 75° percentile, max.

**Come interpretarle**:
- **std alta rispetto alla media**: forte variabilità, possibili outlier
- **Mediana molto diversa dalla media**: distribuzione asimmetrica (skewed)
- **Max molto distante dal 75° percentile**: probabile presenza di outlier superiori

---

## 5. Analisi degli Outlier

### 5.1 Visualizzazione con Boxplot

```python
sns.boxplot(data=df.drop("Frutto", axis=1))
```

Il **boxplot** (o diagramma a scatola e baffi) rappresenta:
- **Linea centrale**: mediana (50° percentile)
- **Box**: range interquartile (IQR = 75° - 25° percentile)
- **Baffi**: estensione fino a 1.5 × IQR
- **Punti oltre i baffi**: potenziali outlier

È il primo strumento visivo per identificare anomalie nella distribuzione delle feature.

### 5.2 Rilevamento quantitativo con Z-Score

```python
from scipy.stats import zscore

def calcolo_outlier(col_name):
    z = zscore(df[col_name])
    outliers = df[abs(z) > 3]
    print(f"Outlier in '{col_name}': {len(outliers)}")
    return outliers
```

**Z-score**: misura quante deviazioni standard un valore dista dalla media della sua colonna.

```
z = (x - media) / deviazione_standard
```

**Perché soglia = 3?**  
In una distribuzione normale, il 99.73% dei dati cade entro ±3 deviazioni standard dalla media. Valori con |z| > 3 sono quindi statisticamente anomali. È la soglia standard nel mondo scientifico: abbastanza conservativa da non eliminare variazioni naturali, ma abbastanza sensibile da catturare anomalie reali.

**Risultati nel progetto**:
- `Diametro medio (mm)`: 4 outlier → alcuni frutti hanno dimensioni anomale
- `Durezza buccia (1-10)`: 2 outlier → alcuni frutti hanno bucce insolitamente dure
- Altre feature: nessun outlier significativo

**Decisione**: gli outlier non vengono rimossi perché sono pochi (6 su 500 = 1.2%) e probabilmente rappresentano variazioni naturali reali tra esemplari della stessa specie.

### 5.3 Scatter Plot con evidenziazione outlier

```python
sns.scatterplot(y=df[col_name], x=range(len(df)), ...)
sns.scatterplot(y=outliers[col_name], x=outliers.index, color="red", ...)
```

Il **scatter plot** mostra la distribuzione di ogni campione come punto. Gli outlier vengono sovrapposti in rosso per localizzarli visivamente nel dataset.

---

## 6. Encoding delle Variabili Categoriche

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["Frutto"] = encoder.fit_transform(df["Frutto"])

# Mapping risultante:
# {'Arancia': 0, 'Banana': 1, 'Kiwi': 2, 'Mela': 3, 'Uva': 4}
```

### Perché LabelEncoder e non OneHotEncoder?

Il **LabelEncoder** assegna un intero a ogni classe categorica (0, 1, 2, ...).

L'**OneHotEncoder** crea una colonna binaria per ogni classe (es. `[1,0,0,0,0]` per Arancia).

**Regola generale**:
- **LabelEncoder** → usare SOLO sul target (variabile da predire), mai sulle feature di input
- **OneHotEncoder** → usare sulle feature categoriche di input

Nel nostro caso, `Frutto` è il target, quindi LabelEncoder è la scelta corretta. Usarlo su una feature di input creerebbe un falso ordinamento numerico (il modello interpreterebbe "Kiwi=2" come "più grande" di "Banana=1", il che non ha senso).

### `fit_transform` vs `fit` + `transform`

- **`fit_transform(X)`**: impara il mapping dai dati e li trasforma in un solo passaggio
- **`fit(X_train)` + `transform(X_test)`**: impara il mapping dal training set e lo applica separatamente al test set

Per il target si usa `fit_transform` su tutto il dataset prima della suddivisione, perché il target non viene usato per addestrare il modello e non c'è rischio di data leakage.

---

## 7. Matrice di Correlazione

```python
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
```

### `df.corr()`

Calcola la **correlazione di Pearson** tra tutte le coppie di colonne numeriche. Il coefficiente r varia tra -1 e +1:

| Valore di r | Interpretazione |
|-------------|----------------|
| r ≈ +1 | Correlazione positiva forte (al crescere di X cresce Y) |
| r ≈ -1 | Correlazione negativa forte (al crescere di X decresce Y) |
| r ≈ 0 | Nessuna relazione lineare |

### Cosa cercare nella matrice

1. **Correlazioni feature ↔ target**: feature con |r| alto sono i predittori più informativi
2. **Correlazioni feature ↔ feature**: r alto tra due feature suggerisce ridondanza (multicollinearità). Per KNN questo non è un problema critico, ma guida la feature selection

**Risultati del progetto**:
- `Diametro medio ↔ Frutto`: r = -0.51 (correlazione più forte con il target)
- `Peso ↔ Lunghezza media`: r = +0.58 (rischio ridondanza → testato nella feature selection)
- `Diametro medio ↔ Dolcezza`: r = -0.57 (pattern utile per distinguere frutti)

### `sns.heatmap`

```python
sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
```

| Parametro | Descrizione |
|-----------|-------------|
| `annot=True` | Mostra i valori numerici nelle celle |
| `fmt=".2f"` | Formatta i numeri a 2 decimali |
| `cmap="coolwarm"` | Palette: blu per valori negativi, rosso per positivi |
| `cbar=True` | Mostra la barra colore laterale |

---

## 8. Preparazione dei Dati per il Modello

Questa è la sezione più critica: le scelte fatte qui influenzano direttamente le performance del modello.

### 8.1 Suddivisione Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| `test_size=0.2` | 20% test | Standard industria: 80% per training (400 campioni), 20% per valutazione (100 campioni). Abbastanza dati per addestrare e un test set significativo |
| `random_state=42` | Seed fisso | Garantisce la riproducibilità: eseguendo il notebook più volte si ottengono sempre gli stessi risultati |

**Perché non usare il 100% dei dati per il training?**  
Se si usassero tutti i dati per addestrare il modello, non avremmo un insieme di dati "mai visti" per valutarne le performance reali. Il modello potrebbe andare bene sui dati di training ma fallire su nuovi dati (overfitting).

### 8.2 Normalizzazione con StandardScaler

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit + transform sul training
X_test = scaler.transform(X_test)        # solo transform sul test
```

**StandardScaler** trasforma ogni feature in modo che abbia media=0 e deviazione standard=1:

```
x_normalizzato = (x - media_training) / std_training
```

**Perché è FONDAMENTALE per KNN?**

Il KNN calcola distanze tra campioni. Se le feature hanno scale diverse, quelle con valori numericamente più grandi dominano il calcolo della distanza:

- Senza scaling: `Peso` (range 8-111 g) domina `Dolcezza` (range 1-7)
- Con scaling: tutte le feature contribuiscono equamente alla distanza

**Perché fit SOLO sul training set?**

La regola d'oro del machine learning: **il test set non deve mai influenzare il preprocessing**.

Se si facesse `fit` su tutto il dataset (training + test insieme), le statistiche del test set (media, std) influenzerebbero la normalizzazione del training set. Questo si chiama **data leakage** e porta a stime ottimistiche e non realistiche delle performance.

Procedura corretta:
1. `scaler.fit_transform(X_train)`: calcola media e std dal training, trasforma X_train
2. `scaler.transform(X_test)`: usa la media e std del training per trasformare X_test

### 8.3 Ricerca del K ottimale con GridSearchCV

```python
param_grid = {'n_neighbors': range(1, 101)}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_['n_neighbors'])  # K migliore
print(grid_search.best_score_)                   # Accuracy media in CV
```

**GridSearchCV** testa automaticamente tutti i valori di K nel range specificato e seleziona quello con la miglior performance in cross-validation.

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| `param_grid` | K da 1 a 100 | Range ampio che copre il trade-off underfitting/overfitting |
| `cv=5` | 5-fold | Standard: bilanciamento tra affidabilità della stima e costo computazionale |
| `scoring='accuracy'` | Accuracy | Dataset bilanciato (100 campioni per classe): l'accuracy è una metrica affidabile |

**Come funziona la 5-fold Cross-Validation?**

Il training set (400 campioni) viene diviso in 5 parti uguali da 80 campioni:

```
Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]
```

Per ogni valore di K, si addestra il modello 5 volte (una per fold) e si calcola la media degli score. Il K con la media più alta è il migliore.

**Perché 5 fold e non 10?**  
Con 400 campioni di training, 5 fold da 80 campioni ciascuno garantiscono già una stima robusta. 10 fold ridurrebbe il validation set a 40 campioni per fold (meno affidabile) aumentando il tempo di calcolo senza benefici significativi.

---

## 9. Test 1-4: Feature Selection

La **Feature Selection** è il processo di scegliere il sottoinsieme ottimale di feature per il modello. Rimuovere feature ridondanti o poco informative può:
- Migliorare le performance (meno rumore)
- Ridurre il tempo di calcolo
- Migliorare l'interpretabilità

Vengono testati 4 scenari con la stessa funzione `split_and_scale_data`:

| Test | Feature Rimosse | K ottimale | Accuracy CV |
|------|----------------|-----------|-------------|
| 1 | Nessuna (baseline) | 95 | **93.0%** ✓ |
| 2 | Peso (g) | 44 | 88.25% |
| 3 | Lunghezza media (mm) | 11 | 91.25% |
| 4 | Dolcezza (1-10) | 23 | 84.75% |

**Risultato**: mantenere tutte le feature fornisce la migliore performance. Nonostante la correlazione tra Peso e Lunghezza (r=0.58), entrambe contribuiscono informazioni utili per la classificazione.

---

## 10. Test 5: Metriche di Distanza KNN

Il KNN usa una **metrica di distanza** per trovare i K vicini più prossimi. La scelta della metrica influenza quali campioni vengono considerati "simili".

```python
knn = KNeighborsClassifier(n_neighbors=k_ottimale, metric='euclidean')
knn = KNeighborsClassifier(n_neighbors=k_ottimale, metric='manhattan')
knn = KNeighborsClassifier(n_neighbors=k_ottimale, metric='chebyshev')
knn = KNeighborsClassifier(n_neighbors=k_ottimale, metric='minkowski', p=4)
```

### Le 4 metriche testate

**Distanza Euclidea** (p=2, default):
```
d(x,y) = √( Σ (xᵢ - yᵢ)² )
```
Distanza "in linea d'aria" nello spazio n-dimensionale. La più intuitiva e la più comune.

**Distanza di Manhattan** (p=1):
```
d(x,y) = Σ |xᵢ - yᵢ|
```
Somma delle differenze assolute. Simula il percorso su una griglia (come camminare per isolati in una città). È più robusta agli outlier rispetto all'euclidea perché non eleva al quadrato le differenze.

**Distanza di Chebyshev**:
```
d(x,y) = max( |xᵢ - yᵢ| )
```
Considera solo la dimensione con la differenza massima. Utile quando si sospetta che una singola feature sia dominante.

**Distanza di Minkowski** (generalizzazione, p=4):
```
d(x,y) = ( Σ |xᵢ - yᵢ|ᵖ )^(1/p)
```
Con p=1 diventa Manhattan, con p=2 diventa Euclidea, con p→∞ diventa Chebyshev. Valori intermedi come p=4 offrono un comportamento ibrido.

**Nota**: con dati normalizzati (StandardScaler), la distanza euclidea è generalmente la scelta migliore.

---

## 11. Test 6: Pesi nella Votazione KNN

Quando i K vicini votano la classe da assegnare al campione, il parametro `weights` definisce il peso di ogni voto:

```python
knn_uniform = KNeighborsClassifier(n_neighbors=k, weights='uniform')
knn_distance = KNeighborsClassifier(n_neighbors=k, weights='distance')
```

### `uniform` (default)
Tutti i K vicini hanno lo stesso peso nel voto. Il campione viene assegnato alla classe più frequente tra i K vicini.

```
Classe predetta = moda( classi dei K vicini )
```

### `distance`
Ogni vicino pesa inversamente alla sua distanza: i vicini più vicini contano di più.

```
peso_i = 1 / distanza_i
Classe predetta = classe con somma pesi maggiore
```

**Quando preferire `distance`?**  
Con K grande (es. K=95), i 95 vicini includono campioni a distanze molto diverse. Se nelle zone di confine tra classi (es. Arancia vs Kiwi) esistono campioni molto vicini di una classe e campioni lontani dell'altra, la votazione pesata può migliorare la precisione.

---

## 12. Test 7: Confronto tra Algoritmi

Per validare la scelta del KNN, lo confrontiamo con 4 altri algoritmi di classificazione classici. Tutti usano lo stesso training set scalato e lo stesso test set.

```python
algoritmi = {
    'KNN': KNeighborsClassifier(n_neighbors=k_ottimale),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
}

for nome, modello in algoritmi.items():
    modello.fit(X_train, y_train)
    acc = modello.score(X_test, y_test)
```

### I 5 algoritmi

**K-Nearest Neighbors (KNN)**  
Classifica il campione in base alla classe più frequente tra i K vicini più prossimi. Nessuna assunzione sulla distribuzione dei dati. Sensibile alla scala (richiede StandardScaler).

**Decision Tree (Albero Decisionale)**  
Costruisce una serie di regole if-then (es. "se Peso > 50g E Diametro < 30mm → Banana"). Altamente interpretabile. Tende all'overfitting senza pruning.

**Random Forest**  
Ensemble di N alberi decisionali addestrati su sottoinsiemi casuali dei dati e delle feature. Combina le predizioni votando (bagging). Molto robusto, raramente fa overfitting.  
`n_estimators=100`: 100 alberi, buon bilanciamento tra performance e velocità.

**SVM (Support Vector Machine)**  
Trova l'iperpiano che massimizza il margine tra le classi nello spazio delle feature. Il kernel RBF (Radial Basis Function) permette di separare classi non linearmente separabili proiettandole in uno spazio ad alta dimensionalità.

**Naive Bayes Gaussiano**  
Modello probabilistico basato sul teorema di Bayes. Assume che ogni feature sia indipendente dalle altre (assunzione "naive" spesso violata nella realtà) e che seguano una distribuzione gaussiana. Molto veloce, funziona bene anche con pochi dati.

---

## 13. Test 8: Cross-Validation con Diversi K-Fold

```python
for n_fold in [3, 5, 10]:
    scores = cross_val_score(knn, X_scaled, y, cv=n_fold, scoring='accuracy')
    print(f"CV-{n_fold}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### `cross_val_score`

Differenza rispetto a `GridSearchCV`:
- `GridSearchCV`: cerca il miglior iperparametro K testando diversi valori
- `cross_val_score`: valuta un modello già configurato su più fold, restituendo un array di score

**Parametri**:
| Parametro | Descrizione |
|-----------|-------------|
| `estimator` | Il modello da valutare |
| `X` | Feature (tutto il dataset scalato) |
| `y` | Target |
| `cv` | Numero di fold |
| `scoring` | Metrica da calcolare |

### Interpretazione dei risultati

- **Media alta**: il modello performa bene in generale
- **Std bassa**: il modello è stabile e non dipende fortemente da come vengono suddivisi i dati
- **Coerenza tra CV-3, CV-5, CV-10**: se le medie sono simili, il modello è robusto

### Trade-off del numero di fold

| K-Fold | Vantaggi | Svantaggi |
|--------|----------|-----------|
| **3** | Veloce, utile per dataset piccoli | Maggiore varianza nella stima, fold di validation più grandi |
| **5** | Bilanciamento ottimale | — |
| **10** | Stima più accurata (meno bias) | Più lento, fold di validation più piccoli (possibile instabilità con dataset piccoli) |

---

## 14. Modello Finale e Valutazione

```python
knn = KNeighborsClassifier(
    n_neighbors=k_ottimale,
    weights='uniform',
    metric='euclidean'
)
knn.fit(X_train, y_train)

score = knn.score(X_test, y_test)
y_pred = knn.predict(X_test)

# IMPORTANTE: l'ordine corretto è classification_report(y_true, y_pred)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
```

### Metriche di valutazione

**Accuracy**:
```
Accuracy = campioni_classificati_correttamente / totale_campioni
```
Affidabile solo con dataset bilanciati. Con dataset sbilanciati (es. 90% classe A, 10% classe B) un modello che predice sempre A avrebbe 90% di accuracy ma sarebbe inutile.

**Classification Report** — per ogni classe:

| Metrica | Formula | Significato |
|---------|---------|-------------|
| **Precision** | TP / (TP + FP) | Su tutti i campioni predetti come classe X, quanti lo erano davvero? |
| **Recall** | TP / (TP + FN) | Su tutti i campioni realmente di classe X, quanti sono stati trovati? |
| **F1-score** | 2 × (P × R) / (P + R) | Media armonica di precision e recall (bilancia i due) |
| **Support** | — | Numero di campioni reali di quella classe nel test set |

**Errore di classificazione**:
```python
error_rate = 1 - score  # Complemento dell'accuracy
```

### Curva accuracy/errore al variare di K

```python
for k in range(5, 101, 5):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    acc = knn_k.score(X_test, y_test)
```

Visualizza come le performance cambiano al variare di K, confermando visivamente che il K scelto da GridSearchCV è effettivamente ottimale.

**Pattern tipico della curva**:
- K piccoli (1-10): alta varianza, il modello memorizza il training set (overfitting)
- K intermedi: zona ottimale
- K molto grandi: underfitting, il modello diventa troppo generico

---

## 15. Matrice di Confusione

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
```

### Come leggere la matrice di confusione

La matrice ha dimensione N×N dove N è il numero di classi.

```
                  Predetto
              Arancia  Banana  Kiwi  Mela  Uva
Reale Arancia [  8        0      1     1    0 ]   ← 8 corretti, 2 errori
      Banana  [  0       14      0     0    0 ]   ← 14 corretti, 0 errori
      Kiwi    [  2        0     21     1    0 ]   ← 21 corretti, 3 errori
      Mela    [  1        0      0    27    0 ]   ← 27 corretti, 1 errore
      Uva     [  0        0      0     0   24 ]   ← 24 corretti, 0 errori
```

- **Diagonale**: classificazioni corrette
- **Fuori diagonale**: errori (riga = classe reale, colonna = classe predetta)

**Vantaggio rispetto all'accuracy**: rivela DOVE sbaglia il modello. Se due classi si confondono frequentemente, suggerisce che le loro feature sono troppo simili per distinguerle con il dataset attuale.

---

## 16. Analisi degli Errori

```python
df_test = pd.DataFrame({
    'Valore Reale': y_test.values,
    'Valore Predetto': y_pred,
    'Corretto': y_test.values == y_pred
})

errori_arancia = df_test[(df_test['Valore Reale'] == 0) & (~df_test['Corretto'])]
errori_kiwi = df_test[(df_test['Valore Reale'] == 2) & (~df_test['Corretto'])]
```

### Perché Arancia (0) e Kiwi (2) si confondono?

La confusione tra due classi nel KNN indica che i campioni di quelle classi sono **vicini nello spazio delle feature**. Con K=95 (valore alto), il modello considera un'ampia zona e tende ad ammorbidire i confini tra classi simili.

**Possibili soluzioni** (testate e scartate perché peggioravano le performance):
- Bilanciamento artificiale (SMOTE, class_weight)
- K più piccolo (aumento della varianza)

**Conclusione**: con le feature disponibili, un residuo di confusione tra classi simili è fisiologico.

### Verifica bilanciamento delle classi

```python
df["Frutto"].value_counts()
```

Il dataset è perfettamente bilanciato (100 campioni per classe). Questo è importante perché:
- Giustifica l'uso di **accuracy** come metrica principale
- Esclude la necessità di tecniche di bilanciamento (oversampling/undersampling)
- Garantisce che il modello non abbia bias verso classi più numerose

---

## 17. Conclusioni

### Riepilogo di tutti i test

| # | Test | Obiettivo | Tecnica |
|---|------|-----------|---------|
| 1 | Tutte le feature | Baseline | GridSearchCV + 5-fold CV |
| 2 | Senza Peso | Feature selection | GridSearchCV + 5-fold CV |
| 3 | Senza Lunghezza | Feature selection | GridSearchCV + 5-fold CV |
| 4 | Senza Dolcezza | Feature selection | GridSearchCV + 5-fold CV |
| 5 | Metriche di distanza | Iperparametro KNN | Confronto Euclidea/Manhattan/Chebyshev/Minkowski |
| 6 | Pesi votazione | Iperparametro KNN | Confronto uniform/distance |
| 7 | Confronto algoritmi | Validazione scelta KNN | KNN vs DT vs RF vs SVM vs NB |
| 8 | K-Fold CV | Stabilità del modello | cross_val_score con cv=3, 5, 10 |

### Configurazione finale ottimale

```python
KNeighborsClassifier(
    n_neighbors = <K_ottimale>,   # Trovato da GridSearchCV
    weights     = 'uniform',       # Tutti i vicini pesano ugualmente
    metric      = 'euclidean'      # Appropriata con dati standardizzati
)
```

Con preprocessing:
```python
StandardScaler()   # Normalizzazione feature (fit solo su X_train)
LabelEncoder()     # Encoding del target
train_test_split(test_size=0.2, random_state=42)  # 80/20 split
```

### Performance raggiunte

- **Accuracy**: ~94% sul test set
- **Classi perfette**: Banana e Uva (100% recall)
- **Classi problematiche**: Arancia e Kiwi (feature parzialmente sovrapposte)
- **Stabilità**: confermata con CV-3, CV-5 e CV-10

### Lezioni chiave del progetto

1. **L'EDA non è opzionale**: capire i dati prima di modellare evita errori costosi
2. **Lo scaling è critico per KNN**: senza StandardScaler le performance degradano significativamente
3. **Il data leakage è un rischio reale**: fit del scaler sempre e solo sul training set
4. **GridSearchCV è più affidabile del trial-and-error manuale**: esplora sistematicamente lo spazio degli iperparametri
5. **La feature selection va verificata empiricamente**: le correlazioni suggeriscono ipotesi, ma i test le confermano o smentiscono
6. **La matrice di confusione è più informativa dell'accuracy**: rivela dove il modello sbaglia e perché

---

*Guida realizzata per il progetto TropicTaste Inc. — Classificazione KNN di Frutti Esotici*
