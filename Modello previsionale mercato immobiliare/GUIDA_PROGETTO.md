# Guida Progetto: Modello Previsionale per il Mercato Immobiliare

> **Progetto**: RealEstateAI Solutions  
> **Obiettivo**: stimare il prezzo di un immobile a partire dalle sue caratteristiche  
> **Tecniche principali**: regressione lineare, Ridge, Lasso, Elastic Net  
> **Dataset**: 545 immobili, 13 variabili descrittive

---

## Indice

1. [Obiettivo del progetto](#1-obiettivo-del-progetto)
2. [Dataset utilizzato](#2-dataset-utilizzato)
3. [Controllo delle variabili categoriche](#3-controllo-delle-variabili-categoriche)
4. [Preprocessing](#4-preprocessing)
5. [Analisi esplorativa](#5-analisi-esplorativa)
6. [Modelli testati](#6-modelli-testati)
7. [Ricerca degli iperparametri](#7-ricerca-degli-iperparametri)
8. [Confronto dei risultati](#8-confronto-dei-risultati)
9. [Analisi dei coefficienti](#9-analisi-dei-coefficienti)
10. [Analisi dei residui](#10-analisi-dei-residui)
11. [Conclusioni finali](#11-conclusioni-finali)

---

## 1. Obiettivo del progetto

Il progetto costruisce un modello predittivo per stimare il prezzo di un immobile sulla base delle sue caratteristiche principali.

Il problema è di **regressione supervisionata**: il target da prevedere è `price`, una variabile numerica continua.

L'obiettivo non è solo ottenere una previsione accurata, ma anche confrontare modelli diversi per capire:

- quali variabili incidono di più sul prezzo;
- se la regolarizzazione migliora la stabilità del modello;
- se Lasso o Elastic Net riescono a semplificare il modello eliminando feature;
- se gli errori del modello mostrano pattern particolari.

---

## 2. Dataset utilizzato

Il dataset contiene **545 immobili** descritti da caratteristiche fisiche, strutturali e qualitative.

Le variabili principali sono:

| Variabile | Significato |
|---|---|
| `price` | Prezzo dell'immobile, variabile target |
| `area` | Superficie dell'immobile |
| `bedrooms` | Numero di camere da letto |
| `bathrooms` | Numero di bagni |
| `stories` | Numero di piani |
| `mainroad` | Presenza su strada principale |
| `guestroom` | Presenza di stanza per ospiti |
| `basement` | Presenza di seminterrato |
| `hotwaterheating` | Presenza di riscaldamento acqua |
| `airconditioning` | Presenza di aria condizionata |
| `parking` | Numero di posti auto |
| `prefarea` | Presenza in zona preferenziale |
| `furnishingstatus` | Stato di arredamento |

Il notebook controlla inizialmente:

- dimensione del dataset;
- tipi di dato;
- valori mancanti;
- righe duplicate;
- numero di valori distinti per ogni colonna.

---

## 3. Controllo delle variabili categoriche

Una parte importante del progetto riguarda il riconoscimento delle variabili categoriche.

Nel dataset usato nel notebook, tutte le colonne risultano numeriche (`int64`). Questo però non significa che siano tutte variabili quantitative continue.

Alcune colonne sono categoriche già codificate:

- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`

Queste variabili hanno valori `0/1`, quindi rappresentano categorie binarie.

La variabile:

- `furnishingstatus`

ha valori `0/1/2`, quindi rappresenta tre livelli ordinati di arredamento.

Per questo nel notebook è stato aggiunto un controllo sui valori unici di ogni colonna: `df.info()` da solo mostra il tipo tecnico del dato, ma non basta sempre per distinguere una variabile numerica reale da una variabile categorica già codificata.

---

## 4. Preprocessing

Il preprocessing segue questi passaggi:

1. Separazione tra feature `X` e target `y`.
2. Suddivisione in training set e test set.
3. Standardizzazione delle feature con `StandardScaler`.

La standardizzazione è fondamentale perché i modelli regolarizzati penalizzano i coefficienti.

Senza scaling, variabili su scale molto diverse, come `area` e `bedrooms`, verrebbero trattate in modo poco equilibrato. Dopo lo scaling, invece, i coefficienti diventano confrontabili.

Lo scaler viene addestrato solo sul training set:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

Questo evita data leakage, cioè evita che il modello usi informazioni statistiche provenienti dal test set.

---

## 5. Analisi esplorativa

La sezione esplorativa contiene diversi controlli grafici.

### Distribuzione del prezzo

Il notebook confronta:

- distribuzione originale di `price`;
- distribuzione di `log(Price + 1)`.

Il confronto serve a capire se il target è sbilanciato. Nel dataset, la media è maggiore della mediana: questo indica una coda a destra, dovuta a pochi immobili con prezzo molto alto.

La trasformazione logaritmica rende la distribuzione più compatta, ma il progetto mantiene il prezzo nella scala originale per interpretare gli errori in unità economiche reali.

### Boxplot delle variabili numeriche

I boxplot mostrano:

- mediana;
- quartili;
- baffi;
- possibili outlier.

Il grafico evidenzia che `price` e `area` hanno scale molto più ampie rispetto a variabili come `bedrooms`, `bathrooms`, `stories` e `parking`.

Per variabili discrete, come `stories` o `parking`, più outlier possono sovrapporsi nello stesso punto. Per questo il notebook stampa anche il dettaglio degli outlier per valore.

### Matrice di correlazione

La matrice di correlazione mostra quali variabili hanno una relazione lineare più forte con il prezzo.

Le variabili più rilevanti risultano coerenti con il dominio immobiliare:

- `area`;
- `bathrooms`;
- `stories`;
- `airconditioning`;
- `prefarea`.

---

## 6. Modelli testati

Il notebook confronta quattro modelli:

| Modello | Descrizione |
|---|---|
| OLS | Regressione lineare classica, senza regolarizzazione |
| Ridge | Regressione con penalità L2 |
| Lasso | Regressione con penalità L1 |
| Elastic Net | Combinazione di penalità L1 e L2 |

OLS viene usato come baseline.

Ridge, Lasso ed Elastic Net sono modelli regolarizzati: aggiungono una penalità per evitare coefficienti troppo grandi o instabili.

---

## 7. Ricerca degli iperparametri

### Ridge

Per Ridge viene cercato il valore migliore di `alpha` tramite Grid Search.

Il valore scelto è circa:

```text
alpha ≈ 29.15
```

Nel grafico, questo valore si trova nella zona in cui il CV-MSE è basso e stabile. Quando `alpha` diventa troppo alto, il CV-MSE cresce: significa che la penalità diventa eccessiva e il modello perde capacità predittiva.

### Lasso

Nel primo test, il miglior valore di `alpha` per Lasso cadeva sul bordo destro della griglia:

```text
alpha = 1000
```

Per questo è stato aggiunto un test con valori più alti.

La griglia estesa trova:

```text
alpha ≈ 2616.50
```

Il CV-MSE migliora leggermente, ma la differenza è molto piccola. Il test serve soprattutto a verificare che valori molto più alti peggiorino il modello.

### Elastic Net

Elastic Net ottimizza due parametri:

- `alpha`;
- `l1_ratio`.

Il risultato finale è:

```text
alpha = 0.10
l1_ratio = 0.20
```

Questo indica un comportamento più vicino a Ridge che a Lasso.

---

## 8. Confronto dei risultati

Le metriche usate sono:

| Metrica | Significato |
|---|---|
| MSE | Errore quadratico medio |
| RMSE | Radice del MSE, più leggibile perché nella scala del prezzo |
| MAE | Errore assoluto medio |
| R² | Quota di variabilità del prezzo spiegata dal modello |
| CV-MSE | MSE medio in cross-validation |

Il confronto mostra che i modelli sono molto vicini tra loro.

Sul test set, Lasso risulta leggermente migliore:

- MSE più basso;
- R² più alto;
- RMSE leggermente inferiore.

In cross-validation, Ridge ed Elastic Net risultano leggermente più favorevoli come CV-MSE medio.

La differenza però è contenuta: non emerge un vincitore netto.

---

## 9. Analisi dei coefficienti

I coefficienti indicano il peso assegnato dal modello alle variabili.

Poiché le feature sono standardizzate, i coefficienti sono confrontabili tra loro.

Le variabili con peso maggiore risultano:

- `bathrooms`;
- `area`;
- `airconditioning`;
- `stories`;
- `prefarea`.

Questo è coerente con il problema: superficie, numero di bagni, comfort, piani e posizione influenzano fortemente il prezzo.

Un punto importante emerso dopo il test esteso:

```text
Lasso non azzera nessun coefficiente.
Elastic Net non azzera nessun coefficiente.
```

Entrambi mantengono:

```text
12 coefficienti non nulli su 12
```

Quindi in questo progetto Lasso ed Elastic Net non producono un modello più snello: mantengono tutte le variabili e modificano solo il peso assegnato a ciascuna.

---

## 10. Analisi dei residui

I residui sono gli errori del modello:

```text
residuo = prezzo reale - prezzo previsto
```

Se il residuo è vicino a zero, la previsione è buona.

Se il residuo è positivo, il modello ha sottostimato il prezzo.

Se il residuo è negativo, il modello ha sovrastimato il prezzo.

La sezione 12 analizza:

- distribuzione dei residui;
- residui rispetto ai valori predetti;
- Q-Q plot;
- valori reali vs valori predetti.

Dai grafici emerge che:

- la maggior parte degli errori è concentrata vicino allo zero;
- i modelli faticano di più sugli immobili con prezzo alto;
- i residui non sono perfettamente normali, soprattutto nelle code;
- i quattro modelli hanno una struttura degli errori molto simile.

Questo suggerisce che il limite non dipende solo dalla scelta del modello, ma anche dalle informazioni disponibili nel dataset.

---

## 11. Conclusioni finali

Il progetto mostra che Ridge, Lasso ed Elastic Net hanno performance molto simili sul dataset immobiliare.

La conclusione principale è che la regolarizzazione aiuta a controllare i coefficienti, ma non cambia drasticamente la capacità predittiva rispetto alla regressione lineare classica.

### Lettura dei modelli

**Ridge**  
È una scelta conservativa e stabile. Mantiene tutte le feature e applica una regolarizzazione moderata. È adatto quando si vuole evitare che i coefficienti diventino troppo estremi.

**Lasso**  
Dopo il test esteso usa `alpha ≈ 2616.50` ed è leggermente migliore sul test set. Tuttavia non azzera nessun coefficiente, quindi in questo caso non produce un modello più semplice.

**Elastic Net**  
Usa `alpha = 0.10` e `l1_ratio = 0.20`, quindi si comporta più come Ridge che come Lasso. Mantiene tutte le feature e non porta un vantaggio evidente rispetto a Ridge.

### Scelta pratica

Se l'obiettivo è ottenere la metrica migliore sul test set, Lasso è leggermente favorito.

Se l'obiettivo è avere un modello stabile e conservativo, Ridge è una scelta molto solida.

Elastic Net resta utile come confronto, ma in questo esperimento non aggiunge un vantaggio chiaro.

Il miglioramento futuro più promettente non è cambiare modello, ma arricchire il dataset con variabili più informative, ad esempio:

- posizione geografica precisa;
- anno di costruzione;
- stato dell'immobile;
- qualità delle finiture;
- distanza da servizi e trasporti.

---

*Guida realizzata per il progetto RealEstateAI Solutions — Modello previsionale per il mercato immobiliare.*
