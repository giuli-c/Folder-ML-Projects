# Repository Progetti ProfessionAI

Questo repository raccoglie una selezione di progetti di data science e machine learning sviluppati nel percorso ProfessionAI. Ogni cartella contiene un progetto autonomo, generalmente organizzato come notebook Jupyter, con eventuale guida di accompagnamento in formato Markdown.

I progetti coprono casi d'uso diversi: classificazione tabellare, regressione, computer vision, transfer learning e modelli predittivi per applicazioni business.

## Struttura del repository

```text
.
|-- Classificatore frutti esotici/
|   |-- Classificatore_FruttiEsotici_KNN.ipynb
|   `-- GUIDA_PROGETTO.md
|-- Cross selling assicurativo/
|   `-- Cross-Sellling Assicurativo.ipynb
|-- Cyber Security per la sanità tramite Reinforcement Learning/
|   |-- Cyber_Security_Sanita_Reinforcement_Learning.ipynb
|   `-- GUIDA_PROGETTO.md
|-- Data Augmentation per la sicurezza delle centrali elettriche/
|   |-- Data_Augmentation_Sicurezza_Centrali_Elettriche.ipynb
|   `-- GUIDA_PROGETTO.md
|-- Modello previsionale mercato immobiliare/
|   |-- Modello_Previsionale_Mercato_Immobiliare.ipynb
|   `-- GUIDA_PROGETTO.md
|-- Ottimizzazione rete per il settore food/
|   |-- Classificazione_Cibo_GourmetAI.ipynb
|   `-- GUIDA_PROGETTO.md
|-- Riconoscimento di fiori/
|   |-- Riconoscimento di fiori per AgriTech.ipynb
|   `-- GUIDA_PROGETTO_FIORI.md
|-- Ricooscimento Animli CIFAR/
|   |-- Rinoscimento_animali_CIFAR10.ipynb
|   `-- GUIDA_PROGETTO_CIFAR.md
`-- README.md
```

## Progetti contenuti

### 1. Classificatore frutti esotici

Progetto di classificazione multiclasse basato su K-Nearest Neighbors (KNN). L'obiettivo e' classificare automaticamente diverse tipologie di frutti a partire da caratteristiche numeriche come peso, diametro, lunghezza, durezza della buccia e dolcezza.

Il notebook include esplorazione dei dati, analisi degli outlier, encoding del target, normalizzazione delle feature, ricerca del valore ottimale di K, confronto tra metriche di distanza e valutazione finale tramite accuracy, classification report e matrice di confusione.

File principali:
- `Classificatore_FruttiEsotici_KNN.ipynb`
- `GUIDA_PROGETTO.md`

### 2. Cross selling assicurativo

Progetto di machine learning applicato al settore assicurativo. L'obiettivo e' prevedere se un cliente gia' titolare di assicurazione sanitaria possa essere interessato ad acquistare anche una polizza per il veicolo.

Il notebook affronta un problema di classificazione binaria con target sbilanciato. Sono presenti analisi esplorativa, encoding delle variabili categoriche, studio delle correlazioni, osservazioni sulle variabili piu' influenti e gestione dello sbilanciamento tramite tecniche come class weights, oversampling o undersampling.

File principale:
- `Cross-Sellling Assicurativo.ipynb`

### 3. Cyber Security per la sanità tramite Reinforcement Learning

Progetto di Reinforcement Learning applicato alla sicurezza informatica in ambito sanitario.
L'obiettivo e' addestrare un agente difensivo capace di individuare un attaccante che si muove
in una rete simulata prima che raggiunga i dati sensibili dei pazienti, usando l'ambiente
[`gym-idsgame`](https://github.com/Limmen/gym-idsgame).

Il notebook confronta un agente **SARSA** (tabellare, on-policy) sullo scenario *random attack*
con un agente **Double DQN** (rete neurale in PyTorch, off-policy) sugli scenari *random attack*
e *maximal attack*, oltre a due baseline ingenue (difensore casuale e a regola fissa). Include
la verifica a codice di alcune incompatibilita' non documentate della libreria, un wrapper
dedicato per l'ambiente, metriche di valutazione basate su tassi di rilevamento/violazione/timeout
e un'analisi critica di quando l'apprendimento porta davvero un vantaggio misurabile.

File principali:
- `Cyber_Security_Sanita_Reinforcement_Learning.ipynb`
- `GUIDA_PROGETTO.md`

### 4. Modello previsionale mercato immobiliare

Progetto di regressione dedicato alla stima dei prezzi immobiliari. Il caso d'uso riguarda la previsione del valore di una proprieta' sulla base di caratteristiche come superficie, numero di stanze, bagni, piani, servizi disponibili e posizione.

Il notebook confronta modelli di regressione lineare con regolarizzazione: Ridge, Lasso ed Elastic Net. Il flusso include preprocessing, encoding, normalizzazione, train/test split, Grid Search con cross-validation, confronto tramite metriche come MSE, RMSE, MAE e R2, analisi dei coefficienti e valutazione dei residui.

File principale:
- `Modello_Previsionale_Mercato_Immobiliare.ipynb`
- `GUIDA_PROGETTO.md`

### 5. Riconoscimento di fiori

Progetto di computer vision per la classificazione binaria di immagini di fiori, in particolare daisy e dandelion. Utilizza tecniche di transfer learning con EfficientNet-B0 tramite PyTorch e la libreria `timm`.

Il progetto include caricamento immagini con `ImageFolder`, data augmentation, fine-tuning progressivo, confronto tra strategie di training, uso di metriche come F1-score macro, early stopping, scheduler del learning rate, label smoothing e Mixup.

File principali:
- `Riconoscimento di fiori per AgriTech.ipynb`
- `GUIDA_PROGETTO_FIORI.md`

### 6. Riconoscimento animali CIFAR

Progetto di deep learning su immagini basato sul dataset CIFAR-10, trasformato in un problema di classificazione binaria tra animali e veicoli. Il contesto applicativo e' il riconoscimento di oggetti rilevanti per sistemi di guida autonoma.

Il notebook implementa una CNN con PyTorch e confronta piu' esperimenti: baseline, dropout, batch normalization, combinazione dropout e batch normalization, data augmentation, loss pesata e learning rate scheduler. La metrica principale e' il recall della classe animale, per ridurre i falsi negativi.

File principali:
- `Rinoscimento_animali_CIFAR10.ipynb`
- `GUIDA_PROGETTO_CIFAR.md`

### 7. Ottimizzazione rete per il settore food

Progetto di computer vision per GourmetAI Inc.: classificazione multiclasse di immagini di cibo
(14 categorie di piatti) tramite transfer learning con EfficientNet-B0 (PyTorch + `timm`).

Il notebook include esplorazione del dataset (EDA), data augmentation su più livelli, cinque
esperimenti incrementali (baseline, fine-tuning progressivo, fine-tuning completo, augmentation
forte, Mixup), metriche multiclasse (F1 macro/weighted, top-3 accuracy, matrice di confusione
normalizzata con analisi delle coppie di classi più confuse) e un'analisi qualitativa degli errori
sul test set.

File principali:
- `Classificazione_Cibo_GourmetAI.ipynb`
- `GUIDA_PROGETTO.md`

### 8. Data Augmentation per la sicurezza delle centrali elettriche

Progetto di computer vision per CyberEye Solutions: verifica se una pipeline di Data Augmentation generativa (captioning con BLIP, parafrasi con Qwen2.5-1.5B-Instruct, generazione di immagini con sdxl-turbo) applicata al dataset Oxford-IIIT Pet (37 razze) migliora un classificatore EfficientNet-B0 (PyTorch + `timm`) rispetto a un training solo su dati reali.

Il notebook confronta, a parita' di architettura e configurazione (learning rate differenziati, dropout, weight decay, label smoothing), un modello Baseline (solo immagini reali) e un modello Augmented (reali + sintetiche). Include valutazione della qualita' dei sintetici tramite CLIP score, metriche multiclasse (accuracy, top-3 accuracy, precision/recall/F1 macro e weighted, matrice di confusione), error analysis caso per caso e un intervallo di confidenza bootstrap sulla differenza di accuracy tra i due modelli.

File principali:
- `Data_Augmentation_Sicurezza_Centrali_Elettriche.ipynb`
- `GUIDA_PROGETTO.md`

## Come utilizzare il repository

1. Aprire la cartella del progetto di interesse.
2. Consultare la guida Markdown, se presente, per comprendere obiettivo, dataset, scelte metodologiche e risultati.
3. Eseguire il notebook `.ipynb` in Jupyter Notebook, JupyterLab, Visual Studio Code o Google Colab.
4. Installare le librerie richieste indicate nei notebook prima dell'esecuzione.

## Tecnologie principali

- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- PyTorch, torchvision
- timm
- albumentations
- Gymnasium, gym-idsgame
- Jupyter Notebook / Google Colab

## Note

Alcuni notebook scaricano o caricano dataset durante l'esecuzione. In questi casi puo' essere necessaria una connessione internet o il caricamento manuale dei file dati, soprattutto se il notebook viene eseguito in Google Colab.

Le cartelle mantengono i nomi originali dei progetti presenti nel repository.
