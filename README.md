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
|-- Modello previsionale mercato immobiliare/
|   |-- Modello_Previsionale_Mercato_Immobiliare.ipynb
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

### 3. Modello previsionale mercato immobiliare

Progetto di regressione dedicato alla stima dei prezzi immobiliari. Il caso d'uso riguarda la previsione del valore di una proprieta' sulla base di caratteristiche come superficie, numero di stanze, bagni, piani, servizi disponibili e posizione.

Il notebook confronta modelli di regressione lineare con regolarizzazione: Ridge, Lasso ed Elastic Net. Il flusso include preprocessing, encoding, normalizzazione, train/test split, Grid Search con cross-validation, confronto tramite metriche come MSE, RMSE, MAE e R2, analisi dei coefficienti e valutazione dei residui.

File principale:
- `Modello_Previsionale_Mercato_Immobiliare.ipynb`
- `GUIDA_PROGETTO.md`

### 4. Riconoscimento di fiori

Progetto di computer vision per la classificazione binaria di immagini di fiori, in particolare daisy e dandelion. Utilizza tecniche di transfer learning con EfficientNet-B0 tramite PyTorch e la libreria `timm`.

Il progetto include caricamento immagini con `ImageFolder`, data augmentation, fine-tuning progressivo, confronto tra strategie di training, uso di metriche come F1-score macro, early stopping, scheduler del learning rate, label smoothing e Mixup.

File principali:
- `Riconoscimento di fiori per AgriTech.ipynb`
- `GUIDA_PROGETTO_FIORI.md`

### 5. Riconoscimento animali CIFAR

Progetto di deep learning su immagini basato sul dataset CIFAR-10, trasformato in un problema di classificazione binaria tra animali e veicoli. Il contesto applicativo e' il riconoscimento di oggetti rilevanti per sistemi di guida autonoma.

Il notebook implementa una CNN con PyTorch e confronta piu' esperimenti: baseline, dropout, batch normalization, combinazione dropout e batch normalization, data augmentation, loss pesata e learning rate scheduler. La metrica principale e' il recall della classe animale, per ridurre i falsi negativi.

File principali:
- `Rinoscimento_animali_CIFAR10.ipynb`
- `GUIDA_PROGETTO_CIFAR.md`

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
- Jupyter Notebook / Google Colab

## Note

Alcuni notebook scaricano o caricano dataset durante l'esecuzione. In questi casi puo' essere necessaria una connessione internet o il caricamento manuale dei file dati, soprattutto se il notebook viene eseguito in Google Colab.

Le cartelle mantengono i nomi originali dei progetti presenti nel repository.
