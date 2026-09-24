# Guida progetto — FoodVision AI per GourmetAI Inc.

Notebook: `Classificazione_Cibo_Gourmet.ipynb`.

## 1. Obiettivo e stato del lavoro

Classificare fotografie di cibo attraverso transfer learning, valutando augmentation,
fine-tuning e regolarizzazione. La traccia ammette una o più architetture: qui usiamo
EfficientNet-B0 preaddestrata su ImageNet con una testa adattata alle classi rilevate.
Non richiede esplicitamente compressione, pruning o ottimizzazione della latenza.

Questa revisione modifica il protocollo sperimentale: i risultati completi vanno prodotti
eseguendo tutti i nuovi esperimenti. I commenti nel notebook sono indicazioni per leggere
le misure, non risultati già ottenuti. Non è fissata dalla traccia una soglia numerica di successo.

## 2. Esecuzione

1. Aprire il notebook in un ambiente Jupyter con PyTorch/torchvision compatibili; su Colab
   scegliere un runtime GPU. La prima cella installa `timm`.
2. Eseguire setup e preparazione dati. La cella 1 usa `wget` e `unzip` per il file ZIP;
   la cella successiva usa `mkdir` e `mv` per organizzarlo. I controlli evitano download e
   spostamenti ripetuti quando i dati sono già presenti. Questi comandi sono pensati per
   Colab/Linux, non per un kernel Jupyter Windows nativo.
3. Controllare conteggi, integrità, duplicati e campioni visivi.
   Eseguire la cella 5.5 per applicare il piano ai loader prima del training definitivo.
4. Eseguire dalla configurazione degli esperimenti in avanti: sette run controllati e due
   run di approfondimento, con massimo 25 epoche ciascuno (il 9, EfficientNet-B2 a 288 pixel,
   è il più pesante). La cella 1B monta Google Drive e salva lì i checkpoint
   (`FoodVision_checkpoints/`): se la sessione Colab si interrompe, rieseguendo il notebook
   gli esperimenti già completati vengono ricaricati invece di essere riaddestrati.
   Per forzare un nuovo training impostare `resume_from_checkpoint=False` o cancellare il file.
5. Ogni esperimento ha una cella `TEST` dedicata e una successiva cella di analisi dei risultati.
   Eseguirle in ordine; la cella di confronto raccoglie le coppie configurazione/metriche
   e i modelli già calcolati. Se si cambia una configurazione, rieseguire il relativo test
   e poi il confronto; i risultati conservano la configurazione effettivamente eseguita.
6. Completare interpretazione e conclusioni usando i report effettivi. Registrare hardware,
   versioni delle librerie e tempi. Salvare notebook eseguito e checkpoint.

Se si modifica `batch_size`, Mixup richiede un valore pari e almeno un batch completo.
Su ambienti che non supportano bene i worker Jupyter usare `num_workers=0` in `base_cfg`.

## 3. Dataset e split

La versione descritta contiene 14 categorie e 14.000 immagini: 640/160/200 per classe
in train/validation/test, ossia 64%/16%/20%. Il notebook verifica i dati caricati con
`ImageFolder`: i conteggi a runtime sono il riferimento. Usa gli split già forniti e
controlla che la mappa classe → indice coincida fra essi. Il numero delle uscite è ricavato
 dalle classi di training e verificato prima dell'addestramento.

L'augmentation viene applicata dopo la separazione degli split, solo al training.
La scansione completa verifica decodifica, dimensioni, modalità e duplicati byte per byte
di tutti i file indicizzati. Non esclude quasi duplicati o errori semantici nelle etichette.
Nomi di file eterogenei suggeriscono possibili fonti diverse, ma non ne provano la provenienza.

## 4. Componenti del notebook

- `Config`: percorsi, architettura, numero di classi, learning rate, seed, augmentation,
  fine-tuning e regolarizzazione.
- `FilteredImageFolder`: caricamento dei soli file accettati dal piano della cella 5.5.
- `ExperimentRunner`: trasformazioni, loader, modello, loss, ottimizzatore, training,
  validation, early stopping e ripristino del miglior checkpoint.
- `Evaluator`: curve, F1, accuracy, top-3, report per classe, matrice di confusione
  normalizzata per riga e coppie di classi confuse più spesso.

## 5. Preprocessing e augmentation

Dimensione di input, mean, std, interpolazione e crop di valutazione vengono recuperati
 dalla configurazione del modello. Validation e test usano trasformazioni deterministiche.
Il training usa un crop casuale con area relativa 0.7–1.0 e rapporto 0.85–1.15,
più flip orizzontale al 50%. Il flip verticale esplicito è disattivato in tutti i livelli.

| Livello | Color jitter | RandAugment | Random erasing |
|---|---|---|---|
| basic | 0 | disattivato | 0 |
| moderate | 0.2 | m7 | 10% |
| strong | 0.3 | m9 | 20% |

Questi sono pacchetti di trasformazioni: confrontarli non isola ogni singola operazione.
Verificare visivamente che crop, colori e trasformazioni automatiche mantengano riconoscibile
il piatto. Un'intensità maggiore può anche peggiorare il risultato.

## 6. Training e checkpoint

Ogni esperimento riparte dai pesi preaddestrati e dallo stesso seed. La testa è nuova.
Con backbone congelato si allenano soltanto i parametri restituiti da `get_classifier()`;
il feature extractor resta in modalità evaluation, così anche BatchNorm e dropout
non cambiano il comportamento della baseline durante il training.

Lo sblocco in due fasi rende allenabile tutto il backbone dall'epoca 4, non solo gli ultimi
layer. In quel momento AdamW e scheduler vengono ricreati e il contatore di early stopping
azzerato. Non viene applicato un dimezzamento esplicito del learning rate allo sblocco.

Parametri comuni: lr iniziale 5e-4, weight decay 1e-4, batch 32, seed 42,
ReduceLROnPlateau sul F1 di validation (factor 0.5, patience 2), early stopping patience 4.
AMP è attiva solo su CUDA; il risparmio effettivo di memoria dipende dalle operazioni.
Il seed migliora la ripetibilità, ma non garantisce risultati identici tra hardware e versioni.

Il checkpoint migliore è scelto tramite F1 macro di validation, ripristinato e salvato
come `best_<nome_esperimento>.pth`. Contiene lo state dict: per il riuso conservare anche
configurazione, ordine delle classi e preprocessing. I modelli dei run terminati vengono
spostati in RAM CPU; solo il selezionato torna sul device per il test.

## 7. Esperimenti e confronti

| # | Esperimento | Confronto | Fattore modificato |
|---|---|---|---|
| 1 | baseline_basic | riferimento | sola testa, basic |
| 2 | frozen_moderate | 2 vs 1 | basic → moderate |
| 3 | progressive_moderate | 3 vs 2 | sblocco completo dall'epoca 4 |
| 4 | full_ft_moderate | 4 vs 2 | backbone allenabile dall'inizio |
| 5 | full_ft_smoothing | 5 vs 4 | label smoothing 0.1 |
| 6 | strong_aug_label_smoothing | 6 vs 5 | moderate → strong |
| 7 | strong_aug_mixup | 7 vs 6 | Mixup alpha 0.2 |
| 8 | strong_aug_label_smoothing_288px | 8 vs 6 | risoluzione da 224 a 288 pixel (`img_size=288`) |
| 9 | strong_aug_label_smoothing_288px_b2 | 9 vs 8 | EfficientNet-B0 → EfficientNet-B2 (`model_name`) |

Gli esperimenti 8 e 9 approfondiscono la configurazione dell'esperimento 6. Il budget massimo è di
25 epoche per tutti; l'early stopping (patience 4) può fermare prima ogni run.
`img_size` in `Config` sostituisce la risoluzione nativa del modello; mean e std restano quelle
del pretraining. I checkpoint già salvati dei primi sette esperimenti restano validi.

Il confronto 3 vs 4 valuta strategie diverse, includendo il riavvio di ottimizzatore e
scheduler del run in due fasi. Il budget massimo è uguale, ma early stopping può produrre
numeri di epoche differenti: non si rivendica un confronto a parità di tempo di calcolo.

## 8. Loss e metriche

La loss di training è CrossEntropy standard, LabelSmoothingCrossEntropy oppure
SoftTargetCrossEntropy con Mixup. Smoothing e Mixup possono essere combinati.
La loss di validation/test resta CrossEntropy standard.

Con Mixup l'accuracy e il F1 di training sono NaN intenzionalmente: confrontare le immagini
miscelate con una sola delle etichette originali darebbe una misura fuorviante.
La loss media usa il numero di campioni effettivamente elaborati, anche quando viene
scartato un batch incompleto. I grafici train/val vanno interpretati tenendo conto dei
diversi obiettivi e delle augmentation, senza leggere ogni gap come overfitting.

F1 macro è la media non pesata del F1 per classe ed è il criterio di selezione.
Accuracy, F1 weighted, top-3 e matrice di confusione aiutano a interpretarlo. Top-3 significa
che l'etichetta corretta è tra tre proposte, non che una predizione errata sia semanticamente vicina.

## 9. Selezione, test ed errori

La classifica usa il F1 di validation a precisione piena; l'arrotondamento è solo grafico.
A parità esatta prevale l'ordine predefinito. Il test viene valutato solo sul modello selezionato
nel run. La griglia degli errori riutilizza le predizioni del report, senza ripetere
l'inferenza. Non si cambiano gli iperparametri in funzione del risultato del test.

La documentazione precedente riportava prove ridotte anche sul test (F1 0.763 e 0.807).
Non sono misure della revisione corrente e non vengono utilizzate per scegliere il modello.
Quella consultazione storica impedisce di presentare il test come holdout mai osservato:
una verifica pienamente indipendente richiede nuovi dati esterni.

## 10. Risultati ottenuti

| Esperimento | Val F1 macro | Val accuracy | Migliore epoca |
|---|---:|---:|---:|
| 1 baseline_basic | 0.8547 | 0.8537 | 19 (early stopping) |
| 2 frozen_moderate | 0.8654 | 0.8645 | 25 |
| 3 progressive_moderate | 0.9092 | 0.9092 | 23 |
| 4 full_ft_moderate | 0.9128 | 0.9128 | 24 |
| 5 full_ft_smoothing | 0.9194 | 0.9192 | 16 (early stopping) |
| 6 strong_aug_label_smoothing | 0.9229 | 0.9228 | 20 (early stopping) |
| 7 strong_aug_mixup | 0.9211 | 0.9210 | 25 |
| 8 strong_aug_label_smoothing_288px | **0.9281** | **0.9282** | 13 (early stopping) |
| 9 strong_aug_label_smoothing_288px_b2 | 0.9257 | 0.9255 | 11 (early stopping) |

Budget massimo di 25 epoche per tutti gli esperimenti. Il modello selezionato su validation è
`strong_aug_label_smoothing_288px` (fine-tuning completo, augmentation strong, label smoothing 0.1,
immagini 288x288). Sul test ottiene F1 macro 0.9106, accuracy 0.9108 e top-3 0.9735: circa
1.75 punti sotto la validation.

Il guadagno più ampio viene dal fine-tuning del backbone (circa +4.5 punti rispetto al backbone
congelato). Label smoothing (+0.66), augmentation strong (+0.35) e risoluzione 288 (+0.52) danno
ciascuno meno di un punto: con un solo seed non sono significativi presi singolarmente, ma insieme
portano +1.5 punti rispetto al fine-tuning senza regolarizzazione. Mixup non migliora (-0.18).
EfficientNet-B2 (esperimento 9) non migliora: F1 0.9257 contro 0.9281, di fatto equivalente, con gli
errori spostati tra le classi; a parità di prestazioni resta preferibile il modello B0, più leggero.
Le classi più difficili sono Taco (F1 test 0.82), Cheesecake, Apple Pie e Taquito; la risoluzione
maggiore riduce le confusioni tra dessert ma non quella tra Taco e Taquito.

Un primo run con budget di 15 epoche aveva selezionato `strong_aug_label_smoothing`
(F1 test 0.9015); poiché sei esperimenti su sette si erano fermati al limite di epoche, il budget
è stato portato a 25 e sono stati aggiunti gli esperimenti a 288 pixel e con EfficientNet-B2.
Il test è stato quindi consultato più volte durante lo sviluppo: va dichiarato tra i limiti.

Un gap contenuto fra validation e test non dimostra assenza di bias.

Rimangono da valutare generalizzazione a foto esterne, nuove categorie e possibili quasi duplicati
fra split. La qualità sul dataset non equivale automaticamente a idoneità alla produzione.

## 11. Verifiche tecniche della revisione

Validati formato notebook e sintassi delle celle Python. Eseguiti sette run funzionali
su CPU con EfficientNet-B0 senza pesi preaddestrati, immagini sintetiche, quattro classi,
input 32×32 e due epoche. Verificati congelamento dei buffer BatchNorm, sblocco completo,
checkpoint, Mixup con batch finale incompleto, selezione senza arrotondamento, report finale
e griglia degli errori.

Ambiente della verifica: PyTorch 2.14.0+cpu, timm 1.0.29. Questo controllo non scarica il
dataset né i pesi pretrained, non verifica CUDA/AMP e non misura le prestazioni sul cibo.
Il training completo sui dati reali resta da eseguire.

### Verifica dell'allineamento delle celle

La presentazione usa configurazioni esplicite, una cella TEST separata per esperimento e commenti uniformi.
L'allineamento non cambia configurazioni numeriche, training, augmentation o selezione.
Verificati sintassi IPython, equivalenza delle configurazioni e collegamento dei risultati
con esecuzioni simulate. La preparazione dati è stata verificata su un archivio locale
simulando i comandi Colab, anche alla seconda esecuzione; nessun nuovo training completo
o download del dataset reale è stato eseguito per questa modifica.

## Controlli EDA aggiunti prima del training

- Tutti i file indicizzati da ImageFolder, nei tre split: `verify()` e decodifica `load()`,
  larghezza, altezza, aspect ratio, modalità, numero di canali e formato.
- Hash SHA-256 per copie identiche byte per byte; segnalazione dei gruppi tra split
  e con classi diverse. Non rileva foto ricompresse, ridimensionate o quasi identiche.
- Controllo visivo delle etichette nella 5.4; anteprima basic/moderate/strong
  nella cella CONTROLLO INPUT, su un campione delle immagini conservate dalla 5.5.
- Batch di controllo: input RGB, float32, forma prevista, valori finiti e range dopo
  ToTensor e normalizzazione. La normalizzazione usa mean/std del pretrained, non 0.5 arbitrari.

I report completi vengono salvati in `dataset_audit/`. I file non leggibili fermano la cella
con un elenco da correggere; nessun file viene eliminato automaticamente. I duplicati vengono
segnalati per revisione prima di interpretare i risultati. Non modificare gli split in base
alle performance del modello. Il test è incluso solo nei controlli tecnici, non nelle
anteprime o nel confronto visivo usato per scegliere le trasformazioni.

La scansione legge l'intero dataset e può richiedere alcuni minuti. Il controllo dei tensori
è campionato in CONTROLLO INPUT; etichette sbagliate, orientamenti insoliti e quasi duplicati richiedono revisione
visiva. Questi controlli migliorano la copertura senza certificare l'assenza di ogni anomalia.

Verifica tecnica delle nuove celle: superate prove su file sintetici RGB/L/RGBA, file
corrotto, duplicati tra split e con etichette diverse, assenza di duplicati, grafici e
trasformazioni reali con batch 4×3×224×224. Non è stata eseguita qui la scansione del
dataset food completo: i report effettivi si ottengono eseguendo il notebook aggiornato.

### Analisi dei duplicati rilevati

La cella 5.2, dopo il report di integrità, conta i gruppi e prepara una proposta:
una copia per hash con priorità test, validation, training; tutti i membri dei gruppi
con etichette diverse restano da revisionare. Mostra le immagini in conflitto e salva
`duplicate_groups.csv`, `duplicate_plan.csv` e `counts_after_proposal.csv` in `dataset_audit/`.
I conteggi indicano cosa cambierebbe; nessun file o loader viene modificato dalla cella.
Per applicare la scelta eseguire la cella 5.5: filtra i loader e controlla
che nessuna classe diventi vuota; i file originali restano invariati.

### Semplificazione EDA e stato della pulizia

Le celle 6 e 6A sono state rimosse. CONTROLLO INPUT estrae da sola il proprio campione
dalle immagini conservate e verifica dtype, shape e range dei tensori e le augmentation. Non si duplicano i grafici del controllo visivo 5.4.

La priorità della deduplicazione è test > validation > train, coerente in codice e commenti.
La revisione dei due conflitti porta a escludere tutti i cinque file (pizza incompatibile
con le etichette, hamburger ambiguo). La cella 5.5 applica il piano tramite una lista di file ammessi: i loader
usano solo i record `keep`, conservando gli originali. Verifica conteggi, classi non vuote
e unicità degli hash prima dell’utilizzo.

### Applicazione effettiva: cella 5.5

Eseguire la 5.5 dopo la revisione dei conflitti, quindi tutte le celle successive.
`accepted_paths` contiene i percorsi ammessi; `base_cfg.allowed_paths` li trasferisce
a tutti gli esperimenti. `FilteredImageFolder` applica il filtro a train/val/test.
Senza una lista accettata i loader non partono.
CONTROLLO INPUT usa un campione delle immagini conservate.

I report `accepted_duplicate_plan.csv`, `retained_images.csv` e `clean_counts.csv`
documentano le decisioni. Non si modificano i file originali. Se il contenuto del dataset
cambia, ripetere l'audit SHA-256 e l'accettazione. Dopo una nuova accettazione ricreare
configurazioni, loader e modelli; le istanze già create non vengono aggiornate in automatico.
