# Guida Progetto: Cyber Security per la Sanità tramite Reinforcement Learning

> **Progetto**: DeepGuard Inc. per GreenGuard Solutions  
> **Obiettivo**: addestrare un agente difensivo che rilevi un attaccante prima che raggiunga i dati sensibili  
> **Ambiente**: [`gym-idsgame`](https://github.com/Limmen/gym-idsgame), scenari `random_attack-v21` e `maximal_attack-v21`  
> **Algoritmi**: SARSA tabellare e Double DQN in PyTorch

## Indice

1. [Obiettivo](#1-obiettivo)
2. [Ambiente e rappresentazione](#2-ambiente-e-rappresentazione)
3. [Installazione su Google Colab](#3-installazione-su-google-colab)
4. [Compatibilità e wrapper](#4-compatibilità-e-wrapper)
5. [Metriche e baseline](#5-metriche-e-baseline)
6. [SARSA](#6-sarsa)
7. [Double DQN](#7-double-dqn)
8. [Risultati](#8-risultati)
9. [Appendice con librerie esterne](#9-appendice-con-librerie-esterne)
10. [Limiti e sviluppi futuri](#10-limiti-e-sviluppi-futuri)

## 1. Obiettivo

Il progetto verifica se il Reinforcement Learning può produrre una strategia difensiva utile in una rete sanitaria simulata. L'agente controlla il **difensore**, mentre l'attaccante è un bot interno all'ambiente:

- **SARSA** viene addestrato sullo scenario `random_attack-v21`;
- **Double DQN (DDQN)** viene addestrato sia su `random_attack-v21` sia su `maximal_attack-v21`.

Nel primo scenario l'attaccante sceglie casualmente tra le mosse legali; nel secondo segue una strategia deterministica e attacca il punto raggiungibile più favorevole.

## 2. Ambiente e rappresentazione

Il notebook usa la versione **v21**, composta da 4 nodi: `Start`, due server intermedi e `Data`. L'attaccante parte da `Start` e può raggiungere `Data` attraverso uno dei due server.

Per ogni nodo il difensore osserva:

- 4 livelli di difesa, uno per ogni tipo di attacco;
- 1 livello di rilevamento.

L'osservazione è quindi una matrice `4 × 5`, appiattita in un vettore di **20 valori**. Anche lo spazio del difensore contiene **20 azioni**: per ciascun nodo è possibile incrementare uno dei quattro attributi di difesa oppure il livello di rilevamento.

I valori arrivano fino a 9. La probabilità di rilevare un attacco fallito è pari al livello di rilevamento diviso 10; un livello pari a 9 corrisponde quindi al 90%.

La versione v21 abilita la reconnaissance e usa la reward `dense_rewards_v3`. Il reward non identifica in modo affidabile l'esito: un rilevamento può avere reward 0 e una violazione può produrre valori negativi variabili. Il notebook determina quindi l'esito leggendo direttamente `env.state.detected` e `env.state.hacked`.

## 3. Installazione su Google Colab

Il notebook installa la versione corrente del repository GitHub e le dipendenze necessarie:

```python
!pip install -q gymnasium
!pip install -q --no-deps "git+https://github.com/Limmen/gym-idsgame.git"
!pip install -q numpy opencv-python matplotlib seaborn torch
```

L'opzione `--no-deps` evita di installare componenti pesanti non necessari all'esecuzione headless. L'appendice installa separatamente TorchRL e Tianshou soltanto se viene eseguita.

## 4. Compatibilità e wrapper

Il notebook gestisce esplicitamente alcune particolarità di `gym-idsgame`:

1. `reset()` e `step()` restituiscono l'osservazione dell'attaccante; quella del difensore viene letta con `env.get_observation()[1]`.
2. `env.action_space` descrive le azioni dell'attaccante; per il difensore si usa `env.defender_action_space`.
3. In v21 lo spazio di osservazione dichiarato non coincide con le osservazioni effettive. La dimensione viene ricavata dai dati restituiti dall'ambiente.
4. Due bot agent fanno riferimento a `idsgame_util` senza importarlo. Un monkey-patch inserisce il modulo mancante come protezione contro il relativo `NameError`.
5. La libreria non distingue correttamente fine naturale e timeout tramite `terminated` e `truncated`; l'esito viene classificato dallo stato reale del gioco.

La classe `DefenderGymWrapper`, condivisa da SARSA e DDQN, espone:

- `reset()` → osservazione del difensore appiattita in `float32`;
- `step(azione)` → osservazione, reward del difensore, `terminated`, `truncated` ed esito;
- `detect_action_for(node_id)` → azione che incrementa il rilevamento del nodo indicato.

Gli esiti possibili sono `detected`, `breached` e `timeout`.

## 5. Metriche e baseline

La valutazione usa 300 episodi e misura:

- tasso di rilevamento (`detected_rate`);
- tasso di violazione (`breached_rate`);
- tasso di timeout (`timeout_rate`);
- reward medio;
- durata media degli episodi.

Il timeout interno è fissato a 100 passi, ma nelle esecuzioni riportate non si verifica. Sono incluse due baseline:

- un difensore casuale;
- l'euristica **Sempre-rileva-Data**, che aumenta sempre il rilevamento del nodo `Data`.

## 6. SARSA

SARSA è implementato con una Q-table sotto forma di `defaultdict`, evitando di preallocare l'intero spazio teorico degli stati. È un algoritmo on-policy: anche l'azione successiva usata nell'aggiornamento viene scelta con la policy epsilon-greedy corrente.

Configurazione principale:

- 5.000 episodi, con valutazioni greedy ogni 1.000;
- `alpha=0.1`;
- `gamma=0.95`;
- epsilon da 1,0 a 0,05, con decadimento moltiplicativo 0,999.

## 7. Double DQN

DDQN usa una rete MLP `20 → 64 → 64 → 20`. La rete online seleziona l'azione successiva e la rete target ne valuta il valore, riducendo il bias di sovrastima del DQN classico.

Configurazione principale:

- 5.000 episodi per scenario, con valutazioni greedy ogni 1.000;
- replay buffer da 20.000 transizioni;
- batch da 64;
- Huber loss e ottimizzatore Adam;
- sincronizzazione hard della rete target ogni 200 aggiornamenti;
- stessa configurazione per `random_attack-v21` e `maximal_attack-v21`.

## 8. Risultati

La tabella riporta la valutazione finale della specifica esecuzione salvata nel notebook, su 300 episodi e con policy greedy.

| Difensore | Scenario | Rilevamenti | Violazioni | Reward medio |
|---|---|---:|---:|---:|
| Difensore casuale | random attack | 51,7% | 48,3% | -12,563 |
| Sempre-rileva-Data | random attack | 77,0% | 23,0% | -5,980 |
| SARSA | random attack | 53,3% | 46,7% | -12,133 |
| DDQN | random attack | 43,7% | 56,3% | -14,647 |
| Difensore casuale | maximal attack | 35,0% | 65,0% | -16,900 |
| Sempre-rileva-Data | maximal attack | 62,7% | 37,3% | -9,707 |
| DDQN | maximal attack | 65,3% | 34,7% | -9,013 |

Su `random_attack`, SARSA supera di poco la baseline casuale ma resta lontano dall'euristica; DDQN risulta inferiore a entrambe. I checkpoint mostrano forti oscillazioni e non una convergenza stabile.

Su `maximal_attack`, DDQN supera nettamente il difensore casuale e, in questa esecuzione, anche l'euristica fissa di circa 2,6 punti percentuali. Il margine ridotto e la variabilità tra esecuzioni non consentono però di affermare una superiorità stabile senza esperimenti con più seed.

## 9. Appendice con librerie esterne

Il notebook contiene un'appendice esplorativa dedicata a TorchRL e Tianshou. Poiché `DefenderGymWrapper` è una normale classe Python e non una sottoclasse completa di `gymnasium.Env`, viene introdotto un adattatore separato con `observation_space` e `action_space` conformi allo standard.

La prova Tianshou usa Double DQN e un budget dimostrativo di 25.000 passi. L'appendice non sostituisce gli esperimenti principali e richiede dipendenze aggiuntive.

## 10. Limiti e sviluppi futuri

- È stato studiato un solo ambiente piccolo, `v21`, con 4 nodi.
- Le prestazioni variano tra esecuzioni; servono più seed, medie, deviazioni standard e intervalli di confidenza.
- Gli agenti non mostrano una convergenza stabile in tutti gli scenari.
- L'attaccante usa una policy predefinita e non si adatta al difensore.
- Il passaggio a reti più grandi rende rapidamente impraticabile SARSA tabellare.

Sviluppi naturali sono la valutazione su topologie più grandi, una ricerca sistematica degli iperparametri, il confronto statistico tra più seed e uno scenario di self-play con attaccante e difensore entrambi adattivi.

---

*Guida aggiornata sulla struttura e sui risultati del notebook `Cyber_Security_Sanita_Reinforcement_Learning.ipynb`.*
