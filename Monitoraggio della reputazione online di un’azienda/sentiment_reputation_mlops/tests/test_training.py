"""Test del retraining: controlli iniziali, checkpoint, report e automazione.
Non scaricano modelli e non pubblicano nulla; il test Trainer usa un modello minuscolo su CPU."""
import ast
from pathlib import Path
from unittest.mock import Mock, patch
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
from tempfile import TemporaryDirectory
from train import ValidationCheckpoint, split_retraining_data, build_replay_training
import contextlib
import io
import json
import tempfile
import human_retrain


# ======================================================================
# Controlli prima di caricare il modello
# ======================================================================

class InternalTrainingPreflightTests(unittest.TestCase):
    def test_default_internal_source_rejects_before_model_loading(self):
        """
        Se i dati approvati sono insufficienti, main() deve fermarsi SUBITO,
        prima di scaricare il modello base o di fissare il seed - non ha
        senso scaricare centinaia di MB di pesi per poi scoprire che manca
        il dataset.

        Tecnica: main() viene estratta dal sorgente di train.py via AST ed
        eseguita in un ambiente dove resolve_base_model() e set_seed() sono
        finte (Mock), per non dover importare davvero transformers/torch
        solo per questo test. review_data.training_budget è patchata per
        sollevare ValueError (simula dati insufficienti): il test verifica
        sia che l'errore risalga fino al chiamante, sia che
        resolve_base_model()/set_seed() non siano MAI state chiamate.
        """
        path = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        resolve = Mock()
        env = {"resolve_base_model": resolve, "set_seed": Mock()}
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), env)
        with patch("review_data.training_budget", side_effect=ValueError("insufficiente")) as budget:
            with self.assertRaisesRegex(ValueError, "insufficiente"):
                env["main"](1500, 200, 3, 42, no_push=True)
        budget.assert_called_once()
        resolve.assert_not_called()
        env["set_seed"].assert_not_called()


# ======================================================================
# Replay, validation e selezione del checkpoint
# ======================================================================

# unittest esegue ogni metodo test_* separatamente; anche pytest li riconosce.
# Questa classe non viene eseguita quando si avvia train.py o il notebook.
class TrainingSelectionTests(unittest.TestCase):
    def test_replay_counts_exclusions_and_reproducibility(self):
        """
        build_replay_training() deve rispettare esattamente il budget
        richiesto, escludere i testi riservati (validation/test) e restare
        riproducibile con lo stesso seed.

        Esempio concreto: 1800 testi "nuovi" e 900 "originali" (TweetEval)
        a disposizione, richiesti 1500 totali con 500 di replay -> il
        risultato deve avere esattamente 1000 testi che iniziano per "new "
        e 500 che iniziano per "old ", nessun duplicato, e nessuno dei due
        testi esclusi ("new 0", "old 0") tra i selezionati. Duplicati
        aggiunti apposta al pool "originale" non devono contare due volte.
        Richiamando la funzione una seconda volta con gli stessi argomenti
        si deve ottenere ESATTAMENTE lo stesso risultato (stesso seed).
        Con n_replay=0, tutti i 1500 testi devono venire dal pool "nuovo".
        """
        new = pd.DataFrame({"text": [f"new {i}" for i in range(1800)], "label": [0, 1, 2] * 600})
        original = pd.DataFrame({"text": [f"old {i}" for i in range(900)], "label": [0, 1, 2] * 300})
        original = pd.concat([original, original.iloc[:10]], ignore_index=True)
        excluded = {"new 0", "old 0"}
        result = build_replay_training(new, original, 1500, 500, 42, excluded)
        self.assertEqual(len(result), 1500)
        self.assertEqual(result.text.str.startswith("new ").sum(), 1000)
        self.assertEqual(result.text.str.startswith("old ").sum(), 500)
        self.assertFalse(result.text.duplicated().any())
        self.assertFalse(set(result.text) & excluded)
        pd.testing.assert_frame_equal(result, build_replay_training(new, original, 1500, 500, 42, excluded))
        without_replay = build_replay_training(new, original, 1500, 0, 42, excluded)
        self.assertTrue(without_replay.text.str.startswith("new ").all())

    def test_replay_rejects_insufficient_non_overlapping_data(self):
        """
        Se le due fonti (nuovo/originale) condividono gli stessi testi, non
        vanno usati due volte - e se la quota richiesta non è realizzabile
        con i testi rimasti, la funzione deve segnalarlo con un errore
        esplicito, non restituire silenziosamente meno esempi del richiesto.

        Esempio: stesso identico pool (3 testi, uno per classe) passato sia
        come "nuovo" sia come "originale". Chiedendo 6 testi totali con 3 di
        replay servirebbero 3 testi "nuovi" + 3 "di replay" diversi tra
        loro, ma il pool "originale" (uguale al "nuovo") non ha testi
        rimasti dopo aver escluso quelli già presi per la parte "nuova" ->
        ValueError, non un risultato con meno di 6 righe.
        """
        pool = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 2]})
        with self.assertRaises(ValueError):
            build_replay_training(pool, pool, 6, 3, 42, set())

    def setUp(self):
        """
        Ricrea, prima di ogni test di questa classe, un modello minuscolo
        (un solo layer lineare, non un vero RoBERTa: qui non serve, si
        testa solo la logica di selezione) e un ValidationCheckpoint con F1
        iniziale 0.70 sui dati nuovi e 0.80 sul benchmark originale -
        questi due numeri sono il "punto di partenza" rispetto a cui i test
        sotto misurano se un checkpoint è un miglioramento o una regressione.
        """
        self.model = torch.nn.Linear(2, 1)
        self.df = pd.DataFrame({"text": ["example"], "sentiment": ["neutral"]})
        self.checkpoint = ValidationCheckpoint(None, self.df, self.df, 0.70, 0.80)

    def test_only_improving_non_regressing_checkpoints_are_selected(self):
        """
        Un'epoca diventa il nuovo checkpoint SOLO se migliora sui dati nuovi
        E non rovina il benchmark originale oltre la tolleranza. Quattro
        epoche di esempio, partendo da F1 iniziale 0.70 (nuovo) / 0.80
        (originale, REGRESSION_TOLERANCE=0.02 di default):

        1. consider(0.69, 0.81) -> F1 nuovo PEGGIORA (0.69 < 0.70): rifiutata.
        2. consider(0.90, 0.70) -> F1 nuovo migliora molto, ma l'originale
           crolla da 0.80 a 0.70 (calo 0.10, oltre la tolleranza 0.02):
           rifiutata comunque, il miglioramento sui nuovi dati non basta.
        3. consider(0.75, 0.79) -> F1 nuovo migliora (0.70 -> 0.75) e
           l'originale cala solo di 0.01 (entro la tolleranza): ACCETTATA,
           diventa il best_epoch (epoca 3) e stale_epochs torna a 0.
        4. consider(0.7505, 0.80) -> F1 nuovo migliora appena (+0.0005),
           sotto il min_delta di 0.001 richiesto per contare come un vero
           miglioramento: rifiutata. Il best_epoch resta quindi la 3.
        """
        self.assertFalse(self.checkpoint.consider(0.69, 0.81, self.model, 1))
        self.assertFalse(self.checkpoint.consider(0.90, 0.70, self.model, 2))
        self.assertTrue(self.checkpoint.consider(0.75, 0.79, self.model, 3))
        self.assertEqual(self.checkpoint.stale_epochs, 0)
        self.assertFalse(self.checkpoint.consider(0.7505, 0.80, self.model, 4))
        self.assertEqual(self.checkpoint.best_epoch, 3)

    def test_checkpoint_is_an_independent_copy_and_restores_weights(self):
        """
        La testa "migliore" salvata da consider() deve essere una copia
        indipendente dei pesi, non un riferimento allo stesso tensore: se il
        modello continua ad allenarsi (i suoi pesi cambiano) dopo che un
        checkpoint è stato selezionato, la copia salvata non deve cambiare
        con lui, e restore() deve poter riportare il modello esattamente a
        quei pesi salvati.

        Il test salva i pesi originali, chiama consider() (che li copia
        internamente), poi altera deliberatamente i pesi del modello
        (+10, per renderli chiaramente diversi) e verifica che restore()
        recuperi esattamente i pesi di partenza, non quelli alterati.
        """
        before = self.model.weight.detach().clone()
        self.checkpoint.consider(0.75, 0.80, self.model, 1)
        with torch.no_grad():
            self.model.weight.add_(10)
        self.assertTrue(self.checkpoint.restore(self.model))
        self.assertTrue(torch.equal(before, self.model.weight))

    def test_no_candidate_and_early_stop(self):
        """
        Se nessuna epoca produce un checkpoint valido, il callback deve
        chiedere al Trainer di fermarsi (early stopping) - e restore() deve
        onestamente dire "niente da ripristinare" (False), non restituire
        un checkpoint fasullo.

        Con patience=2 (default di setUp) e F1 sui dati nuovi bloccata a
        0.60 per due epoche di fila (via patch di train.evaluate, per non
        dover eseguire una vera inferenza) - sempre sotto l'iniziale 0.70 -
        nessuna delle due epoche viene mai accettata: dopo la seconda,
        should_training_stop deve diventare True e best_head deve restare
        None (nessun checkpoint mai salvato).
        """
        self.assertFalse(self.checkpoint.restore(self.model))
        control = SimpleNamespace(should_training_stop=False)
        with patch("train.evaluate", return_value={"f1_macro": 0.60}):
            for epoch in (1, 2):
                self.checkpoint.on_epoch_end(None, SimpleNamespace(epoch=epoch), control, self.model)
        self.assertTrue(control.should_training_stop)
        self.assertIsNone(self.checkpoint.best_head)

    def test_validation_texts_are_disjoint_and_repeatable(self):
        """
        split_retraining_data() deve separare train e validation SENZA
        sovrapposizioni (data leakage: lo stesso testo non può servire sia
        per allenare sia per validare), e con lo stesso seed deve produrre
        sempre la stessa validation.

        Esempio: 60 testi distinti più 3 duplicati (63 righe totali, ma i
        duplicati vanno scartati prima dello split) divisi con n_val=12 ->
        risultato atteso 48 in train + 12 in validation (60 totali, non 63),
        nessun testo condiviso tra i due insiemi. Richiamando la funzione
        una seconda volta con lo stesso seed (42), la validation ottenuta
        deve essere IDENTICA (stesse righe, stesso ordine).
        """
        frame = pd.DataFrame({"text": [f"text {i}" for i in range(60)], "label": [0, 1, 2] * 20})
        split = Dataset.from_pandas(pd.concat([frame, frame.iloc[:3]], ignore_index=True))
        train, validation = split_retraining_data(split, 12, 42)
        self.assertEqual(len(train), 48)
        self.assertEqual(len(validation), 12)
        self.assertFalse(set(train.text) & set(validation.text))
        again = split_retraining_data(split, 12, 42)
        pd.testing.assert_frame_equal(validation, again[1])

    def test_real_trainer_calls_callback_and_keeps_backbone_frozen(self):
        """
        Test di integrazione: un vero Trainer di transformers, su CPU, con
        un RoBERTa minuscolo inizializzato casualmente (stessa architettura
        del modello reale, ma piccolissimo - niente pesi scaricati da
        HuggingFace). Solo le metriche di validazione sono simulate (via
        patch di train.evaluate), il resto (training loop, callback,
        backbone congelato) gira per davvero.

        Scenario simulato, 4 epoche massime ma patience=1: epoca 1 F1 nuovo
        0.75/originale 0.80 (migliora, entro tolleranza -> accettata, best
        epoca 1); epoca 2 F1 nuovo 0.76/originale 0.70 (F1 nuovo migliora
        ancora ma l'originale crolla di 0.10, oltre la tolleranza -> non
        accettata, primo "stallo"). Con patience=1 basta questo per fermare
        il training: deve interrompersi esattamente all'epoca 2 (non
        arrivare alla 3ª/4ª), recuperare i pesi dell'epoca 1 (best_epoch),
        e lasciare INVARIATI tutti i pesi del backbone (congelato: solo la
        testa "classifier.*" deve comparire tra i pesi salvati/modificati).
        """
        model = RobertaForSequenceClassification(RobertaConfig(
            vocab_size=20, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, num_labels=3,
            max_position_embeddings=16,
        ))
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
        frozen = {name: p.detach().clone() for name, p in model.base_model.named_parameters()}
        callback = ValidationCheckpoint(None, self.df, self.df, 0.70, 0.80, patience=1)
        dataset = Dataset.from_dict({
            "input_ids": [[0, 4, 5, 2]] * 4,
            "attention_mask": [[1, 1, 1, 1]] * 4,
            "labels": [0, 1, 2, 0],
        })
        with TemporaryDirectory() as output:
            args = TrainingArguments(output_dir=output, num_train_epochs=3,
                per_device_train_batch_size=2, use_cpu=True, report_to=[],
                save_strategy="no", disable_tqdm=True, dataloader_pin_memory=False)
            trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=[callback])
            with patch("train.evaluate", side_effect=[
                {"f1_macro": 0.75}, {"f1_macro": 0.80},
                {"f1_macro": 0.76}, {"f1_macro": 0.70},
            ]):
                trainer.train()
            self.assertEqual(trainer.state.epoch, 2)
            self.assertEqual(callback.best_epoch, 1)
            self.assertTrue(callback.restore(model))
        for name, parameter in model.base_model.named_parameters():
            self.assertTrue(torch.equal(parameter, frozen[name]))
        self.assertTrue(all(name.startswith("classifier.") for name in callback.best_head))


# ======================================================================
# Confronto prima/dopo ed esito del training
# ======================================================================

class TrainingReportTests(unittest.TestCase):
    def test_reports_precede_rejection_and_publication_stays_blocked(self):
        """
        Verifica il blocco finale di main() (tutto ciò che viene dopo
        trainer.train(): confronto prima/dopo, salvataggio del report,
        decisione di pubblicare o rifiutare) su tre scenari, senza mai
        allenare o scaricare un vero modello.

        REGRESSION_TOLERANCE è fissata a 0.02 e before_original["f1_macro"]
        a 0.8 in tutti e tre gli scenari; cambia solo `score` (la finta F1
        "dopo") e se il checkpoint di validation è stato accettato:
        - accepted=False, score=0.85 -> rifiutato COMUNQUE: la validation
          non è passata, non importa quanto sia buono lo score sul test.
        - accepted=True, score=0.60 -> validation passata, ma il calo di F1
          sul benchmark originale (0.8 -> 0.60 = -0.20) supera di gran lunga
          la tolleranza (0.02): rifiutato dal gate di regressione.
        - accepted=True, score=0.81 -> validation passata e F1 sul benchmark
          originale addirittura migliorata (0.8 -> 0.81): entro tolleranza,
          ACCETTATO.

        In tutti e tre i casi il report va comunque salvato una volta sola
        (anche per una prova rifiutata, per poterla analizzare) e i due
        confronti ("Confronto su nuovo"/"Confronto su originale") devono
        essere stampati PRIMA di un eventuale rifiuto - un rifiuto non deve
        mai nascondere i numeri che lo hanno causato.

        Tecnica: main() viene estratta via AST e tagliata subito dopo la
        riga trainer.train(), cosi' il test esegue il vero codice di
        decisione/pubblicazione senza eseguire il training vero e proprio
        (dataset, tokenizer e modello sono tutti oggetti finti in `env`).
        """
        path = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        index = next(i for i, n in enumerate(main.body)
                     if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == "trainer.train")
        function = ast.FunctionDef(name="report", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=main.body[index + 1:], decorator_list=[])
        compare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "print_comparison")
        code = compile(ast.fix_missing_locations(ast.Module(
            body=[compare, function], type_ignores=[])), str(path), "exec")
        frame = pd.DataFrame({"text": ["example"], "sentiment": ["neutral"], "label": [1]})
        frame.attrs["source_counts"] = {"Training nuovo selezionato": {"neutral": 1}}

        for accepted, score, rejected in [(False, 0.85, True), (True, 0.60, True), (True, 0.81, False)]:
            with self.subTest(accepted=accepted, score=score):
                reports = []
                env = {
                    "checkpoint": SimpleNamespace(restore=lambda model: accepted, best_epoch=1),
                    "retraining_df": frame, "retrain_val_df": frame, "original_val_df": frame,
                    "retrain_train_df": frame, "original_dataset": {"train": SimpleNamespace(to_pandas=lambda: frame)},
                    "LABEL_MAP": {1: "neutral"}, "predictions_before_new": ["neutral"],
                    "predictions_before_original": ["neutral"], "base_model_name": "fake", "new_dataset_name": "nuovo",
                    "n_replay": 0, "n_eval": 1, "n_val": 1, "seed": 42, "learning_rate": 1e-6,
                    "epochs": 1, "trainer": SimpleNamespace(state=SimpleNamespace(epoch=1)),
                    "report_dir": "fake", "save_analysis_report": lambda *a: reports.append(a),
                    "model": object(), "tokenizer": object(),
                    "pipeline": lambda *a, **k: None,
                    "evaluate": lambda *a, **k: {"accuracy": score, "f1_macro": score},
                    "retrain_test_df": frame, "original_test_df": frame,
                    "before_new": {"accuracy": 0.8, "f1_macro": 0.8},
                    "before_original": {"accuracy": 0.8, "f1_macro": 0.8},
                    "RETRAIN_DATASET": "nuovo", "ORIGINAL_BENCHMARK_DATASET": "originale",
                    "REGRESSION_TOLERANCE": 0.02, "no_push": True,
                }
                exec(code, env)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    if rejected:
                        with self.assertRaises(SystemExit) as error:
                            env["report"]()
                        self.assertIn("\033[31m", str(error.exception))
                        self.assertIn("NON viene pubblicato", str(error.exception))
                    else:
                        env["report"]()
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0][1]["candidate_accepted"], not rejected)
                self.assertIn("Confronto su nuovo", output.getvalue())
                self.assertIn("Confronto su originale", output.getvalue())

    def test_notebook_contains_the_same_script(self):
        """
        Il notebook Colab include una copia testuale di train.py (in una
        cella con id "retrain-colab-script", per poterlo eseguire anche
        senza clonare la repository). Questo test si assicura che quella
        copia non sia rimasta indietro rispetto al file vero: se train.py
        viene modificato e la cella del notebook non viene aggiornata di
        conseguenza, questo test fallisce con un semplice confronto testuale.
        """
        root = Path(__file__).resolve().parents[2]
        notebook = json.loads((root / "Monitoraggio_Reputazione_Online_MLOps.ipynb").read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c.get("id") == "retrain-colab-script")
        snapshot = ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)
        script = (root / "sentiment_reputation_mlops/train.py").read_text(encoding="utf-8")
        self.assertEqual(snapshot.strip(), script.strip())


# ======================================================================
# Avvio automatico sui dati approvati
# ======================================================================

class HumanRetrainTests(unittest.TestCase):
    def test_insufficient_queue_never_runs_training(self):
        """
        Se readiness() dice "non pronto" (qui forzata via patch a
        (False, {})), human_retrain.main() non deve MAI arrivare a
        lanciare train.py come sotto-processo - niente training "tentato a
        vuoto" solo perché qualcuno ha lanciato lo script.
        """
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(human_retrain, "STATE_PATH", Path(directory) / "state.json"), \
                 patch.object(human_retrain, "readiness", return_value=(False, {})), \
                 patch.object(human_retrain, "dataset_fingerprint", return_value="fingerprint"), \
                 patch.object(human_retrain.subprocess, "run") as run, \
                 patch("sys.argv", ["human_retrain.py"]):
                human_retrain.main()
                run.assert_not_called()

    def test_rejected_attempt_is_recorded_and_not_repeated(self):
        """
        Un tentativo di training rifiutato (train.py restituisce return
        code 1, qui simulato senza eseguire davvero nulla) deve comunque
        essere registrato in retraining_state.json - e un secondo avvio con
        la STESSA fingerprint del dataset non deve ritentare da capo lo
        stesso identico training già scartato.

        Verifica anche il comando costruito per train.py: deve includere
        --reviewed-data e --no-push (nessun --publish passato in questo
        test), e i valori di --n-train/--n-replay (90/30) devono essere
        esattamente quelli calcolati da training_budget() - non i default
        "demo" di train.py (1500/500).

        Primo main(): readiness pronta, training_budget=(90, 30, counts),
        train.py "fallisce" (returncode=1) -> SystemExit(1), un solo
        run.call_count. Secondo main() (stessa fingerprint "fingerprint",
        senza --force): subprocess.run non deve essere richiamato una
        seconda volta - run.call_count resta 1.
        """
        counts = {"train": {"negative": 20, "neutral": 25, "positive": 30}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(human_retrain, "STATE_PATH", Path(directory) / "state.json"), \
                 patch.object(human_retrain, "readiness", return_value=(True, counts)), \
                 patch.object(human_retrain, "training_budget", return_value=(90, 30, counts)), \
                 patch.object(human_retrain, "dataset_fingerprint", return_value="fingerprint"), \
                 patch.object(human_retrain.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run, \
                 patch("sys.argv", ["human_retrain.py"]):
                with self.assertRaises(SystemExit) as error:
                    human_retrain.main()
                self.assertEqual(error.exception.code, 1)
                command = run.call_args.args[0]
                self.assertIn("--no-push", command)
                self.assertIn("--reviewed-data", command)
                self.assertEqual(command[command.index("--n-train") + 1], "90")
                self.assertEqual(command[command.index("--n-replay") + 1], "30")
                human_retrain.main()
                self.assertEqual(run.call_count, 1)
