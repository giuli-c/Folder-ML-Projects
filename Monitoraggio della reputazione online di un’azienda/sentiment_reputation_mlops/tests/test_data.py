"""Test offline dei dati: raccolta Mastodon, revisione e separazione degli split."""
import ast
import html
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import json
import tempfile
from unittest.mock import patch
from review_data import enqueue, load_queue, save_queue, approved_splits, readiness, dataset_fingerprint, training_budget


# ======================================================================
# Raccolta dei post Mastodon
# ======================================================================

class MonitorCollectionTests(unittest.TestCase):
    def test_only_english_relevant_original_posts_are_collected(self):
        """
        Verifica che fetch_recent_posts() (in monitor.py) scarti tutto
        tranne i post davvero utilizzabili, prima ancora di arrivare al modello.

        Il test costruisce una risposta Mastodon finta con 5 varianti dello
        stesso post, di cui solo la PRIMA deve sopravvivere al filtro:
        1. post normale, inglese, non un boost           -> tenuto
        2. language=None (non dichiarato)                 -> scartato
        3. language="it" (non inglese)                     -> scartato
        4. content="unrelated" (non contiene la keyword "acme") -> scartato
        5. reblog non nullo (è un boost, non un post originale) -> scartato

        Controlla anche che l'HTML nel campo "content" venga ripulito
        ("<p>Acme &amp; support</p>" -> "Acme & support") e che il parametro
        limit passato all'API resti sempre <= 40, anche chiedendone di più
        (100, nel test).

        Nota sulla tecnica usata: importare monitor.py per intero
        caricherebbe anche predictor.py e quindi transformers/torch,
        rendendo questo test lento e non più "offline". Per evitarlo, la
        funzione viene estratta dal codice sorgente via AST e rieseguita in
        un ambiente minimo con html/re/requests finti - nessun modello reale
        viene mai caricato qui.
        """
        path = Path(__file__).resolve().parents[1] / "monitor.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fetch_recent_posts")
        post = {"id": "1", "uri": "https://source/posts/1", "url": "https://source/@user/1",
                "content": "<p>Acme &amp; support</p>", "language": "en", "reblog": None}
        response = Mock()
        response.json.return_value = [post, {**post, "language": None},
            {**post, "language": "it"}, {**post, "content": "unrelated"},
            {**post, "reblog": {"id": "boost"}}]
        request = Mock(return_value=response)
        env = {"html": html, "re": re, "requests": SimpleNamespace(get=request, HTTPError=RuntimeError)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), env)
        rows = env["fetch_recent_posts"]("https://instance", 100, ["acme"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["post_id"], post["uri"])
        self.assertEqual(rows[0]["text"], "Acme & support")
        self.assertEqual(request.call_args.kwargs["params"]["limit"], 40)


# ======================================================================
# Coda di revisione e dati approvati
# ======================================================================

class ReviewDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "queue.json"

    def collect(self, texts, scores, audit=0):
        """
        Scorciatoia per i test: simula una chiamata di monitor.py a enqueue().

        Trasforma `texts` e `scores` in post/predizioni finti (etichetta
        sempre "positive": in questi test la classe predetta non conta,
        conta solo la confidence) e li accoda su self.path. Restituisce lo
        stesso numero che restituirebbe enqueue() (post aggiunti davvero,
        non il totale della coda).
        """
        return enqueue([{"post_id": str(i), "text": text} for i, text in enumerate(texts)],
                       [{"sentiment": "positive", "confidence": score} for score in scores],
                       "model", "general_mastodon", .75, audit, self.path)

    def test_training_budget_adapts_and_rejects_missing_data(self):
        """
        training_budget() deve rifiutarsi di partire senza dati, e restringere
        da solo il budget se la classe meno numerosa non ne ha abbastanza.

        Passi del test:
        1. Nessun file di coda -> ValueError (niente da leggere).
        2. File presente, ma con readiness() forzata via patch a dire "pronto,
           con 20/30/25 esempi per classe nel train": chiedendo il budget di
           default (1500 = 1000 nuovi + 500 replay), la classe più scarsa (20)
           non basta per 1000 nuovi, quindi il budget si restringe da solo a
           20*3=60 nuovi + 30 replay (stesso rapporto 2:1) invece di fallire.
        3. Chiedendo invece un budget già piccolo (30 totali, 0 replay), non
           c'è bisogno di restringere nulla: il risultato è esattamente (30, 0).
        4. Tornando alla readiness() vera (senza più il patch) e senza dati
           approvati, la richiesta torna a fallire con ValueError.
        """
        with self.assertRaises(ValueError):
            training_budget(self.path)
        save_queue([], self.path)
        counts = {"train": {"negative": 20, "neutral": 30, "positive": 25}}
        with patch("review_data.readiness", return_value=(True, counts)):
            total, replay, _ = training_budget(self.path, 1500, 500)
            self.assertEqual((total, replay), (90, 30))
            self.assertEqual(training_budget(self.path, 30, 0)[:2], (30, 0))
        with self.assertRaises(ValueError):
            training_budget(self.path)

    def test_threshold_and_audit_are_pending(self):
        """
        Verifica la soglia di confidence (bordo incluso) e il campionamento
        casuale di controllo, più il fatto che ogni post accodato resti
        "pending" (nessuna etichetta assegnata da enqueue()).

        Tre confidence attorno alla soglia di default 0.75, "<=":
        - "a" (0.749) -> incerta, entra;
        - "b" (0.75)  -> incerta, entra (il bordo è incluso);
        - "c" (0.751) -> sicura, esclusa (senza campionamento, audit=0).
        Risultato atteso: solo 2 dei 3 post vengono aggiunti.

        Rilanciando collect() sulle STESSE tre frasi ma con audit=1 (100%
        di campionamento), anche "c" (prima esclusa) viene presa per il
        campione di controllo: la nuova riga aggiunta ha
        selection_reason="random_audit", non "low_confidence".
        """
        self.assertEqual(self.collect(["a", "b", "c"], [.749, .75, .751]), 2)
        rows = load_queue(self.path)
        self.assertTrue(all(r["validated_label"] is None and r["review_status"] == "pending" for r in rows))
        self.assertEqual(approved_splits(self.path), {"train": [], "validation": [], "test": []})
        self.assertEqual(self.collect(["a", "b", "c"], [.749, .75, .751], audit=1), 1)
        self.assertEqual(load_queue(self.path)[2]["selection_reason"], "random_audit")

    def test_dedup_preserves_review(self):
        """
        Una revisione umana già fatta non deve mai essere sovrascritta da
        una raccolta successiva, anche se il testo arriva scritto diversamente.

        Esempio: "Hello World" viene approvato a mano come "negative". Un
        secondo giro di raccolta rimanda quello che è di fatto lo stesso
        post, ma scritto come " HELLO   world " (spazi e maiuscole diverse),
        con un post_id diverso ("other-server-id") e un altro modello/scope
        - text_key() lo riconosce comunque come lo stesso contenuto: la
        coda resta con una sola riga, e "negative" (già approvato) non
        viene toccato dal nuovo arrivo.
        """
        self.collect(["Hello World"], [.5])
        # Riproduce una revisione manuale su dati fittizi del test.
        rows = load_queue(self.path)
        rows[0].update(validated_label="negative", review_status="approved",
                       reviewer="Giulia", reviewed_at="2026-09-19")
        save_queue(rows, self.path)
        enqueue([{"post_id": "other-server-id", "text": " HELLO   world "}],
                [{"sentiment": "neutral", "confidence": .3}], "other-model", "general", path=self.path)
        rows = load_queue(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["validated_label"], "negative")

    def test_only_human_label_is_used_and_exclusion_removes_it(self):
        """
        Solo validated_label (scritta da una persona) deve finire nel
        training, mai predicted_label (la predizione del modello) - ed
        escludere un post lo toglie subito dagli split, senza doverlo
        eliminare dal JSON.

        Il post entra "pending" con predicted_label="positive" (viene dal
        collect(), che finge sempre una predizione "positive"). Dopo
        l'approvazione umana con validated_label="negative", la riga
        approvata compare nello split con label numerica 0 (negative) - non
        2 (positive, cioè quello che aveva detto il modello). Segnandola poi
        come review_status="excluded" (validated_label=None), sparisce da
        tutti gli split, pur restando tracciata nel JSON.
        """
        self.collect(["review this"], [.5])
        # Riproduce una revisione manuale su dati fittizi del test.
        rows = load_queue(self.path)
        rows[0].update(validated_label="negative", review_status="approved",
                       reviewer="Giulia", reviewed_at="2026-09-19")
        save_queue(rows, self.path)
        split = load_queue(self.path)[0]["split"]
        self.assertEqual(approved_splits(self.path)[split][0]["label"], 0)
        rows = load_queue(self.path)
        rows[0].update(validated_label=None, review_status="excluded")
        save_queue(rows, self.path)
        self.assertFalse(any(approved_splits(self.path).values()))

    def test_invalid_or_simulated_approval_fails(self):
        """
        Un JSON di coda modificato a mano in modo scorretto deve bloccare
        approved_splits() con un errore esplicito, non passare come se fosse
        una revisione umana valida.

        Si parte da una riga altrimenti perfettamente approvata, e la si
        "rompe" in quattro modi diversi, uno alla volta:
        - validated_label="wrong" (non è una delle tre classi valide);
        - review_is_simulated=True (dati fittizi, come quelli di questo
          stesso test, marcati esplicitamente come tali - non devono mai
          poter entrare nel training reale);
        - reviewer=None (nessuno risulta aver revisionato il post);
        - reviewed_at=None (nessuna data di revisione registrata).
        Ognuno dei quattro casi, da solo, deve far sollevare ValueError.
        """
        self.collect(["example"], [.5])
        row = load_queue(self.path)[0]
        approved = {**row, "validated_label": "positive", "review_status": "approved",
                    "reviewer": "Giulia", "reviewed_at": "2026-09-19"}
        for changes in [{"validated_label": "wrong"}, {"review_is_simulated": True},
                        {"reviewer": None}, {"reviewed_at": None}]:
            with self.subTest(changes=changes):
                save_queue([{**approved, **changes}], self.path)
                with self.assertRaises(ValueError):
                    approved_splits(self.path)

    def test_readiness_requires_every_class_in_every_split(self):
        """
        readiness() deve guardare OGNI combinazione split×classe
        separatamente (non un totale complessivo) - e dataset_fingerprint()
        deve ignorare l'ordine delle righe ma accorgersi se sparisce un
        esempio approvato.

        Con un solo esempio approvato, readiness() è False (mancano quasi
        tutte le 9 combinazioni di 3 split × 3 classi). Si costruisce quindi
        esattamente 1 esempio approvato per ognuna delle 9 combinazioni, e si
        abbassano i minimi richiesti a 1 per split (via patch di MIN_COUNTS,
        per non dover creare decine di righe finte): a quel punto
        readiness() diventa True. Salvare le stesse righe in ordine inverso
        NON cambia la fingerprint (l'ordine non conta); togliere anche un
        solo esempio approvato la fa tornare non pronta.
        """
        self.collect(["example"], [.5])
        self.assertFalse(readiness(self.path)[0])
        row = load_queue(self.path)[0]
        rows = []
        for split in ("train", "validation", "test"):
            for label in ("negative", "neutral", "positive"):
                rows.append({**row, "text": split + label, "post_id": split + label,
                             "split": split, "validated_label": label, "review_status": "approved",
                             "reviewer": "Giulia", "reviewed_at": "2026-09-19"})
        save_queue(rows, self.path)
        with patch("review_data.MIN_COUNTS", {"train": 1, "validation": 1, "test": 1}):
            self.assertTrue(readiness(self.path)[0])
            before = dataset_fingerprint(self.path)
            save_queue(list(reversed(rows)), self.path)
            self.assertEqual(before, dataset_fingerprint(self.path))
            rows.pop()
            save_queue(rows, self.path)
            self.assertFalse(readiness(self.path)[0])

    def test_duplicate_approved_texts_are_rejected(self):
        """
        Lo stesso testo approvato due volte (qui: una volta nel suo split
        originale e una seconda copia forzata nello split "test") deve
        bloccare approved_splits() con ValueError, non finire silenziosamente
        raddoppiato o tenuto in un solo split - sarebbe una forma di data
        leakage tra split (lo stesso testo visto sia in training sia in test).
        """
        self.collect(["example"], [.5])
        rows = load_queue(self.path)
        rows[0].update(validated_label="neutral", review_status="approved",
                       reviewer="Giulia", reviewed_at="2026-09-19")
        save_queue(rows, self.path)
        rows = load_queue(self.path)
        rows.append({**rows[0], "post_id": "duplicate", "split": "test"})
        save_queue(rows, self.path)
        with self.assertRaises(ValueError):
            approved_splits(self.path)
