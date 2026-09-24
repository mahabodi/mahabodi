"""Python binding tests. Model-dependent tests run only when BODI_LAYA_DIR is set, and say so."""
import os
import threading
import unittest

from mahabodi import Bodi, __version__

FIX = os.path.join(os.path.dirname(__file__), "..", "..", "..", "crates", "mahabodi-core", "tests", "fixtures")


def fixture(name):
    with open(os.path.join(FIX, name + ".md"), encoding="utf-8") as f:
        return f.read()


class MemoryTests(unittest.TestCase):
    def test_ingest_query_density(self):
        b = Bodi()
        r = b.ingest(fixture("robotics"), source="robotics")
        self.assertTrue(r["density"]["passes"], r)
        q = b.query("spacecraft", k=3)
        self.assertTrue(q["matched"])
        self.assertEqual(q["stage"], "exact")
        self.assertTrue(all(h["id"].startswith("F_") for h in q["hits"]))
        miss = b.query("zzqxv wkkpj")
        self.assertFalse(miss["matched"])
        self.assertTrue(miss["handoff"])
        self.assertEqual(b.context("zzqxv wkkpj")["context"], "")
        self.assertTrue(b.density()["passes"])

    def test_snapshot_restore_and_errors(self):
        b = Bodi({"auto_density": True})
        b.ingest("Refunds take five days. Escalate after two days.", format="text")
        snap = b.snapshot()
        b2 = Bodi()
        b2.restore(snap)
        self.assertEqual(b2.query("refunds")["hits"], b.query("refunds")["hits"])
        with self.assertRaises(ValueError):
            b.call("nope")
        with self.assertRaises(ValueError):
            b.decide("x", {"q": {"type": "noul", "instructions": "?"}})  # no model loaded

    def test_threads(self):
        b = Bodi()
        b.ingest(fixture("world_events"))
        errs = []

        def work(i):
            try:
                for j in range(20):
                    if j % 7 == 0:
                        b.ingest("Thread %d note %d about space elevators." % (i, j), format="text")
                    b.query("elevator")
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])

    def test_version(self):
        self.assertTrue(__version__)


@unittest.skipUnless(os.environ.get("BODI_LAYA_DIR"), "SKIPPED: set BODI_LAYA_DIR to run Laya model tests")
class LayaTests(unittest.TestCase):
    def test_decide(self):
        b = Bodi()
        b.load_laya(os.environ["BODI_LAYA_DIR"])
        q = {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                       "criteria": {"world": "world news", "sports": "sports", "business": "business", "sci_tech": "science and technology"}}}
        r = b.decide({"article": "The Lakers beat the Celtics 110-102 in overtime on Sunday."}, q)
        self.assertEqual(r["answers"]["topic"]["choice"], "sports")
        p = b.predict({"article": "The Lakers beat the Celtics 110-102 in overtime on Sunday."}, q)
        self.assertEqual(p["answers"]["topic"]["choice"], "sports")
        b.ingest("## [ID: Refund_Policy]\n**Action:** Refund\n**Logic:** Duplicate charges are always refunded within five days.\n")
        g = b.decide_with_memory("I was charged twice for one order", {"refund": {"type": "noul", "instructions": "Should we refund?"}}, query="duplicate charges refund")
        self.assertTrue(g["memory"]["used"])
        self.assertIn("noul", g["answers"]["refund"])


@unittest.skipUnless(os.environ.get("BODI_LAYA_DIR") and os.environ.get("BODI_EMBEDDER_DIR"),
                     "SKIPPED: set BODI_LAYA_DIR and BODI_EMBEDDER_DIR to run experience-memory tests")
class ExperienceTests(unittest.TestCase):
    def test_learn_changes_decisions_and_forget_restores(self):
        b = Bodi()
        b.load_laya(os.environ["BODI_LAYA_DIR"])
        b.load_embedder(os.environ["BODI_EMBEDDER_DIR"])
        q = {"t": {"type": "choice", "instructions": "Which team handles this?", "criteria": ["alpha", "beta"]}}
        base = b.decide("my blue widget stopped working", q, cache=False)["answers"]["t"]
        b.learn(["the blue widget broke", "blue widget failing again", "red gadget is late", "red gadget shipment delayed"], q,
                [{"t": "alpha"}, {"t": "alpha"}, {"t": "beta"}, {"t": "beta"}])
        self.assertEqual(b.stats()["experience_cases"], 4)
        mem = b.decide("my blue widget stopped working", q, cache=False, experience_k=2, experience_weight=3.0)["answers"]["t"]
        self.assertEqual(mem["choice"], "alpha")
        self.assertIn("experience", mem["bodi"])
        self.assertEqual(len(b.embed_text(["x", "y"])[0]), 384)
        b.forget()
        again = b.decide("my blue widget stopped working", q, cache=False)["answers"]["t"]
        self.assertEqual(again["probabilities"], base["probabilities"])
        with self.assertRaises(ValueError):
            b.learn(["x"], q, [{"t": "gamma"}])  # unknown label


if __name__ == "__main__":
    unittest.main()
