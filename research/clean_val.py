"""Clean validation sets for the fine-tune baselines on the suites whose usual validation items come from
Laya's own training mix (ag_news, boolq) - pre-registered with the reviewer.

  boolq:   google/boolq VALIDATION split, seed-0 shuffle (bench.SEED), positions 2000..2299
  ag_news: fancyzhx/ag_news TEST split (no validation split), seed-0 shuffle, positions 2000..2299

Both are disjoint by index from bench.py's test items 0..499, fresh 500..999 / 1000..1499 and fresh-4 1500..1999
(same shuffle), and asserted disjoint by normalised text from those items and from the memory / training items.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites  # noqa: E402
from tune_experience import plain_text  # noqa: E402

CLEAN = {"boolq", "ag_news"}
LO, HI = 2000, 2300


def norm(st):
    return " ".join(plain_text(st).lower().split())


def clean_val(name, train_states):
    """(states, gold) for the clean validation set of `name`, with disjointness asserted."""
    assert name in CLEAN, name
    S = build_suites(HI, {name})[name]
    assert len(S["gold"]) >= HI, "%s has only %d items" % (name, len(S["gold"]))
    states, gold = S["states"][LO:HI], S["gold"][LO:HI]
    used = {norm(s) for s in S["states"][:LO]} | {norm(s) for s in train_states}
    overlap = [i for i, s in enumerate(states) if norm(s) in used]
    # exact duplicates of used items would leak; drop them rather than fail on natural duplicates
    keep = [i for i in range(len(states)) if i not in set(overlap)]
    return [states[i] for i in keep], [gold[i] for i in keep], {"positions": "%d..%d" % (LO, HI - 1), "dropped_duplicates": len(overlap), "n": len(keep)}
