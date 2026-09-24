"""Golden cases for Rust<->Laya parity: token ids/markers from Laya's own build_sequence and
answers from Laya's own PyTorch Agent.predict. Cases target the known divergence risks:
dict key order (states and choice criteria), float formatting, list states (left
truncation), long states, labels, many options (per-option budget truncation), unicode.

    .venv/bin/python research/golden_parity.py models/laya-v2 crates/mahabodi-core/tests/fixtures/laya_parity.json
"""
import json, os, sys
os.environ.setdefault("USE_TF", "0")
from laya import Agent
from laya.common import build_sequence

model_dir, out = sys.argv[1], sys.argv[2]
subfolder = sys.argv[3] if len(sys.argv) > 3 else None
agent = Agent("convaiinnovations/laya", subfolder=subfolder, device="cpu")
long_state = {"ticket": " ".join("Customer reports error code %d on checkout page after update %d." % (i, i) for i in range(120))}
cases = [
    ({"zeta": "last key first", "alpha": 1, "mid": [1, 2.5, None], "nested": {"b": 1e-07, "a": 1e16}},
     {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                "criteria": {"world": "world news", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}}}),
    ({"article": "The Lakers beat the Celtics 110-102 in overtime on Sunday."},
     {"topic": {"type": "choice", "instructions": "Topic?", "criteria": ["world", "sports", "business", "sci_tech"]},
      "positive": {"type": "noul", "instructions": "Is this good news for Lakers fans?"},
      "excite": {"type": "score", "instructions": "How exciting?", "criteria": ["dull", "ok", {"desc": "thrilling", "w": 0.5}]}}),
    ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello, how can I help?"},
      {"role": "user", "content": "I was charged twice for order 1234, please refund the duplicate."}] * 30,
     {"refund": {"type": "noul", "instructions": "Does the user ask for a refund?", "criteria": {"true": "asks for money back"}, "labels": {"false": "B", "true": "A"}}}),
    (long_state, {"sev": {"type": "score", "instructions": "Severity", "criteria": ["low", "medium", "high", "critical"]}}),
    ("Ich wurde zweimal belastet — bitte erstatten. Café ☕ naïve résumé",
     {"lang": {"type": "choice", "instructions": "Which language?", "criteria": {"de": "German", "en": "English", "fr": "French"}}}),
    ("my card got declined at the atm",
     {"intent": {"type": "choice", "instructions": "Which banking intent?",
                 "criteria": {"intent %02d %s" % (i, w): None for i, w in enumerate(
                     ["card declined", "atm withdrawal", "refund status", "exchange rate", "top up failed", "pin blocked",
                      "lost card", "transfer pending", "cash fee", "card arrival", "verify identity", "change pin",
                      "direct debit", "apple pay", "virtual card", "wrong amount", "country support", "beneficiary", "fiat", "crypto"])}}}),
]
if subfolder == "multilingual":
    cases += [
        ("मेरे खाते से दो बार पैसे कट गए, कृपया वापस करें",
         {"refund": {"type": "noul", "instructions": "Does the user ask for a refund?"}}),
        ({"utterance": "明天早上七点叫醒我"},
         {"intent": {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
                     "criteria": {"alarm_set": "alarm set", "weather_query": "weather query", "play_music": "play music"}}}),
        ("Мой счёт списали дважды, верните деньги",
         {"sev": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}}),
    ]
golden = []
for state, qs in cases:
    rows = []
    for qid, qdef in qs.items():
        q = agent._to_internal(qdef)
        ids, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"],
                                      truncate_left=isinstance(state, list))
        rows.append({"qid": qid, "ids": ids, "markers": markers})
    golden.append({"state": state, "questions": qs, "rows": rows, "laya": agent.predict(state, qs)["answers"]})
with open(out, "w", encoding="utf-8") as f:
    json.dump(golden, f, ensure_ascii=False)
print("wrote %d cases to %s" % (len(golden), out))
