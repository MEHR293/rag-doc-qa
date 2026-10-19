"""Compare RAG vs a vanilla-LLM baseline on a small labelled set.

Metrics
- Answerable questions: correct if the answer contains the expected keyword.
- Unanswerable questions (facts NOT in the docs): a response counts as a hallucination
  unless it declines to answer. RAG should decline; a vanilla LLM tends to invent.

Usage: python -m eval.evaluate
"""
import json
from pathlib import Path

from app.ingest import ingest_path
from app.rag import REFUSAL, RAGPipeline

DECLINE_MARKERS = ["don't know", "do not know", "not provided", "no information",
                   "cannot", "can't", "not specified", "not mentioned"]


def declined(ans: str) -> bool:
    a = ans.lower()
    return REFUSAL.lower() in a or any(m in a for m in DECLINE_MARKERS)


def main():
    p = RAGPipeline()
    p.reset()
    p.add_chunks(ingest_path(Path("data/sample/acme_handbook.txt")))
    qs = json.loads(Path("eval/questions.json").read_text())

    stats = {m: {"correct": 0, "ans": 0, "halluc": 0, "unans": 0} for m in ("rag", "baseline")}
    for item in qs:
        for mode, fn in (("rag", p.ask), ("baseline", p.baseline)):
            ans = fn(item["question"])["answer"]
            s = stats[mode]
            if item["answerable"]:
                s["ans"] += 1
                s["correct"] += any(e in ans for e in item["expected"])
            else:
                s["unans"] += 1
                s["halluc"] += not declined(ans)

    print(f"{'mode':<10}{'accuracy':>12}{'hallucination rate':>22}")
    for mode, s in stats.items():
        acc = f"{s['correct']}/{s['ans']}"
        hal = f"{s['halluc']}/{s['unans']}"
        print(f"{mode:<10}{acc:>12}{hal:>22}")


if __name__ == "__main__":
    main()
