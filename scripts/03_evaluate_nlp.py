"""Stage 3: compare VADER, FinBERT and the fine-tuned model on the held-out Twitter set.

Writes artifacts/nlp_eval.json and prints which model the pipeline should use.
"""

import json
import logging

from sdpo import config
from sdpo.nlp.evaluate import classification_report_dict
from sdpo.nlp.finetune import load_finetune_data
from sdpo.nlp.score import HFScorer, VaderScorer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test = load_finetune_data()["test"]
    texts, y = test["text"], test["label"]
    scorers = {"vader": VaderScorer(), "finbert": HFScorer("ProsusAI/finbert"),
               "finetuned": HFScorer(str(config.ARTIFACTS_DIR / "models" / "finetuned"))}
    report = {name: classification_report_dict(y, s.predict_labels(texts))
              for name, s in scorers.items()}
    report["best"] = max(scorers, key=lambda n: report[n]["macro_f1"])
    out = config.ARTIFACTS_DIR / "nlp_eval.json"
    out.write_text(json.dumps(report, indent=2))
    for name in scorers:
        print(f"{name:10s} acc={report[name]['accuracy']:.3f} "
              f"macro_f1={report[name]['macro_f1']:.3f}")
    print(f"best: {report['best']} -> {out}")


if __name__ == "__main__":
    main()
