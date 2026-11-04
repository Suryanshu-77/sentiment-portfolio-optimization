"""Stage 2: fine-tune roberta-base on financial sentiment (GPU, a few minutes).

Writes artifacts/models/finetuned/.
"""

import logging

from sdpo import config
from sdpo.nlp.finetune import finetune

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    finetune(config.ARTIFACTS_DIR / "models" / "finetuned")
