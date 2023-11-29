from torch import nn
from transformers import Trainer


class BinaryTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # This prevents Roberta to compute its own loss (since it has no target)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float())
        return (loss, outputs) if return_outputs else loss
