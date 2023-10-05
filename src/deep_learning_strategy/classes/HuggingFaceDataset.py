import re
from typing import Dict

from datasets import Features, Value, ClassLabel, NamedSplit, Dataset

from dataset.ami2020_misogyny_detection.scripts.dataset_handling import train_val_test
from src.feature_extraction.text_features import separate_html_entities


class HuggingFaceDataset:

    def __init__(self, augment_training=False, target="M"):
        self.split_data = train_val_test(target=target, add_synthetic_train=augment_training,
                                         preprocessing_function=self.preprocessing)

        self.hg_wrapped_data = {k: {"text": v["x"], "label": v["y"]} for k, v in self.split_data.items()}

        feat = Features({
            "text": Value("string"),
            "label": ClassLabel(num_classes=2, names=["no", "yes"], names_file=None, id=None)}
        )

        self.train_data = Dataset.from_dict(self.hg_wrapped_data["train"], split=NamedSplit("train"), features=feat)
        self.test_data = Dataset.from_dict(self.hg_wrapped_data["test"], split=NamedSplit("test"), features=feat)

    def get_test_data(self) -> Dataset:
        return self.test_data

    def get_train_data(self) -> Dataset:
        return self.train_data

    def get_train_val_test_split(self) -> Dict:
        return self.split_data

    @staticmethod
    def preprocessing(text_string: str) -> str:
        # Separate EMOJIS from adjacent words if necessary
        text_string = separate_html_entities(text_string)

        # Remove all substrings with < "anything but spaces" >
        text_string = re.sub("<\S+>", "", text_string, flags=re.RegexFlag.IGNORECASE).strip()

        # Remove double spaces
        return re.sub(" +", " ", text_string).strip()
