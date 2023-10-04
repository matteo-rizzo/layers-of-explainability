import re

from dataset.ami2020_misogyny_detection.scripts.dataset_handling import train_val_test
from src.feature_extraction.text_features import separate_html_entities


class MisogynyDataset:

    def __init__(self, augment_training=False, target="M"):
        self.split_data = train_val_test(target=target, add_synthetic_train=augment_training,
                                         preprocessing_function=self.preprocessing)

    def get_test_data(self):
        return self.split_data["test"]["x"]

    def get_test_ids(self):
        return self.split_data["test"]["ids"]

    def get_synthetic_test_data(self):
        return self.split_data["test_synt"]["x"]

    def get_synthetic_test_ids(self):
        return self.split_data["test_synt"]["ids"]

    def get_test_groundtruth(self):
        return self.split_data["test"]["y"]

    def get_train_val_test_split(self):
        return self.split_data

    def get_path_to_testset(self):
        return self.split_data["test_set_path"]

    @staticmethod
    def preprocessing(text_string: str) -> str:
        # Separate EMOJIS from adjacent words if necessary
        text_string = separate_html_entities(text_string)

        # Remove all substrings with < "anything but spaces" >
        text_string = re.sub("<\S+>", "", text_string, flags=re.RegexFlag.IGNORECASE).strip()

        # Remove double spaces
        return re.sub(" +", " ", text_string).strip()
