"""
Script to GS and fit a classifier on review dataset, to use as feature extractor

INSTRUCTIONS:
1. Set settings variables
    - SK_CLASSIFIER_TYPE
    - SK_CLASSIFIER_PARAMS: dict of parameters that must be passed to the classifier instance
2. Set dataset in variable DATASET
3. Set DO_GRID_SEARCH variable

Classifier/grid search configuration is to be set in "src/text_classification/config/classifier.yml"
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
from torch import nn

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.classes.torch_models.MLP import MLP
from src.text_classification.classes.training.GridSearchUtility import GridSearchUtility
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml
from xgboost import XGBClassifier


def update_params_composite_classifiers(train_config: dict, SK_CLASSIFIER_TYPE: type, SK_CLASSIFIER_PARAMS: dict) -> dict:
    """
    Some classifiers (ensemble, boosting, etc.) may need specific configuration depending on the type.
    For instance, AdaBoost takes an "estimator" argument to set the base estimator.
    This cannot be specified in YAML, and each estimator can have its hyperparameters and grid search params.
    Hence, this method updates the configuration of AdaBoost with all parameters of nested classifiers.
    """
    if SK_CLASSIFIER_TYPE in [AdaBoostClassifier]:  # potentially works for other "nested" cases
        base_est_name = SK_CLASSIFIER_PARAMS.setdefault("estimator", DecisionTreeClassifier()).__class__.__name__
        base_est_config = {f"estimator__{k}": v for k, v in train_config[base_est_name].items()}
        base_est_gs_config = {f"estimator__{k}": vs for k, vs in train_config["grid_search_params"][base_est_name].items()}
        train_config[f"{SK_CLASSIFIER_TYPE.__name__}"].update(base_est_config)
        train_config["grid_search_params"][f"{SK_CLASSIFIER_TYPE.__name__}"].update(base_est_gs_config)
    return train_config


# def target_conversion(x: torch.Tensor):
#     return np.round(torch.sigmoid(x).detach().numpy())


def create_skorch_model_arguments(train_data: pd.DataFrame) -> dict:
    """ Create parameters to train Neural model with skorch """
    layers = [
        (512, 0.1, True, nn.ReLU()),
        (512, 0.2, True, nn.ReLU()),
        (256, 0.1, True, nn.ReLU()),
        (256, 0.1, True, nn.ReLU()),
        (128, 0.1, True, nn.ReLU()),
        (128, 0.2, True, nn.ReLU()),
        (64, 0.1, True, nn.ReLU()),
        (32, 0.1, True, nn.ReLU()),
        (16, 0.1, True, nn.ReLU()),
        (8, 0.1, True, nn.ReLU()),
        (1, 0.1, False, None)
    ]

    network_model = MLP(input_dim=len(train_data.columns) - 1, layers=layers)

    classifier = dict(
        module=network_model,
        callbacks=[EarlyStopping(),
                   Checkpoint(f_params="params_{last_epoch[epoch]}.pt",
                              dirname=Path("dumps") / "nlp_models" / "checkpoints")
                   # EpochScoring("f1", name="valid_f1", lower_is_better=False, target_extractor=target_conversion)
                   ],
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.AdamW,
        verbose=True,
        device="cuda"
    )

    return classifier


DATASET: AbcDataset = CallMeSexistDataset()
DO_GRID_SEARCH = False

EXCLUDE_LIST = [
    "EmpathFeatures_stealing",
    "EmpathFeatures_ship",
    "EmpathFeatures_confusion",
    "EmpathFeatures_surprise",
    "EmpathFeatures_anonymity",
    "EmpathFeatures_gain",
    "EmpathFeatures_torment",
    "EmpathFeatures_positive_emotion",
    "EmpathFeatures_trust",
    "TopicLM_travel_&_adventure",
    "TextGrammarErrors_error_PUNCTUATION",
    "EmpathFeatures_leisure",
    "EmpathFeatures_prison",
    "EmpathFeatures_injury",
    "EmpathFeatures_ancient",
    "LIWCFeatures_leisure",
    "EmpathFeatures_vehicle",
    "EmpathFeatures_healing",
    "EmpathFeatures_religion",
    "EmpathFeatures_competing",
    "EmpathFeatures_programming",
    "EmotionLex_NRC-VAD-Lexicon_arousal_avgLexVal",
    "LIWCFeatures_tentat",
    "EmpathFeatures_ocean",
    "TextEmotion_confusion",
    "EmpathFeatures_sailing",
    "EmpathFeatures_royalty",
    "EmpathFeatures_computer",
    "LIWCFeatures_friend",
    "EmpathFeatures_alcohol",
    "EmpathFeatures_dominant_personality",
    "LIWCFeatures_quant",
    "TextEmotion_neutral",
    "TextGrammarErrors_error_AMERICAN_ENGLISH_STYLE",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_joy_avgLexVal",
    "EmpathFeatures_movement",
    "EvidenceType_Anecdote",
    "EmpathFeatures_appearance",
    "TextGrammarErrors_error_TYPOS",
    "EmpathFeatures_car",
    "EmpathFeatures_government",
    "EmpathFeatures_cooking",
    "EmpathFeatures_speaking",
    "TextGrammarErrors_error_MISC",
    "EmpathFeatures_hipster",
    "EmpathFeatures_weakness",
    "EmpathFeatures_independence",
    "EmpathFeatures_fight",
    "EmpathFeatures_swimming",
    "EmpathFeatures_neglect",
    "TextEmotion_curiosity",
    "EmpathFeatures_shopping",
    "LIWCFeatures_funct",
    "EmpathFeatures_heroic",
    "EmpathFeatures_fun",
    "LIWCFeatures_swear",
    "EvidenceType_None",
    "EmpathFeatures_war",
    "EmpathFeatures_weather",
    "EmpathFeatures_deception",
    "EmpathFeatures_dominant_heirarchical",
    "LIWCFeatures_hear",
    "EmpathFeatures_beauty",
    "EmpathFeatures_kill",
    "EmpathFeatures_achievement",
    "TextGrammarErrors_error_GRAMMAR",
    "EmpathFeatures_weapon",
    "EmpathFeatures_office",
    "EmpathFeatures_pain",
    "EmpathFeatures_musical",
    "LIWCFeatures_posemo",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_sadness_avgLexVal",
    "TextStatistics_difficult_words",
    "TextEmotion_optimism",
    "TextGrammarErrors_error_STYLE",
    "EmpathFeatures_rage",
    "EmpathFeatures_school",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_disgust_avgLexVal",
    "EmpathFeatures_negative_emotion",
    "EmpathFeatures_envy",
    "EmpathFeatures_body",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_anticipation_avgLexVal",
    "EmpathFeatures_farming",
    "GenderBiasDetector_LABEL_0",
    "TextEmotion_nervousness",
    "LIWCFeatures_number",
    "TextStatistics_sents",
    "TextGrammarErrors_error_TYPOGRAPHY",
    "EmpathFeatures_timidity",
    "TextGrammarErrors_error_COLLOCATIONS",
    "EmpathFeatures_fashion",
    "EmpathFeatures_zest",
    "EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_sadness_avgLexVal",
    "EmpathFeatures_breaking",
    "EmpathFeatures_restaurant",
    "TextEmotion_disappointment",
    "EmpathFeatures_monster",
    "EmpathFeatures_valuable",
    "TextEmotion_annoyance",
    "EmpathFeatures_tool",
    "EmpathFeatures_beach",
    "TextStatistics_flesch",
    "EmpathFeatures_money",
    "EmpathFeatures_internet",
    "EmpathFeatures_rural",
    "EmpathFeatures_communication",
    "EmpathFeatures_morning",
    "EmpathFeatures_economics",
    "EmpathFeatures_poor",
    "TextEmotion_caring",
    "EmpathFeatures_superhero",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_fear_avgLexVal",
    "EmpathFeatures_driving",
    "EmpathFeatures_childish",
    "EvidenceType_Testimony",
    "EmpathFeatures_hearing",
    "EmpathFeatures_terrorism",
    "EmpathFeatures_air_travel",
    "EmpathFeatures_divine",
    "LIWCFeatures_certain",
    "EmpathFeatures_shape_and_size",
    "TextEmotion_fear",
    "LIWCFeatures_verb",
    "EmpathFeatures_emotional",
    "LIWCFeatures_insight",
    "EmpathFeatures_banking",
    "EmpathFeatures_disgust",
    "EmpathFeatures_occupation",
    "LIWCFeatures_assent",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_trust_avgLexVal",
    "EmpathFeatures_optimism",
    "EmpathFeatures_death",
    "EmpathFeatures_animal",
    "TextGrammarErrors_error_NONSTANDARD_PHRASES",
    "EmpathFeatures_listen",
    "LIWCFeatures_health",
    "EmpathFeatures_pride",
    "EmpathFeatures_eating",
    "EmpathFeatures_love",
    "LIWCFeatures_auxverb",
    "EmpathFeatures_strength",
    "TextEmotion_remorse",
    "EmpathFeatures_order",
    "EmpathFeatures_magic",
    "EmpathFeatures_hygiene",
    "EmpathFeatures_worship",
    "LIWCFeatures_incl",
    "GibberishDetector_word salad",
    "EmpathFeatures_fear",
    "EmpathFeatures_ugliness",
    "EmpathFeatures_anger",
    "LIWCFeatures_negate",
    "LIWCFeatures_nonfl",
    "EmpathFeatures_hiking",
    "EmpathFeatures_irritability",
    "TextStatistics_avg_sent_length",
    "EmpathFeatures_nervousness",
    "EmpathFeatures_college",
    "LIWCFeatures_preps",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_positive_avgLexVal",
    "EmpathFeatures_health",
    "EmpathFeatures_social_media",
    "LIWCFeatures_negemo",
    "LIWCFeatures_death",
    "EmpathFeatures_wealthy",
    "EmpathFeatures_politeness",
    "EmpathFeatures_sleep",
    "EmpathFeatures_science",
    "LIWCFeatures_money",
    "EmpathFeatures_giving",
    "TextStatistics_automated_readability",
    "EmpathFeatures_aggression",
    "LIWCFeatures_they",
    "EmpathFeatures_noise",
    "EvidenceType_Other",
    "TextGrammarErrors_error_CONFUSED_WORDS",
    "EmpathFeatures_horror",
    "LIWCFeatures_time",
    "EmpathFeatures_smell",
    "TextPolarity_polarity",
    "EmpathFeatures_music",
    "EmpathFeatures_anticipation",
    "EmpathFeatures_negotiate",
    "EmpathFeatures_home",
    "TextEmotion_anger",
    "ParrotAdequacy_neutral",
    "TextGrammarErrors_error_REDUNDANCY",
    "LIWCFeatures_inhib",
    "EmpathFeatures_politics",
    "TextStatistics_avg_letters_per_word",
    "EmpathFeatures_sports",
    "EmpathFeatures_joy",
    "EmpathFeatures_work",
    "EmpathFeatures_phone",
    "EmpathFeatures_violence",
    "EmpathFeatures_party",
    "TextGrammarErrors_error_SEMANTICS",
    "SarcasmDetector_LABEL_1",
    "EmpathFeatures_ridicule",
    "TextEmotion_realization",
    "LIWCFeatures_motion",
    "EmpathFeatures_law",
    "LIWCFeatures_ingest",
    "EmpathFeatures_payment",
    "LIWCFeatures_i",
    "LIWCFeatures_percept",
    "EmpathFeatures_blue_collar_job",
    "EmpathFeatures_hate",
    "TextEmotion_gratitude",
    "ParrotAdequacy_entailment",
    "EvidenceType_Statistics/Study",
    "EmpathFeatures_philosophy",
    "TextPolarity_subjectivity",
    "EmpathFeatures_art",
    "EmpathFeatures_cleaning",
    "EmpathFeatures_meeting",
    "EmpathFeatures_fire",
    "EmpathFeatures_writing",
    "EmpathFeatures_sound",
    "LIWCFeatures_anx",
    "TopicLM_gaming",
    "IronyDetector_irony",
    "EmpathFeatures_exercise",
    "EmpathFeatures_real_estate",
    "TextGrammarErrors_error_BRITISH_ENGLISH",
    "EmpathFeatures_disappointment",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_surprise_avgLexVal",
    "EmpathFeatures_dance",
    "EmpathFeatures_contentment",
    "TopicLM_fitness_&_health",
    "EmpathFeatures_dispute",
    "EmpathFeatures_night",
    "EmpathFeatures_crime",
    "EmpathFeatures_pet",
    "LIWCFeatures_affect",
    "EmpathFeatures_technology",
    "LIWCFeatures_present",
    "EmpathFeatures_business",
    "TextEmotion_disgust",
    "EmpathFeatures_toy",
    "EmpathFeatures_clothing",
    "EmpathFeatures_reading",
    "EmpathFeatures_cheerfulness",
    "EmpathFeatures_plant",
    "EmpathFeatures_legend",
    "LIWCFeatures_relativ",
    "EmpathFeatures_liquid",
    "EmpathFeatures_military",
    "EmpathFeatures_shame",
    "EmpathFeatures_lust",
    "EmpathFeatures_help",
    "EmpathFeatures_warmth",
    "TextEmotion_relief",
    "EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_anger_avgLexVal",
    "EmpathFeatures_urban",
    "EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_fear_avgLexVal",
    "EmpathFeatures_power",
    "LIWCFeatures_adverb",
    "EmpathFeatures_affection",
    "SarcasmDetector_LABEL_0",
    "TopicLM_celebrity_&_pop_culture",
    "EmpathFeatures_suffering",
    "EmpathFeatures_leader",
    "TextEmotion_joy",
    "LIWCFeatures_sad",
    "EmpathFeatures_water",
    "TextEmotion_pride",
    "EmpathFeatures_sadness",
    "EmpathFeatures_journalism",
    "EmpathFeatures_sympathy",
    "TextStatistics_words",
    "TextGrammarErrors_error_COMPOUNDING",
    "EmpathFeatures_messaging",
    "LIWCFeatures_feel",
    "LIWCFeatures_see",
    "LIWCFeatures_ppron",
    "EmpathFeatures_fabric",
    "EmpathFeatures_medieval",
    "EmpathFeatures_exasperation",
    "LIWCFeatures_we",
    "EmpathFeatures_exotic",
    "EmpathFeatures_cold",
    "LIWCFeatures_relig",
    "EmpathFeatures_tourism",
    "EmpathFeatures_celebration",
    "EmpathFeatures_vacation",
    "LIWCFeatures_social",
    "EmpathFeatures_white_collar_job"
]


def main():
    # Define which feature to use, or None to use everything
    keep_features = None
    if EXCLUDE_LIST:
        print(len(EXCLUDE_LIST))
    # ['TextEmotion_admiration', 'TextEmotion_annoyance', 'TextEmotion_pride',
    #              'polarity', 'EmpathFeatures_fun', 'EmpathFeatures_lust',
    #              'EmpathFeatures_messaging',
    #              'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_sadness_avgLexVal']
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, features=keep_features, exclude_features=EXCLUDE_LIST)
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    # SETTINGS:
    # ------------- SK learn classifiers
    SK_CLASSIFIER_TYPE: type = XGBClassifier
    SK_CLASSIFIER_PARAMS: dict = dict()  # dict(estimator=LogisticRegression())

    # ------------- TORCH with SKORCH
    # SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
    # SK_CLASSIFIER_PARAMS: dict = create_skorch_model_arguments(data_train)

    update_params_composite_classifiers(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)

    if DO_GRID_SEARCH:
        gsu = GridSearchUtility(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)
        clf = gsu.grid_search_best_params(data_train, DATASET.compute_metrics)
    else:
        tmu = TrainingModelUtility(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)
        clf = tmu.train_classifier(data_train)
        tmu.evaluate(data_test, DATASET.compute_metrics)

    # TO BE DONE: Implement saving if needed
    save_dir = Path("dumps") / "nlp_models" / clf.__class__.__name__ / f"model_{time.time()}.pkl"
    save_dir.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, save_dir)


if __name__ == "__main__":
    main()
