import pandas as pd
from pyfume import *


# 'TextStatistics_words	TextStatistics_sents	TextStatistics_avg_sent_length	TextStatistics_avg_syllables_per_word'
# '	TextStatistics_avg_letters_per_word	TextStatistics_flesch	TextStatistics_automated_readability	'
# 'TextStatistics_dale_chall	TextStatistics_lix	LIWCFeatures_funct	LIWCFeatures_conj	LIWCFeatures_cogmech	'
# 'LIWCFeatures_tentat	LIWCFeatures_excl	LIWCFeatures_pronoun	LIWCFeatures_ppron	LIWCFeatures_i	LIWCFeatures_we	'
# 'LIWCFeatures_you	LIWCFeatures_shehe	LIWCFeatures_they	LIWCFeatures_ipron	LIWCFeatures_article	'
# 'LIWCFeatures_verb	LIWCFeatures_auxverb	LIWCFeatures_past	LIWCFeatures_present	LIWCFeatures_future	'
# 'LIWCFeatures_adverb	LIWCFeatures_preps	LIWCFeatures_negate	LIWCFeatures_quant	'
# 'LIWCFeatures_number	LIWCFeatures_swear	LIWCFeatures_social	LIWCFeatures_family	'
# 'LIWCFeatures_friend	LIWCFeatures_humans	LIWCFeatures_affect	LIWCFeatures_posemo	'
# 'LIWCFeatures_negemo	LIWCFeatures_anx	LIWCFeatures_anger	LIWCFeatures_sad	'
# 'LIWCFeatures_insight	LIWCFeatures_cause	LIWCFeatures_discrep	LIWCFeatures_certain	'
# 'LIWCFeatures_inhib	LIWCFeatures_incl	LIWCFeatures_percept	LIWCFeatures_see	'
# 'LIWCFeatures_hear	LIWCFeatures_feel	LIWCFeatures_bio	LIWCFeatures_body	'
# 'LIWCFeatures_health	LIWCFeatures_sexual	LIWCFeatures_ingest	LIWCFeatures_relativ	'
# 'LIWCFeatures_motion	LIWCFeatures_space	LIWCFeatures_time	LIWCFeatures_work	'
# 'LIWCFeatures_achieve	LIWCFeatures_leisure	LIWCFeatures_home	LIWCFeatures_money'
# '	LIWCFeatures_relig	LIWCFeatures_death	LIWCFeatures_assent	LIWCFeatures_nonfl	'
# 'LIWCFeatures_filler	EmpathFeatures_help	EmpathFeatures_office	EmpathFeatures_dance	'
# 'EmpathFeatures_money	EmpathFeatures_wedding	EmpathFeatures_domestic_work	'
# 'EmpathFeatures_sleep	EmpathFeatures_medical_emergency	'
# 'EmpathFeatures_cold	EmpathFeatures_hate	EmpathFeatures_cheerfulness	EmpathFeatures_aggression'
# '	EmpathFeatures_occupation	EmpathFeatures_envy	EmpathFeatures_anticipation	EmpathFeatures_family	'
# 'EmpathFeatures_vacation	EmpathFeatures_crime	EmpathFeatures_attractive	EmpathFeatures_masculine	'
# 'EmpathFeatures_prison	EmpathFeatures_health	EmpathFeatures_pride	EmpathFeatures_dispute	EmpathFeatures_nervousness	'
# 'EmpathFeatures_government	EmpathFeatures_weakness	EmpathFeatures_horror	EmpathFeatures_swearing_terms	'
# 'EmpathFeatures_leisure	EmpathFeatures_suffering	EmpathFeatures_royalty	EmpathFeatures_wealthy	'
# 'EmpathFeatures_tourism	EmpathFeatures_furniture	EmpathFeatures_school	EmpathFeatures_magic	'
# 'EmpathFeatures_beach	EmpathFeatures_journalism	EmpathFeatures_morning	EmpathFeatures_banking	'
# 'EmpathFeatures_social_media	EmpathFeatures_exercise	EmpathFeatures_night	EmpathFeatures_kill	'
# 'EmpathFeatures_blue_collar_job	EmpathFeatures_art	EmpathFeatures_ridicule	EmpathFeatures_play'
# '	EmpathFeatures_computer	EmpathFeatures_college	EmpathFeatures_optimism	EmpathFeatures_stealing	'
# 'EmpathFeatures_real_estate	EmpathFeatures_home	EmpathFeatures_divine	EmpathFeatures_sexual	'
# 'EmpathFeatures_fear	EmpathFeatures_irritability	EmpathFeatures_superhero	EmpathFeatures_business	'
# 'EmpathFeatures_driving	EmpathFeatures_pet	EmpathFeatures_childish	EmpathFeatures_cooking	'
# 'EmpathFeatures_exasperation	EmpathFeatures_religion	EmpathFeatures_hipster	EmpathFeatures_internet	'
# 'EmpathFeatures_surprise	EmpathFeatures_reading	EmpathFeatures_worship	EmpathFeatures_leader	'
# 'EmpathFeatures_independence	EmpathFeatures_movement	EmpathFeatures_body	EmpathFeatures_noise	'
# 'EmpathFeatures_eating	EmpathFeatures_medieval	EmpathFeatures_zest	EmpathFeatures_confusion	EmpathFeatures_water	'
# 'EmpathFeatures_sports	EmpathFeatures_death	EmpathFeatures_healing	EmpathFeatures_legend	EmpathFeatures_heroic	'
# 'EmpathFeatures_celebration	EmpathFeatures_restaurant	EmpathFeatures_violence	EmpathFeatures_programming	'
# 'EmpathFeatures_dominant_heirarchical	EmpathFeatures_military	EmpathFeatures_neglect	EmpathFeatures_swimming	'
# 'EmpathFeatures_exotic	EmpathFeatures_love	EmpathFeatures_hiking	EmpathFeatures_communication	'
# 'EmpathFeatures_hearing	EmpathFeatures_order	EmpathFeatures_sympathy	EmpathFeatures_hygiene	'
# 'EmpathFeatures_weather	EmpathFeatures_anonymity	'
# 'EmpathFeatures_trust	EmpathFeatures_ancient	EmpathFeatures_deception	EmpathFeatures_fabric	'
# 'EmpathFeatures_air_travel	EmpathFeatures_fight	EmpathFeatures_dominant_personality	EmpathFeatures_music	'
# 'EmpathFeatures_vehicle	EmpathFeatures_politeness	EmpathFeatures_toy	EmpathFeatures_farming'
# '	EmpathFeatures_meeting	EmpathFeatures_war	EmpathFeatures_speaking	EmpathFeatures_listen	'
# 'EmpathFeatures_urban	EmpathFeatures_shopping	EmpathFeatures_disgust	EmpathFeatures_fire	EmpathFeatures_tool'
# '	EmpathFeatures_phone	EmpathFeatures_gain	EmpathFeatures_sound	EmpathFeatures_injury	'
# 'EmpathFeatures_sailing	EmpathFeatures_rage	EmpathFeatures_science	EmpathFeatures_work	EmpathFeatures_appearance'
# '	EmpathFeatures_valuable	EmpathFeatures_warmth	EmpathFeatures_youth	EmpathFeatures_sadness	EmpathFeatures_fun	'
# 'EmpathFeatures_emotional	EmpathFeatures_joy	EmpathFeatures_affection	EmpathFeatures_traveling	'
# 'EmpathFeatures_fashion	EmpathFeatures_ugliness	EmpathFeatures_lust	EmpathFeatures_shame	'
# 'EmpathFeatures_torment	EmpathFeatures_economics	EmpathFeatures_anger	EmpathFeatures_politics'
# '	EmpathFeatures_ship	EmpathFeatures_clothing	EmpathFeatures_car	EmpathFeatures_strength	EmpathFeatures_technology'
# '	EmpathFeatures_breaking	EmpathFeatures_shape_and_size	EmpathFeatures_power	'
# 'EmpathFeatures_white_collar_job	EmpathFeatures_animal	EmpathFeatures_party	'
# 'EmpathFeatures_terrorism	EmpathFeatures_smell	EmpathFeatures_disappointment	'
# 'EmpathFeatures_poor	EmpathFeatures_plant	EmpathFeatures_pain	EmpathFeatures_beauty'
# '	EmpathFeatures_timidity	EmpathFeatures_philosophy	EmpathFeatures_negotiate	'
# 'EmpathFeatures_negative_emotion	EmpathFeatures_cleaning	EmpathFeatures_messaging	'
# 'EmpathFeatures_competing	EmpathFeatures_law	EmpathFeatures_friends	EmpathFeatures_payment	'
# 'EmpathFeatures_achievement	EmpathFeatures_alcohol	EmpathFeatures_liquid	EmpathFeatures_feminine'
# '	EmpathFeatures_weapon	EmpathFeatures_children	EmpathFeatures_monster	EmpathFeatures_ocean	'
# 'EmpathFeatures_giving	EmpathFeatures_contentment	EmpathFeatures_writing	EmpathFeatures_rural	'
# 'EmpathFeatures_positive_emotion	EmpathFeatures_musical	'
# 'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_anger_avgLexVal	'
# 'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_anticipation_avgLexVal	EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_disgust_avgLexVal	'
# 'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_fear_avgLexVal	EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_joy_avgLexVal'
# '	EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_negative_avgLexVal	EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_positive_avgLexVal	'
# 'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_sadness_avgLexVal	EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_surprise_avgLexVal	'
# 'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_trust_avgLexVal	EmotionLex_NRC-VAD-Lexicon_valence_avgLexVal	'
# 'EmotionLex_NRC-VAD-Lexicon_arousal_avgLexVal	EmotionLex_NRC-VAD-Lexicon_domination_avgLexVal	'
# 'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_anger_avgLexVal	'
# 'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_anticipation_avgLexVal	'
# 'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_disgust_avgLexVal	EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_fear_avgLexVal	'
# 'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_joy_avgLexVal	EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_sadness_avgLexVal'
# '	EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_surprise_avgLexVal	EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_trust_avgLexVal')


def reduce_features():
    train_path = 'dataset/call_me_sexist_sexism_detection/CallMeSexistDataset_train_features.csv'
    test_path = 'dataset/call_me_sexist_sexism_detection/CallMeSexistDataset_test_features.csv'

    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)

    tr = tr[[
        "Wellformedness_LABEL_0",
        "ParrotAdequacy_contradiction",
        "GibberishDetector_clean",
        "TextEmotion_anger",
        "polarity",
        "subjectivity",
        "LIWCFeatures_shehe",
        "LIWCFeatures_negate",
        "LIWCFeatures_anger",
        "LIWCFeatures_negemo",
        "LIWCFeatures_sexual",
        "LIWCFeatures_swear",
        "EmpathFeatures_hate",
        "EmpathFeatures_masculine",
        "EmpathFeatures_nervousness",
        "label"]]
    te = te[[
        "Wellformedness_LABEL_0",
        "ParrotAdequacy_contradiction",
        "GibberishDetector_clean",
        "TextEmotion_anger",
        "polarity",
        "subjectivity",
        "LIWCFeatures_shehe",
        "LIWCFeatures_negate",
        "LIWCFeatures_anger",
        "LIWCFeatures_negemo",
        "LIWCFeatures_sexual",
        "LIWCFeatures_swear",
        "EmpathFeatures_hate",
        "EmpathFeatures_masculine",
        "EmpathFeatures_nervousness",
        "label"]]

    tr.to_csv('dataset/call_me_sexist_sexism_detection/CallMeSexistDataset_train_features_reduced.csv', index=False)
    te.to_csv('dataset/call_me_sexist_sexism_detection/CallMeSexistDataset_test_features_reduced.csv', index=False)


def pyfume_classification():
    # NOTE: takes a ton of time
    # Set the path to the data and choose the number of clusters
    train_path = 'dataset/call_me_sexist_sexism_detection/CallMeSexistDataset_train_features_reduced.csv'
    test_path = 'dataset/call_me_sexist_sexism_detection/CallMeSexistDataset_test_features_reduced.csv'
    nr_clus = 2

    # Load and normalize the data using min-max normalization
    train_dl = DataLoader(train_path, normalize='minmax')
    test_dl = DataLoader(test_path, normalize='minmax')
    variable_names = train_dl.variable_names

    # Split the data using the hold-out method in a training (default: 75%)
    # and test set (default: 25%).
    x_train, y_train, x_test, y_test = train_dl.dataX, train_dl.dataY, test_dl.dataX, test_dl.dataY

    # Select features relevant to the problem
    fs = FeatureSelector(dataX=x_train, dataY=y_train, nr_clus=nr_clus, variable_names=variable_names)
    selected_feature_indices, variable_names = fs.wrapper(fs_number_of_folds=11)

    # Adapt the training and test input data after feature selection
    x_train = x_train[:, selected_feature_indices]
    x_test = x_test[:, selected_feature_indices]

    # Cluster the training data (in input-output space) using FCM with default settings
    cl = Clusterer(x_train=x_train, y_train=y_train, nr_clus=nr_clus)
    cluster_centers, partition_matrix, _ = cl.cluster(method="fcm")

    # Estimate the membership funtions of the system (default: mf_shape = gaussian)
    ae = AntecedentEstimator(x_train=x_train, partition_matrix=partition_matrix)
    antecedent_parameters = ae.determineMF()

    # Calculate the firing strength of each rule for each data instance
    fsc = FireStrengthCalculator(antecedent_parameters=antecedent_parameters, nr_clus=nr_clus,
                                 variable_names=variable_names)
    firing_strengths = fsc.calculate_fire_strength(data=x_train)

    # Estimate the parameters of the consequent functions
    ce = ConsequentEstimator(x_train=x_train, y_train=y_train, firing_strengths=firing_strengths)
    consequent_parameters = ce.suglms()

    # Build a first-order Takagi-Sugeno model using Simpful. Specify the optional
    # 'extreme_values' argument to specify the universe of discourse of the input
    # variables if you which to use Simpful's membership function plot functionalities.
    simpbuilder = SugenoFISBuilder(antecedent_sets=antecedent_parameters,
                                   consequent_parameters=consequent_parameters,
                                   variable_names=variable_names)
    model = simpbuilder.get_model()

    # Calculate the mean squared error (MSE) of the model using the test data set
    test = SugenoFISTester(model=model, test_data=x_test, variable_names=variable_names, golden_standard=y_test)

    # MSE = test.calculate_MSE()
    ACC = test.calculate_accuracy()
    (f1_score, precision, recall) = test.calculate_f1_precision_recall()
    test.calculate_performance()

    # print('The mean squared error of the created model is', MSE)
    print('The accuracy of the created model is', ACC)
    print(f'f1_score: {f1_score}, precision: {precision}, recall: {recall}')


if __name__ == "__main__":
    # reduce_features()
    pyfume_classification()
