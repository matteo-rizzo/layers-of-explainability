import pandas as pd
from pyfume import *
from simpful import *


def madness():
    FS = FuzzySystem(show_banner=False)

    # -------------------------------------------------------------------------------------------
    def sarcasm_function(x):
        return x

    S_1 = FuzzySet(function=lambda x: 1 - sarcasm_function(x), term="low")
    S_2 = FuzzySet(function=sarcasm_function, term="high")
    FS.add_linguistic_variable("Sarcasm", LinguisticVariable([S_1, S_2], concept="Sarcasm degree",
                                                             universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def irony_function(x):
        return x

    I_1 = FuzzySet(function=lambda x: 1 - irony_function(x), term="low")
    I_2 = FuzzySet(function=irony_function, term="high")
    FS.add_linguistic_variable("Irony", LinguisticVariable([I_1, I_2], concept="Irony degree",
                                                           universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def offensiveness_function(x):
        return x

    O_1 = FuzzySet(function=lambda x: 1 - offensiveness_function(x), term="low")
    O_2 = FuzzySet(function=offensiveness_function, term="high")
    FS.add_linguistic_variable("Offensiveness", LinguisticVariable([O_1, O_2], concept="Offensiveness degree",
                                                                   universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def gender_function(x):
        return x

    G_1 = FuzzySet(function=gender_function, term="female")
    G_2 = FuzzySet(function=lambda x: 1 - gender_function(x), term="male")
    FS.add_linguistic_variable("Gender", LinguisticVariable([G_1, G_2], concept="Gender",
                                                            universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def topic_function(x):
        return [0] * 19

    topics = ["arts_and_culture", "business_and_entrepeneurs", "celebrity_and_pop_culture", "diaries_and_daily_life",
              "family", "fashion_and_style", "film_tv_and_video", "fitness_and_health", "food_and_dining", "gaming",
              "learning_and_educational", "music", "news_and_social_concern", "other_hobbies", "relationships",
              "science_and_technology", "sports", "travel_and_adventure", "youth_and_student_life"]
    Ts = [FuzzySet(function=lambda x: topic_function(x)[i], term=topics[i]) for i in range(len(topics))]
    FS.add_linguistic_variable("Topic", LinguisticVariable(Ts, concept="Topic",
                                                           universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def gibberish_function(x):
        return 0, 0, 0, 0

    gibb_types = ["Noise", "Word Salad", "Mild Gibberish", "Clean"]
    Gibs = [FuzzySet(function=lambda x: gibberish_function(x)[i], term=gibb_types[i]) for i in range(len(gibb_types))]
    FS.add_linguistic_variable("Gibberish", LinguisticVariable(Gibs, concept="Gibberish degree",
                                                               universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def parrot_function(x):
        return 0, 0, 0

    parrot_types = ["Neutral", "Contradiction", "Entailment"]
    Parrs = [FuzzySet(function=lambda x: parrot_function(x)[i], term=parrot_types[i]) for i in range(len(parrot_types))]
    FS.add_linguistic_variable("Parrot", LinguisticVariable(Parrs, concept="Parrot degree",
                                                            universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def llm_detector(x):
        return 0

    CG_1 = FuzzySet(function=llm_detector, term="human")
    CG_2 = FuzzySet(function=lambda x: 1 - llm_detector(x), term="chat-GPT")
    FS.add_linguistic_variable("LLM_Detection", LinguisticVariable([CG_1, CG_2], concept="Human or LLM",
                                                                   universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def wellf_function(x):
        return 0

    WF_1 = FuzzySet(function=wellf_function, term="Well-formed")
    WF_2 = FuzzySet(function=lambda x: 1 - wellf_function(x), term="Not well-formed")
    FS.add_linguistic_variable("LLM_Detection", LinguisticVariable([WF_1, WF_2], concept="Wellformedness",
                                                                   universe_of_discourse=[0, 1]))

    # -------------------------------------------------------------------------------------------
    def evidence_function(x):
        return 0, 0, 0, 0, 0, 0, 0

    evidence_types = ["None", "Other", "Assumption", "Definition", "Testimony", "Statistics/Study", "Anecdote"]
    Es = [FuzzySet(function=lambda x: evidence_function(x)[i], term=evidence_types[i]) for i in
          range(len(evidence_types))]
    FS.add_linguistic_variable("Evidence", LinguisticVariable(Es, concept="Evidence types",
                                                              universe_of_discourse=[0, 1]))
    # -------------------------------------------------------------------------------------------
    print(FS.get_rules())


def reduce_dataset():
    tr_path = 'dataset/ami2018_misogyny_detection/big_feat/AMI2018Dataset_train_features.csv'
    te_path = 'dataset/ami2018_misogyny_detection/big_feat/AMI2018Dataset_test_features.csv'

    tr_l_p = 'dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv'
    te_l_p = 'dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv'

    a_x = pd.read_csv(tr_path)
    b_x = pd.read_csv(te_path)
    a_y = pd.read_table(tr_l_p)
    b_y = pd.read_table(te_l_p)

    a_x["irony"] = a_y["irony"]
    a_x["sarcasm"] = a_y["sarcasm"]
    a_x["offensive"] = a_y["offensive"]
    a_x["female"] = a_y["female"]
    a_x["label"] = a_y["label"]
    a_x.columns = [x.replace("/", "_") for x in a_x.columns]
    a_x.columns = [x.replace("&", "n") for x in a_x.columns]
    a_x.columns = [x.replace(" ", "") for x in a_x.columns]

    b_x["irony"] = b_y["irony"]
    b_x["sarcasm"] = b_y["sarcasm"]
    b_x["offensive"] = b_y["offensive"]
    b_x["female"] = b_y["female"]
    b_x["label"] = b_y["label"]
    b_x.columns = [x.replace("/", "_") for x in b_x.columns]
    b_x.columns = [x.replace("&", "n") for x in b_x.columns]
    a_x.columns = [x.replace(" ", "") for x in a_x.columns]

    a_x[["TextEmotion_anger", "polarity", "subjectivity", "word_count", "irony", "sarcasm", "offensive", "female",
         "num_errors", "Wellformedness_LABEL_0", "GibberishDetector_clean", "label"]].to_csv(
        'dataset/ami2018_misogyny_detection/pyfuming/train.csv', index=False)
    b_x[["TextEmotion_anger", "polarity", "subjectivity", "word_count", "irony", "sarcasm", "offensive", "female",
         "num_errors", "Wellformedness_LABEL_0", "GibberishDetector_clean", "label"]].to_csv(
        'dataset/ami2018_misogyny_detection/pyfuming/test.csv', index=False)
def pyfuming_and_chill():
    # Set the path to the data and choose the number of clusters

    tr_path = 'dataset/ami2018_misogyny_detection/pyfuming/train.csv'
    te_path = 'dataset/ami2018_misogyny_detection/pyfuming/test.csv'
    nr_clus = 3
    # Load and normalize the data using min-max normalization
    train_dl = DataLoader(tr_path, normalize='minmax')
    test_dl = DataLoader(te_path, normalize='minmax')
    variable_names = train_dl.variable_names
    x_train, y_train, x_test, y_test = train_dl.dataX, train_dl.dataY, test_dl.dataX, test_dl.dataY
    # Split the data using the hold-out method in a training (default: 75%)
    # and test set (default: 25%).
    # ds = DataSplitter()
    # x_train, y_train, x_test, y_test = ds.holdout(dataX=dl.dataX, dataY=dl.dataY)

    # Select features relevant to the problem
    fs = FeatureSelector(dataX=x_train, dataY=y_train, nr_clus=nr_clus, variable_names=variable_names)
    selected_feature_indices, variable_names = fs.wrapper()

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
    MSE = test.calculate_MSE()

    print('The mean squared error of the created model is', MSE)


# def main():
#     # A simple fuzzy inference system for the tipping problem
#     # Create a fuzzy system object
#     FS = FuzzySystem()
#
#     # Define fuzzy sets and linguistic variables
#     S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
#     S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
#     S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
#     FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality",
#                                                              universe_of_discourse=[0, 10]))
#
#     F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
#     F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
#     FS.add_linguistic_variable("Food",
#                                LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0, 10]))
#
#     # Define output fuzzy sets and linguistic variable
#     T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
#     T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
#     T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
#     FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 25]))
#
#     # Define fuzzy rules
#     R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
#     R2 = "IF (Service IS good) THEN (Tip IS average)"
#     R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
#     FS.add_rules([R1, R2, R3])
#
#     # Set antecedents values
#     FS.set_variable("Service", 4)
#     FS.set_variable("Food", 8)
#
#     # Perform Mamdani inference and print output
#     print(FS.Mamdani_inference(["Tip"]))


if __name__ == "__main__":
    reduce_dataset()
    pyfuming_and_chill()
