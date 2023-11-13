import seaborn as sns
from pyfume import *
from simpful import *

('6.208649e-01*offensive'
 '+3.345043e-01*female'
 '+1.752369e-01*TextEmotion_anger'
 '+-5.237787e-02*subjectivity'
 '+-4.577606e-02*GibberishDetector_clean'
 '+9.929635e-02*sarcasm'
 '+-8.772847e-02*word_count'
 '+-1.158139e-02*irony'
 '+-1.000347e-01')

('5.898416e-01*offensive'
 '+3.326967e-01*female'
 '+1.871315e-01*TextEmotion_anger'
 '+-5.280500e-02*subjectivity'
 '+-4.455483e-02*GibberishDetector_clean'
 '+9.747353e-02*sarcasm'
 '+-8.322537e-02*word_count'
 '+-1.247330e-02*irony'
 '+-8.274593e-02')

def create_system():
    sns.set_style("darkgrid")
    FS = FuzzySystem(show_banner=False)
    RULE1 = "IF (offensive IS cluster1) AND (female IS cluster1) AND (TextEmotion_anger IS cluster1) AND (subjectivity IS cluster1) AND (GibberishDetector_clean IS cluster1) AND (sarcasm IS cluster1) AND (word_count IS cluster1) AND (irony IS cluster1) THEN (OUTPUT IS fun1)"
    RULE2 = ("IF (offensive IS cluster2) AND (female IS cluster2) AND (TextEmotion_anger IS cluster2) AND ("
             "subjectivity IS cluster2) AND (GibberishDetector_clean IS cluster2) AND (sarcasm IS cluster2) AND ("
             "word_count IS cluster2) AND (irony IS cluster2) THEN (OUTPUT IS fun2)")
    FS.add_rules([RULE1, RULE2])
    u_o_d = [0, 1]

    FS.set_output_function('fun1',
                           '6.208649e-01*offensive+3.345043e-01*female+1.752369e-01*TextEmotion_anger+-5.237787e-02*subjectivity+-4.577606e-02*GibberishDetector_clean+9.929635e-02*sarcasm+-8.772847e-02*word_count+-1.158139e-02*irony+-1.000347e-01')
    FS.set_output_function('fun2',
                           '5.898416e-01*offensive+3.326967e-01*female+1.871315e-01*TextEmotion_anger+-5.280500e-02*subjectivity+-4.455483e-02*GibberishDetector_clean+9.747353e-02*sarcasm+-8.322537e-02*word_count+-1.247330e-02*irony+-8.274593e-02')

    plot = False
    scale = 'linear'
    i = 1
    FS_1 = FuzzySet(function=Gaussian_MF(1.077891, 1.030371), term='cluster1')
    FS_2 = FuzzySet(function=Gaussian_MF(0.420197, 0.753033), term='cluster2')
    MF_offensive = LinguisticVariable([FS_1, FS_2], concept='offensive', universe_of_discourse=u_o_d)
    if plot:
        MF_offensive.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('offensive', MF_offensive)

    FS_3 = FuzzySet(function=Gaussian_MF(0.726128, 0.919364), term='cluster1')
    FS_4 = FuzzySet(function=Gaussian_MF(0.288933, 0.883041), term='cluster2')
    MF_female = LinguisticVariable([FS_3, FS_4], concept='female', universe_of_discourse=u_o_d)
    if plot:
        MF_female.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('female', MF_female)

    FS_5 = FuzzySet(function=Gaussian_MF(0.354772, 0.986378), term='cluster1')
    FS_6 = FuzzySet(function=Gaussian_MF(-0.267492, 1.234684), term='cluster2')
    MF_TextEmotion_anger = LinguisticVariable([FS_5, FS_6], concept='TextEmotion_anger', universe_of_discourse=u_o_d)
    if plot:
        MF_TextEmotion_anger.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('TextEmotion_anger', MF_TextEmotion_anger)

    FS_7 = FuzzySet(function=Gaussian_MF(0.492431, 0.876909), term='cluster1')
    FS_8 = FuzzySet(function=Gaussian_MF(0.377121, 0.883985), term='cluster2')
    MF_subjectivity = LinguisticVariable([FS_7, FS_8], concept='subjectivity', universe_of_discourse=u_o_d)
    if plot:
        MF_subjectivity.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('subjectivity', MF_subjectivity)

    FS_9 = FuzzySet(function=Gaussian_MF(0.520734, 0.898764), term='cluster1')
    FS_10 = FuzzySet(function=Gaussian_MF(0.617659, 0.946021), term='cluster2')
    MF_GibberishDetector_clean = LinguisticVariable([FS_9, FS_10], concept='GibberishDetector_clean',
                                                    universe_of_discourse=u_o_d)
    if plot:
        MF_GibberishDetector_clean.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('GibberishDetector_clean', MF_GibberishDetector_clean)

    FS_11 = FuzzySet(function=Gaussian_MF(-0.217270, 1.603306), term='cluster1')
    FS_12 = FuzzySet(function=Gaussian_MF(-0.320223, 1.481392), term='cluster2')
    MF_sarcasm = LinguisticVariable([FS_11, FS_12], concept='sarcasm', universe_of_discourse=u_o_d)
    if plot:
        MF_sarcasm.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('sarcasm', MF_sarcasm)

    FS_13 = FuzzySet(function=Gaussian_MF(0.453146, 0.673907), term='cluster1')
    FS_14 = FuzzySet(function=Gaussian_MF(0.377635, 0.558134), term='cluster2')
    MF_word_count = LinguisticVariable([FS_13, FS_14], concept='word_count', universe_of_discourse=[0, 30])
    if plot:
        MF_word_count.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('word_count', MF_word_count)

    FS_15 = FuzzySet(function=Gaussian_MF(0.350291, 0.909257), term='cluster1')
    FS_16 = FuzzySet(function=Gaussian_MF(0.344420, 0.836801), term='cluster2')
    MF_irony = LinguisticVariable([FS_15, FS_16], concept='irony', universe_of_discourse=u_o_d)
    if plot:
        MF_irony.plot(outputfile=f"{i}", xscale=scale)
        i += 1
    FS.add_linguistic_variable('irony', MF_irony)

    return FS


if __name__ == "__main__":
    fis = create_system()
    te_path = 'dataset/ami2018_misogyny_detection/pyfuming/test.csv'
    test_dl = DataLoader(te_path, normalize='minmax')

    # Calculate the mean squared error (MSE) of the model using the test data set
    test = SugenoFISTester(model=fis, test_data=test_dl.dataX, variable_names=test_dl.variable_names,
                           golden_standard=test_dl.dataY)
    MSE = test.calculate_MSE()
    test.calculate_AUC(show_plot=True)
    print('The mean squared error of the created model is', MSE)
