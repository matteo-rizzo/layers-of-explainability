import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

SUBGROUP = 'subgroup'
SUBSET_SIZE = 'subset_size'
SUBGROUP_AUC = 'subgroup_auc'
NEGATIVE_CROSS_AUC = 'bpsn_auc'
POSITIVE_CROSS_AUC = 'bnsp_auc'


def add_subgroup_columns_from_text(df, text_column, subgroups):
    """Adds a boolean column for each subgroup to the data frame.

      New column contains True if the text contains that subgroup term.
      """
    for term in subgroups:
        # pylint: disable=cell-var-from-loop
        df[term] = df[text_column].apply(lambda x: bool(
            re.search('\\b' + term + '\\b', str(x), flags=re.UNICODE | re.IGNORECASE)))


def compute_bias_metrics_for_subgroup_and_model(dataset,
                                                subgroup,
                                                model,
                                                label_col):
    """Computes per-subgroup metrics for one model and subgroup."""
    record = {
        SUBGROUP: subgroup,
        SUBSET_SIZE: len(dataset[dataset[subgroup]])
    }
    record[column_name(model, SUBGROUP_AUC)] = compute_subgroup_auc(
        dataset, subgroup, label_col, model)
    record[column_name(model, NEGATIVE_CROSS_AUC)] = compute_negative_cross_auc(
        dataset, subgroup, label_col, model)
    record[column_name(model, POSITIVE_CROSS_AUC)] = compute_positive_cross_auc(
        dataset, subgroup, label_col, model)
    return record


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        subgroup_record = compute_bias_metrics_for_subgroup_and_model(
            dataset, subgroup, model, label_col)
        records.append(subgroup_record)
    return pd.DataFrame(records)


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def check_file(path, correct_number_of_columns):
    f = open(path, 'r')
    first_line = f.readlines()[0].split("\t")
    f.close()
    # print(first_line)
    if (len(first_line) != correct_number_of_columns):
        sys.exit('Column format problem.')

    if (isfloat(first_line[0])):
        has_header = 0
    else:
        has_header = 1
    return has_header


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        return np.nan


def column_name(model, metric):
    return model + '_' + metric


def compute_negative_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    # examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    examples = pd.concat([subgroup_negative_examples, non_subgroup_positive_examples])
    return compute_auc(examples[label], examples[model_name])


def compute_positive_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    # examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    examples = pd.concat([subgroup_positive_examples, non_subgroup_negative_examples])
    return compute_auc(examples[label], examples[model_name])


def calculate_overall_auc(df, model_name):
    true_labels = df['misogynous']
    predicted_labels = df[model_name]
    return roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc_test, model_name):
    bias_score = np.average([
        bias_df[model_name + '_subgroup_auc'],
        bias_df[model_name + '_bpsn_auc'],
        bias_df[model_name + '_bnsp_auc']
    ])
    print(f"Bias Score = {bias_score:.5f}")
    print(f"AUC Score = {overall_auc_test:.5f}")
    return np.mean([overall_auc_test, bias_score])


def get_metric_subtask_a(data):
    levels = ["misogynous", "aggressiveness"]
    acc_levels = dict.fromkeys(levels)
    p_levels = dict.fromkeys(levels)
    r_levels = dict.fromkeys(levels)
    f1_levels = dict.fromkeys(levels)
    for l in levels:
        acc_levels[l] = accuracy_score(data[l], data[l + "_pred"])
        p_levels[l], r_levels[l], f1_levels[l], _ = precision_recall_fscore_support(data[l], data[l + "_pred"],
                                                                                    average="macro")
    macro_f1 = np.mean(list(f1_levels.values()))
    return macro_f1, f1_levels


def check_submission_consistency(gold_df, submission_df, levels, type_data):
    # Check length files
    if (len(gold_df) != len(submission_df)):
        sys.exit('Prediction and gold ' + type_data + ' data have different number of lines.')

    # Check predicted classes
    for c in levels:
        gt_class = list(gold_df[c].value_counts().keys())
        if not (submission_df[c].isin(gt_class).all()):
            sys.exit("Wrong value in " + c + " prediction column of " + type_data + " data.")


def check_merge_length(ground_truth, predicted, type_data):
    # Check length files
    if (len(ground_truth) != len(predicted)):
        sys.exit('Prediction and gold ' + type_data + ' data have different number of lines or different IDs.')
