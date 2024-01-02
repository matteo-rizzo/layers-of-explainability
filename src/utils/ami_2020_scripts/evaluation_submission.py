import argparse
import zipfile

import pandas as pd

from src.utils.ami_2020_scripts import evaluation_utils


# import evaluation_utils


def read_input(submission_path, task):
    raw_data_predicted = {}
    synt_data_predicted = {}
    raw_column_names, synt_column_names = '', ''
    converters_raw = {}

    if (submission_path.endswith(".zip")):
        unzipped = zipfile.ZipFile(submission_path, "r")
    else:
        raise Exception("Exception: submission file is not a ZIP file.")
    if not ((task.lower() == "a") or (task.lower() == "b")):
        raise Exception('Exception: task should be either "a" or "b".')
    elif ((task.lower() == "a")):
        raw_column_names = ['id', 'misogynous', 'aggressiveness']
        converters_raw = {0: str, 1: int, 2: int}
    elif ((task.lower() == "b")):
        raw_column_names = ['id', 'misogynous']
        converters_raw = {0: str, 1: int}
        synt_column_names = ['id', 'misogynous']

    run_filelist = [f for f in unzipped.namelist() if
                    ("run" in f) and ("." + task + "." in f.lower()) and (not ('__MACOSX' in f))]

    for runs_admitted in ['run1', 'run2', 'run3']:
        if (len([f for f in run_filelist if (runs_admitted in f)]) == 0):
            continue
        raw_data_path = [f for f in run_filelist if
                         (runs_admitted in f) and ("." + task + "." in f.lower()) and (".r." in f.lower())]
        if (len(raw_data_path) == 1):
            try:
                raw_data_predicted[runs_admitted] = pd.read_csv(unzipped.open(raw_data_path[0]), sep="\t", header=None,
                                                                index_col=None, names=raw_column_names,
                                                                converters=converters_raw)
            except:
                raise Exception('Exception: error in raw prediction file format.')
        elif (len(raw_data_path) > 1):
            raise Exception('Exception: too many raw prediction files for ' + runs_admitted + ' (check file names).')
        elif (len(raw_data_path) == 0):
            raise Exception('Exception: no raw prediction files for ' + runs_admitted + ' (check file names).')

        if ((task.lower() == "b")):
            synt_data_path = [f for f in run_filelist if
                              (runs_admitted in f) and ("." + task + "." in f.lower()) and (".s." in f.lower())]
            if (len(raw_data_path) == 1):
                try:
                    synt_data_predicted[runs_admitted] = pd.read_csv(unzipped.open(synt_data_path[0]), sep="\t",
                                                                     header=None, index_col=None,
                                                                     names=synt_column_names,
                                                                     converters={0: str, 1: int})
                except:
                    raise Exception('Exception: error in synt prediction file format.')
            else:
                raise Exception('Exception: too many synt prediction files in the same run (check file names).')

    return raw_data_predicted, synt_data_predicted


def read_gold(gold_path_raw, gold_path_synt, identityterms_path, task):
    synt_data_gold = None
    identityterms = None

    raw_data_gold = pd.read_csv(gold_path_raw, sep="\t", converters={0: str, 1: str, 2: int, 3: int})
    if (task == 'b'):
        if (gold_path_synt == ""):
            raise Exception('Missing path for gold synt data.')
        if (identityterms_path == ""):
            raise Exception('Missing path for identity terms data.')
        synt_data_gold = pd.read_csv(gold_path_synt, sep="\t", converters={0: str, 1: str, 2: int})
        with open(identityterms_path, "r") as f:
            identityterms = f.read().splitlines()

    return raw_data_gold, synt_data_gold, identityterms


def evaluate_task_a_singlefile(raw_data_predicted_file, raw_data_gold):
    evaluation_utils.check_submission_consistency(raw_data_gold, raw_data_predicted_file, ['misogynous'], 'raw')

    ## Compute macro F1
    results_raw = pd.merge(raw_data_gold, raw_data_predicted_file, on="id", suffixes=("", "_pred"))
    evaluation_utils.check_merge_length(results_raw, raw_data_gold, 'raw')
    macro_f1, f1_levels = evaluation_utils.get_metric_subtask_a(results_raw)

    print("taskA_fscore_misogynous: {0}\n".format(f1_levels["misogynous"]))
    print("taskA_fscore_aggressiveness: {0}\n".format(f1_levels["aggressiveness"]))

    return macro_f1


def evaluate_task_b_singlefile(raw_data_predicted_file, synt_data_predicted_file, raw_data_gold, synt_data_gold,
                               identityterms):
    evaluation_utils.check_submission_consistency(synt_data_gold, synt_data_predicted_file, ['misogynous'], 'synt')
    evaluation_utils.check_submission_consistency(raw_data_gold, raw_data_predicted_file, ['misogynous'], 'raw')

    # Compute bias metric
    results_synt = pd.merge(synt_data_gold, synt_data_predicted_file, on="id", suffixes=("", "_pred"))
    evaluation_utils.check_merge_length(results_synt, synt_data_gold, 'raw')
    evaluation_utils.add_subgroup_columns_from_text(results_synt, 'text', identityterms)
    bias_metrics = evaluation_utils.compute_bias_metrics_for_model(results_synt, identityterms, 'misogynous_pred',
                                                                   'misogynous')

    # Compute AUC
    results_raw = pd.merge(raw_data_gold, raw_data_predicted_file, on="id", suffixes=("", "_pred"))
    evaluation_utils.check_merge_length(results_raw, raw_data_gold, 'synt')
    overall_auc = evaluation_utils.calculate_overall_auc(results_raw, 'misogynous_pred')

    # Compute final metric

    return evaluation_utils.get_final_metric(bias_metrics, overall_auc, model_name='misogynous_pred')


def evaluate_task_b(raw_data_predicted, synt_data_predicted, raw_data_gold, synt_data_gold, identityterms, output_path):
    print("Starting evaluation Subtask B")
    f = open(output_path, "w")
    f.write("run_subtaskB\tscore\n")
    for k in raw_data_predicted.keys():
        if not (k in synt_data_predicted):
            raise Exception('Missing synt prediction data.')
        print('*' * 20, k)

        metric = evaluate_task_b_singlefile(raw_data_predicted[k], synt_data_predicted[k], raw_data_gold,
                                            synt_data_gold, identityterms)
        f.write(k + "\t" + str(metric) + "\n")
    f.close()
    print("Evaluation for Subtask B completed and saved at:", output_path)


def evaluate_task_a(raw_data_predicted, raw_data_gold, output_path):
    print("Starting evaluation Subtask A")
    f = open(output_path, "w")
    f.write("run_subtaskA\tscore\n")
    for k in raw_data_predicted.keys():
        print('*' * 20, k)
        metric = evaluate_task_a_singlefile(raw_data_predicted[k], raw_data_gold)
        f.write(k + "\t" + str(metric) + "\n")
    f.close()
    print("Evaluation for Subtask A completed and saved at:", output_path)


def evaluate(raw_data_predicted, synt_data_predicted, raw_data_gold, synt_data_gold, identityterms, output_path, task):
    if (task.lower() == "a"):
        evaluate_task_a(raw_data_predicted, raw_data_gold, output_path)
    elif (task.lower() == "b"):
        evaluate_task_b(raw_data_predicted, synt_data_predicted, raw_data_gold, synt_data_gold, identityterms,
                        output_path)
    else:
        raise Exception('Task should be either "a" or "b"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVALITA AMI 2020 - Evaluation script.')
    parser.add_argument('--submission_path', type=str, required=True,
                        help='path of the submission file (ZIP format)')
    parser.add_argument('--gold_path_raw', type=str, required=True,
                        help='path of gold raw tsv file')
    parser.add_argument('--gold_path_synt', type=str, required=False, default="",
                        help='path of gold synt tsv file')
    parser.add_argument('--identityterms_path', type=str, required=False, default="",
                        help='path of identity terms tsv file')
    parser.add_argument('--task', type=str, required=True, choices=['a', 'b'],
                        help='task you want to evaluate ("a" or "b")')
    parser.add_argument('--output_path', type=str, required=False, default="result.tsv",
                        help='path of output result file')

    args = parser.parse_args()

    raw_data_predicted, synt_data_predicted = read_input(args.submission_path, args.task)
    raw_data_gold, synt_data_gold, identityterms = read_gold(args.gold_path_raw, args.gold_path_synt,
                                                             args.identityterms_path, args.task)

    evaluate(raw_data_predicted, synt_data_predicted, raw_data_gold, synt_data_gold, identityterms, args.output_path,
             args.task)
