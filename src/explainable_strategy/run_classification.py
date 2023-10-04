from pathlib import Path

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from src.ami2020.dataset import train_val_test, compute_metrics, task_b_eval
from src.ami2020.simple_model.pipeline import naive_classifier
from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml

classifier_type = RidgeClassifier

TEAM_NAME = "myTeam"


def create_ami_submission(predictions: pd.DataFrame | pd.Series, task_type: str, data_type: str, run_type: str,
                          run_id: str, team_name: str) -> str:
    f = f"results/{team_name}.{task_type.upper()}.{data_type}.{run_type}.{run_id}"
    Path(f).parent.mkdir(exist_ok=True, parents=True)

    predictions.to_csv(f, header=False, sep="\t")
    return f


if __name__ == "__main__":
    train_config: dict = load_yaml("src/nlp/params/config.yml")
    clf_params = train_config[classifier_type.__name__]
    synthetic_add: bool = train_config["add_synthetic"]
    task: str = train_config["task"]

    print("*** Predicting misogyny ")
    data = train_val_test(target="M", add_synthetic_train=synthetic_add)
    m_pred, pipe_m = naive_classifier(classifier_type(**clf_params), data, return_pipe=True)
    m_f1 = compute_metrics(m_pred, data["test"]["y"], classifier_type.__name__)["f1"]

    match task:
        case "B":
            print("*** Task B ")
            if not synthetic_add:
                test_synt: dict = train_val_test(target="M", add_synthetic_train=True)["test_synt"]
            else:
                test_synt = data["test_synt"]
            m_synt_pred = pipe_m.predict(test_synt["x"])

            df_pred = pd.Series(m_pred, index=pd.Index(data["test"]["ids"], dtype=str))
            df_pred_synt = pd.Series(m_synt_pred, index=pd.Index(test_synt["ids"], dtype=str))

            # Preparing data for evaluation
            df_pred = df_pred.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
            df_pred_synt = df_pred_synt.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
            # Read gold test set
            task_b_eval(data, df_pred, df_pred_synt)

            # USE BELOW to create submission files for the competition
            # fr = create_ami_submission(df_pred, task_type="B", data_type="r", run_type="c", run_id="run1", team_name=TEAM_NAME)
            # fs = create_ami_submission(df_pred_synt, task_type="B", data_type="s", run_type="c", run_id="run1", team_name=TEAM_NAME)
            # # create a ZipFile object in write mode
            # with zipfile.ZipFile(f"results/{TEAM_NAME}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
            #     # add the file to the zip file
            #     zipf.write(fr)
            #     zipf.write(fs)

        case "A":
            # Get rows with predicted misogyny
            # misogyny_indexes, misogyny_ids = zip(
            #     *[(i, pid) for i, (p, pid) in enumerate(zip(m_pred, data["test"]["ids"])) if p > 0])
            # non_misogyny_ids: set[int] = set(data["test"]["ids"]) - set(misogyny_ids)

            print("*** Task A")
            print("*** Predicting aggressiveness ")
            data = train_val_test(target="A")

            # data_aggressiveness = {
            #     k: {
            #         "x": list(itemgetter(*misogyny_indexes)(v["x"])),
            #         "y": list(itemgetter(*misogyny_indexes)(v["y"])),
            #         "ids": list(itemgetter(*misogyny_indexes)(v["ids"]))
            #     } for k, v in data.items()
            # }

            a_pred = naive_classifier(classifier_type(**clf_params), data)
            a_true = data["test"]["y"]
            a_ids = data["test"]["ids"]
            # a_pred = [0] * len(m_pred)

            # a_pred = np.concatenate([a_pred, np.array([0] * len(non_misogyny_ids))])
            # a_ids = data_aggressiveness["test"]["ids"] + list(non_misogyny_ids)
            # a_true = data_aggressiveness["test"]["y"] + ([0] * len(non_misogyny_ids))

            a_f1 = compute_metrics(a_pred, a_true, classifier_type.__name__)["f1"]

            a_score = (m_f1 + a_f1) / 2
            print(f"\n*** Task A score: {a_score:.5f} ***")

            # df_pred = pd.DataFrame([m_pred, a_pred], columns=a_ids).T
            # fo = create_ami_submission(df_pred, task_type="A", data_type="r", run_type="c", run_id="run1", team_name=TEAM_NAME)
            # with zipfile.ZipFile(f"results/{TEAM_NAME}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
            #     zipf.write(fo)

        case _:
            raise ValueError(f"Unsupported task '{task}'. Only 'A' or 'B' are possible values.")

    # Best task A score: 0.707
