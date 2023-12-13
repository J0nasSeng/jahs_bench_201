import argparse
import functools
import logging
from pathlib import Path
from typing import Tuple, Optional

import joblib
import neps
import pandas as pd
import scipy.stats
from jahs_bench.surrogate import model, config as cfg
import os
from rtpt import RTPT

_log = logging.getLogger(__name__)

xgb_hp_space = {
    "max_depth": neps.IntegerParameter(
        lower=1, upper=15, default=6, default_confidence="low"
    ),
    "min_child_weight": neps.IntegerParameter(
        lower=1, upper=10, default=1, default_confidence="low"
    ),
    "colsample_bytree": neps.FloatParameter(
        lower=0., upper=1., log=False, default=1., default_confidence="low"
    ),
    "learning_rate": neps.FloatParameter(
        lower=0.001, upper=0.5, log=True, default=0.3, default_confidence="low"
    ),
    "colsample_bylevel":  neps.FloatParameter(
        lower=0., upper=1., log=False, default=1., default_confidence="low"
    ),
    "sigmoid_k": neps.FloatParameter(
        lower=0.01, upper=2., log=True, default=1., default_confidence="low"
    )
}

def train_surrogate(working_directory: Path, train_data: pd.DataFrame,
                    valid_data: pd.DataFrame, **config_dict):
    """

    :param working_directory:
    :param train_data: pandas DataFrame
        A pandas DataFrame object with 2-level MultiIndex columns, containing the
        training data. Level 0 should contain the columns "features" and "labels". Level
        1 can contain arbitrary columns.
    :param valid_data: pandas DataFrame
        A pandas DataFrame object with 2-level MultiIndex columns, containing the
        validation data. Level 0 should contain the columns "features" and "labels".
        Level 1 can contain arbitrary columns, but they should match those in
        `train_data`.
    :param config_dict: keyword-arguments
        These specify the various hyperparameters to be used for the model.
    :return:
    """

    xtrain = train_data["features"]
    ytrain = train_data["labels"]

    xvalid = valid_data["features"]
    yvalid = valid_data["labels"]

    _log.info(f"Preparing to train surrogate.")
    pipeline_config = cfg.default_pipeline_config.clone()
    xgb_params = config_dict.copy()
    sigmoid_k = xgb_params.pop("sigmoid_k", None)

    if sigmoid_k is not None:
        pipeline_config.defrost()
        pipeline_config.target_config.params.params[1].k = float(sigmoid_k)
        pipeline_config.freeze()

    surrogate = model.XGBSurrogate(hyperparams=xgb_params, use_gpu=True)

    _log.info("Training surrogate.")
    random_state = None
    scores = surrogate.fit(xtrain, ytrain, random_state=random_state,
                           pipeline_config=pipeline_config)
    _log.info(f"Trained surrogate has scores: {scores}")

    modeldir = working_directory / "xgb_model"
    _log.info(f"Saving trained surrogate to disk at {str(modeldir)}")
    modeldir.mkdir()
    surrogate.dump(modeldir)
    config_file = modeldir / "pipeline_config.pkl.gz"
    joblib.dump(pipeline_config, config_file, protocol=4)

    #_log.info(f"Generating validation scores.")
    #ypred = surrogate.predict(xvalid)
    #valid_score = scipy.stats.kendalltau(yvalid, ypred)

    #_log.info(f"Trained surrogate has validation score: {valid_score}")
    # Return negative KT correlation since NEPS minimizes the loss
    #return -valid_score.correlation

def load_data(datadir: Path, output: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _log.info("Loading training data.")
    train_data: pd.DataFrame = pd.read_pickle(datadir / "train_set.pkl.gz")
    _log.info(f"Loaded training data of shape {train_data.shape}")

    _log.info("Loading validation data.")
    valid_data: pd.DataFrame = pd.read_pickle(datadir / "valid_set.pkl.gz")
    _log.info(f"Loaded validation data of shape {valid_data.shape}")

    features = train_data["features"].columns
    labels = train_data["labels"].columns

    assert features.difference(valid_data["features"].columns).size == 0, \
        "Mismatch between the input features in the training and validation sets."
    assert labels.difference(valid_data["labels"].columns).size == 0, \
        "Mismatch between the output labels in the training and validation sets."

    if output is not None:
        assert output in labels, f"The chosen prediction label {output} is not present " \
                                 f"in the training data."
        selected_cols = train_data[["features"]].columns.tolist() + [("labels", output)]
        train_data = train_data.loc[:, selected_cols]
        valid_data = valid_data.loc[:, selected_cols]

    return train_data, valid_data

def train_surrogate_with_correct_config(working_directory: Path, datadir: Path, output: str,
                dataset: str, max_evaluations_total: int = 5,):
    """Due to the incomatibility of the pre-trained surrogates provided by jahs-bench, we have to retrain them.
    After this they are compatible with scikit-learn 1.1.3, hence we can use them for querying JAHS.

    Instead of performing HPO, we extracted the hyperparameters that were used to train the pre-trained surrogates
    provided by JAHS to ensure reproducibility. 

    Below we iterate over all datasets and metrics reported in JAHS and train a surrogate for each <dataset, metric>-pair.

    NOTE: This function replaces `perform_hpo` in the original JAHS codebase.

    Args:
        working_directory (Path): _description_
        datadir (Path): _description_
        output (str): _description_
        dataset (str): _description_
        max_evaluations_total (int, optional): _description_. Defaults to 5.
    """
    _log.info(f"Performing HPO using the working directory {working_directory}, using "
              f"the data at {datadir}, over {max_evaluations_total} evaluations.")

    train_data, valid_data = load_data(datadir=datadir, output=output)
    #pipeline = functools.partial(train_surrogate, train_data=train_data,
    #                             valid_data=valid_data)
    #neps.run(
    #    run_pipeline=pipeline,
    #    pipeline_space=xgb_hp_space,
    #    max_evaluations_total=max_evaluations_total,
    #    root_directory=working_directory,
    #)
    configs = {
        'cifar10': {
            'FLOPS': {"max_depth": 13, "min_child_weight": 3, "colsample_bytree": 0.9894194984624115, "learning_rate": 0.4298743573798361, "colsample_bylevel": 0.5336792923603212, "objective": "reg:squarederror"},
            'latency': {"max_depth": 8, "min_child_weight": 2, "colsample_bytree": 0.9044929762493081, "learning_rate": 0.034005940668999736, "colsample_bylevel": 0.7143949368559803, "objective": "reg:squarederror"},
            'runtime': {"max_depth": 7, "min_child_weight": 3, "colsample_bytree": 0.9110474003460679, "learning_rate": 0.061978177672735356, "colsample_bylevel": 0.7855260725661742, "objective": "reg:squarederror"},
            'size_MB': {"max_depth": 9, "min_child_weight": 3, "colsample_bytree": 0.7250019651522001, "learning_rate": 0.03418893499953842, "colsample_bylevel": 0.8256687300539098, "objective": "reg:squarederror"},
            'test-acc': {"max_depth": 13, "min_child_weight": 5, "colsample_bytree": 0.9724047005415497, "learning_rate": 0.09053211837801255, "colsample_bylevel": 0.7464788352321519, "objective": "reg:squarederror"},
            'train-acc': {"max_depth": 15, "min_child_weight": 1, "colsample_bytree": 0.9626708601567103, "learning_rate": 0.09764334654817297, "colsample_bylevel": 0.5159746924385222, "objective": "reg:squarederror"},
            'valid-acc': {"max_depth": 11, "min_child_weight": 2, "colsample_bytree": 0.9142965497670843, "learning_rate": 0.036619410301248066, "colsample_bylevel": 0.4271068040877486, "objective": "reg:squarederror"},
        },

        'colorectal_histology': {
            'FLOPS': {"max_depth": 13, "min_child_weight": 5, "colsample_bytree": 0.9372407226996564, "learning_rate": 0.006897726273340075, "colsample_bylevel": 0.8499512425733706, "objective": "reg:squarederror"},
            'latency': {"max_depth": 10, "min_child_weight": 1, "colsample_bytree": 0.6747816018486166, "learning_rate": 0.055682756619298716, "colsample_bylevel": 0.9561869627702166, "objective": "reg:squarederror"},
            'runtime': {"max_depth": 13, "min_child_weight": 6, "colsample_bytree": 0.9953930848011375, "learning_rate": 0.01081646189265034, "colsample_bylevel": 0.8100065222494222, "objective": "reg:squarederror"},
            'size_MB': {"max_depth": 13, "min_child_weight": 4, "colsample_bytree": 0.9848520043667802, "learning_rate": 0.017189491047700872, "colsample_bylevel": 0.6146378544573082, "objective": "reg:squarederror"},
            'test-acc': {"max_depth": 10, "min_child_weight": 2, "colsample_bytree": 0.8130368279314643, "learning_rate": 0.039397832485838453, "colsample_bylevel": 0.9924918785027883, "objective": "reg:squarederror"},
            'train-acc': {"max_depth": 13, "min_child_weight": 4, "colsample_bytree": 0.8106553162313552, "learning_rate": 0.017377212040741375, "colsample_bylevel": 0.7512334442881023, "objective": "reg:squarederror"},
            'valid-acc': {"max_depth": 13, "min_child_weight": 5, "colsample_bytree": 0.7609926012440027, "learning_rate": 0.06331670331888917, "colsample_bylevel": 0.9123578439521998, "objective": "reg:squarederror"},
        },

        'fashion_mnist': {
            'FLOPS': {"max_depth": 10, "min_child_weight": 1, "colsample_bytree": 0.9227343893232429, "learning_rate": 0.06713552556287267, "colsample_bylevel": 0.7804661341159358, "objective": "reg:squarederror"},
            'latency': {"max_depth": 11, "min_child_weight": 2, "colsample_bytree": 0.9097320432186742, "learning_rate": 0.12932263029991045, "colsample_bylevel": 0.47547506240916015, "objective": "reg:squarederror"},
            'runtime': {"max_depth": 15, "min_child_weight": 3, "colsample_bytree": 0.924594690714, "learning_rate": 0.1291835840403784, "colsample_bylevel": 0.4283068300231483, "objective": "reg:squarederror"},
            'size_MB': {"max_depth": 10, "min_child_weight": 1, "colsample_bytree": 0.8614882116280503, "learning_rate": 0.4354844080384641, "colsample_bylevel": 0.7351690284664579, "objective": "reg:squarederror"},
            'test-acc': {"max_depth": 13, "min_child_weight": 8, "colsample_bytree": 0.5514540168174684, "learning_rate": 0.2917854116040824, "colsample_bylevel": 0.8713979765270232, "objective": "reg:squarederror"},
            'train-acc': {"max_depth": 13, "min_child_weight": 1, "colsample_bytree": 0.9922606261280303, "learning_rate": 0.19956458594869395, "colsample_bylevel": 0.546876987090727, "objective": "reg:squarederror"},
            'valid-acc': {"max_depth": 14, "min_child_weight": 4, "colsample_bytree": 0.7342282996177476, "learning_rate": 0.37038817862952605, "colsample_bylevel": 0.820010941153909, "objective": "reg:squarederror"},
        }
    }
    cfg = configs[dataset][output]
    train_surrogate(working_directory, train_data=train_data, valid_data=valid_data, **cfg)

    _log.info("Finished.")

def parse_cli():
    parser = argparse.ArgumentParser(
        "Use the NEPS framework to train a surrogate model by iteratively optimizing "
        "the hyperparameters."
    )
    parser.add_argument("--working_directory", type=Path,
                        help="A working directory to be used by NEPS for various tasks "
                             "that require writing to disk.", default='')
    parser.add_argument("--datadir", type=Path,
                        help="Path to the directory where the cleaned, tidy training "
                             "and validation data splits to be used are stored.")
    parser.add_argument("--output", type=str,
                        help="Which of the available performance metrics to train the "
                             "surrogate for.", default='')
    parser.add_argument("--max_evaluations_total", type=int, default=5,
                        help="Number of evaluations that this NEPS worker should "
                             "perform.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    _log.setLevel(logging.INFO)
    args = parse_cli()
    datasets = ['cifar10', 'colorectal_histology', 'fashion_mnist']
    metrics = ['FLOPS', 'latency', 'runtime', 'size_MB', 'test-acc', 'train-acc', 'valid-acc']
    rt = RTPT('JS', 'JAHS_Surrogates', len(datasets)*len(metrics))
    rt.start()
    for ds in datasets:
        directory = args.datadir / ds
        for m in metrics:
            work_dir = directory / m
            _log.info(f'Train Surrogate using data from {directory}. Save model at {work_dir}')
            train_surrogate_with_correct_config(work_dir, directory, m, ds)
            rt.step()
