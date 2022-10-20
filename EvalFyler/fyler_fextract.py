#!/usr/bin/env python3
"""
Perform training or model selection for the downstream task of using a pretrained
Fyler code encoder from `Fyler` to encode documents and infer phenotypes.
"""

import functools
import io
import multiprocessing
import pathlib
import re
import string
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import json

import click
import numpy as np
import pandas as pd
import sklearn.metrics
from torch import nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

# for sklearn grid search?
import random

random.seed(1337)  # just in case
np.random.seed(1337)

import sys, os, pickle

sys.path.append("../Lib/")
sys.path.append("../Fyler/")
sys.path.append("../Codes/")

import configparser, torch

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, label_binarize

# my python modules
from fyler_dataphenot import (
    FylerDatasetProvider,
    FylerTextDatasetProvider,
    FylerEhr2VecDatasetProvider,
    get_labels,
)
import fyler_bow, utils, metrics
from bow import BagOfWords as DimaBOW

from ehr2vec import Ehr2VecConnection, EHR2VEC_URL


# ignore sklearn warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


BASE = os.environ["DATA_ROOT"]


def grid_search(
    x, y, scoring: str, model_class: type, param_grid: Optional[dict] = None
):
    """Find best model"""

    clf: BaseEstimator
    param_grid: dict

    if issubclass(model_class, LogisticRegression):
        clf = model_class(class_weight="balanced", max_iter=100)
        if param_grid is None:
            param_grid = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    elif issubclass(model_class, SVC):
        clf = model_class(class_weight="balanced", max_iter=500, probability=True)
        if param_grid is None:
            param_grid = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    elif issubclass(model_class, MLPClassifier):
        clf = model_class(max_iter=200, solver="adam", activation="relu")
        if param_grid is None:
            param_grid = {
                "hidden_layer_sizes": [(100,), (200,), (500,), (1000,)],
                "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
                "batch_size": [64, 128, 256, 512],
            }
    else:
        raise ValueError(
            f'Downstream model type "{model_class.__name__}" '
            f"not one of LogisticRegression, SVC, or MLPClassifier"
        )
    print("Scoring with", scoring)
    gs = GridSearchCV(
        clf, param_grid, scoring=scoring, cv=10, verbose=10, error_score="raise"
    )
    gs.fit(x, y)
    print("best model:\n", str(gs.best_estimator_))

    return gs.best_estimator_


def run_evaluation_dense(cfg: configparser.ConfigParser, device, model_class: str):
    """Use pre-trained patient representations"""

    (x_train, y_train, x_test, y_test), labels = data_dense(
        cfg, device, model_class=model_class
    )
    if (
        model_type := cfg.get("model", "model_type", fallback="logistic").lower()
    ) == "logistic":
        downstream_class = LogisticRegression
        kwargs = {"class_weight": "balanced"}
    elif model_type == "svm":
        downstream_class = SVC
        kwargs = {"class_weight": "balanced"}
    elif model_type == "mlp":
        downstream_class = MLPClassifier
        kwargs = {"solver": "adam", "activation": "relu"}
    else:
        raise ValueError(
            f'Downstream model type "{model_class}" not one of "logistic", "svm", or "mlp"'
        )

    if cfg.get("data", "classif_param") == "search":
        param_grid_str = cfg.get("model", "param_grid", fallback=None)
        if param_grid_str is not None:
            param_grid = json.loads(param_grid_str)
        else:
            param_grid = None
        classifier = grid_search(
            x_train,
            y_train,
            "f1_macro",  # if multiclass else "roc_auc",
            downstream_class,
            param_grid,
        )
    else:
        classifier = downstream_class(**kwargs)
        classifier.fit(x_train, y_train)

    print("\n" + "-" * 63 + "\n")

    # TODO (angus @ 2021-10-26): this is a hack to detect whether
    #  the problem is multiclass; encode this better
    multiclass = len(os.listdir(cfg.get("data", "train"))) > 2

    # # sanity check on train set
    # probs_train = classifier.predict_proba(x_train)
    # print("ON TRAIN: ", end="")
    # metrics.report_roc_auc(
    #     label_binarize(y_train, classes=sorted(set(y_train))),
    #     probs_train
    # )
    #
    # print(
    #     "\nON TRAIN: ".join(
    #         [""]
    #         + classification_report(
    #             y_train, np.argmax(probs_train, axis=1), target_names=labels
    #         )
    #         .strip()
    #         .split("\n")
    #     )
    # )
    #
    # print("\n" + "-" * 63 + "\n")

    # real result on test set
    probs = classifier.predict_proba(x_test)
    print("ON TEST:  ", end="")

    if multiclass:
        metrics.report_roc_auc(
            label_binarize(y_test, classes=sorted(set(y_test))), probs
        )
    else:
        metrics.report_roc_auc(y_test, probs[:, 1])

    print(classification_report(y_test, np.argmax(probs, axis=1), target_names=labels))
    # print(
    #     "\nON TEST:  ".join(
    #         [""]
    #         + classification_report(
    #             y_test, np.argmax(probs, axis=1), target_names=labels
    #         )
    #         .strip()
    #         .split("\n")
    #     )
    # )


def get_dense_representations(
    model,
    x,
    device,
    batch_size=16,
):
    """Run sparse x through pretrain model and get dense representations"""

    model.to(device)
    model.eval()

    tensor_dataset = TensorDataset(x)
    data_loader = DataLoader(
        dataset=tensor_dataset,
        sampler=SequentialSampler(tensor_dataset),
        batch_size=batch_size,
    )

    # list of batched dense representations
    dense_x = []

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        batch_inputs = batch[0]

        with torch.no_grad():
            logits = model(batch_inputs, return_hidden=True)
            logits = logits.cpu()
            dense_x.append(logits.numpy())

    return np.vstack(dense_x)


def data_dense(
    cfg: configparser.ConfigParser, device, model_class
) -> Tuple[Tuple[np.ndarray, List[int], np.ndarray, List[int]], List[str]]:
    """Data to feed into code prediction model"""
    train_data = cfg.get("data", "train")
    test_data = cfg.get("data", "test")

    # load model configuration
    pkl = open(cfg.get("data", "config_pickle"), "rb")
    config = pickle.load(pkl)

    # instantiate model and load parameters
    if (model_class_lower := model_class.lower()) == "fyler":
        config["model_dir"] = config.get(
            "model_dir", os.path.dirname(cfg.get("data", "model_file"))
        )
        model = fyler_bow.BagOfWords(**config, save_config=False)
    elif model_class_lower == "codes":
        if "model_dir" in config:
            del config["model_dir"]
        model = DimaBOW(**config, save_config=False)
    elif model_class_lower == "ehr2vec":
        model = Ehr2VecConnection(EHR2VEC_URL)
    elif model_class_lower == "saved":
        model = None
    else:
        raise ValueError(f"model type {model_class!r} not recognized")

    if isinstance(model, nn.Module):
        state_dict = torch.load(cfg.get("data", "model_file"))
        model.load_state_dict(state_dict)
        model.eval()

        # load training data first
        train_data_provider = FylerDatasetProvider(
            train_data, cfg.get("data", "tokenizer_pickle")
        )

        x_train, y_train = train_data_provider.load_as_int_seqs()
        x_train_mat = utils.sequences_to_matrix(x_train, config["input_vocab_size"])

        # make training vectors for target task
        x_train_arr = get_dense_representations(model, x_train_mat, device)

        # now load the test set
        test_data_provider = FylerDatasetProvider(
            test_data, cfg.get("data", "tokenizer_pickle")
        )

        x_test, y_test = test_data_provider.load_as_int_seqs()
        x_test_mat = utils.sequences_to_matrix(x_test, config["input_vocab_size"])

        # make test vectors for target task
        x_test_arr = get_dense_representations(model, x_test_mat, device)

        labels = test_data_provider.labels
    elif model is None:
        # load training data first
        train_data_provider = FylerEhr2VecDatasetProvider(train_data)

        x_train_vecs, y_train = train_data_provider.load_as_vecs()

        x_train_arr = np.vstack(x_train_vecs)

        # now load the test set
        test_data_provider = FylerEhr2VecDatasetProvider(test_data)

        x_test_vecs, y_test = test_data_provider.load_as_vecs()

        x_test_arr = np.vstack(x_test_vecs)

        labels = test_data_provider.labels
    else:
        # load training data first
        train_data_provider = FylerTextDatasetProvider(train_data)

        x_train, y_train = train_data_provider.load_as_strs()

        # make training vectors for target task
        x_train_arr = model.extract_many(x_train)

        # now load the test set
        test_data_provider = FylerTextDatasetProvider(test_data)

        x_test, y_test = test_data_provider.load_as_strs()

        # make test vectors for target task
        x_test_arr = model.extract_many(x_test)

        labels = test_data_provider.labels

    return (x_train_arr, y_train, x_test_arr, y_test), labels


def main(
    cfg: configparser.ConfigParser, stdout=None, device=None, model_class: str = "fyler"
):
    if stdout is not None:
        sys.stdout = stdout

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_evaluation_dense(cfg, device, model_class=model_class)

    sys.stdout = sys.__stdout__


#
# @dataclass()
# class Result:
#     cfg_path: str
#     output: str = field(repr=False)
#     best_epoch: int
#     val_loss: float
#     val_f1: float
#
#     def scores(self):
#         return self.best_epoch, self.val_loss, self.val_f1
#
#
# def parse(output: str):
#     lines = output.strip().split('\n')
#     epochs = [line for line in lines if line.startswith('ep: ')]  # 1-indexed
#     best_epoch_re = re.compile(r'best loss .+ after (\d+) epochs', re.M)
#     best_epoch_result = best_epoch_re.search(output)
#     if best_epoch_result is None:
#         raise RuntimeError(f'Unable to parse output for best epoch number: {output}')
#     best_epoch = int(best_epoch_result.group(1))
#     score_re = re.compile(r'val loss:\s+(\d*\.\d*),\s+val macro f1:\s+(\d*\.\d*)')
#     score_result = score_re.search(epochs[best_epoch-1])
#     if score_result is None:
#         raise RuntimeError(f'Unable to parse output for f1 score: {output}')
#     return best_epoch, float(score_result.group(1)), float(score_result.group(2))


def train_model(
    gpu,
    model_class: str,
    model_dir: pathlib.Path,
    out_dir: Optional[pathlib.Path],
    cfg_path: pathlib.Path,
):
    """ """
    init_time = time.time()
    pid = multiprocessing.current_process().pid
    print(
        f'{time.strftime("%a, %d %b %Y %H:%M:%S")}: {pid}: Training model for {cfg_path!s}'
    )
    if isinstance(gpu, int):
        gpu_id = gpu
    else:
        gpu_id = gpu.get(timeout=30)
    try:
        cfg_obj = configparser.ConfigParser()
        cfg_obj.read(cfg_path)

        expand_cfg(cfg_obj, MODELS=model_dir)

        if gpu_id >= 0:
            device = torch.device("cuda", gpu_id)
        else:
            device = torch.device("cpu")

        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]

        if out_dir is not None:
            out_io = (out_dir / f"{cfg_name}_fextract.txt").open("w", encoding="utf8")
        else:
            out_io = None

        try:
            main(cfg_obj, out_io, device, model_class=model_class)
        finally:
            if out_io is not None:
                out_io.close()

        if out_dir is not None:
            with (out_dir / f"{cfg_name}_fextract.txt").open(
                "r", encoding="utf8"
            ) as out_file:
                out = out_file.read()
        else:
            out = None

        print(f"{pid}: Done. Elapsed: {time.time() - init_time:.2f} seconds")
        return out, cfg_name
    finally:
        if hasattr(gpu, "put"):
            gpu.put(gpu_id)


def expand_cfg(cfg: configparser.ConfigParser, **kwargs):
    """
    Expands a ConfigParser object in place given string template keys and substitution values.

    :param cfg: an instantiated ConfigParser object
    :param kwargs: the template keys and substitution values
    """
    for section in cfg.values():
        for key, value in section.items():
            section[key] = os.path.expandvars(
                string.Template(value).safe_substitute(**kwargs)
            )


@click.group()
def grp():
    pass


@grp.command()
@click.argument("CFG_PATH", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument(
    "EXPERIMENT",
    type=click.Choice(
        ["Fontan", "Eisenmenger", "d-TGA", "cc-TGA", "NYHA-FC"], case_sensitive=False
    ),
)
@click.option(
    "--model",
    type=click.Choice(["fyler", "codes"], case_sensitive=False),
    default="fyler",
)
def fextract(cfg_path, experiment: str, model: str):
    """
    !! DOES NOT WORK - JUST USE BATCH FOR NOW

    Evaluate a single configuration.
    """
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    expand_cfg(cfg, experiment=experiment)
    if model == "fyler":
        model_class = fyler_bow.BagOfWords
    else:
        model_class = DimaBOW
    main(cfg, model_class=model_class)


@grp.command(name="batch")
@click.argument(
    "cfg_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "model_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.option("--num-gpus", "-g", type=int, default=None)
@click.option(
    "--out-dir", "-o", type=click.Path(file_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--exclude", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--model_class",
    "-m",
    type=click.Choice(["fyler", "codes", "ehr2vec", "saved"], case_sensitive=False),
    default="fyler",
)
def batch_(
    cfg_dir: pathlib.Path,
    model_dir: pathlib.Path,
    num_gpus: Optional[int],
    out_dir: Optional[pathlib.Path],
    exclude: Optional[pathlib.Path],
    model_class: str,
):
    """
    Evaluate several models.

    * ``fyler`` model_class is the default fyler code encoder.
    * ``codes`` is the ICD code encoder model.
    * ``ehr2vec`` uses an ehr2vec server as the encoder
    * ``saved`` uses saved ehr2vec vectors; this currently only points to vectors
      in ``train_npy`` and ``test_npy`` folders in the same directory as the ``train``
      and ``test`` input folders.

    :param cfg_dir: path to a directory of config files specifying different models
    :param model_dir: path to the root directory of the model files; fills in
                      $MODELS in the config files
    :param num_gpus: number of GPUs to allocate to the batch evaluation; defaults to maximum available
    :param out_dir: optional; the directory to write the results to. Prints to standard output otherwise.
    :param exclude: optional; file containing names of configs to exclude from the batch
                    (useful for resuming aborted runs)
    :param model_class: the type of model being loaded; defaults to "fyler", but can be any of
                        "fyler", "codes", "ehr2vec", or "saved"
    """
    actual_gpus = torch.cuda.device_count()
    print(f"Have {actual_gpus} GPUs available")
    if num_gpus is None:
        num_gpus = actual_gpus
    if actual_gpus < num_gpus:
        raise RuntimeError("Not enough GPUs available")

    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
    cfg_paths = set(cfg_dir.glob("*.cfg"))

    if exclude is not None:
        with exclude.open("r", encoding="utf8") as exclude_file:
            exclude_paths = {
                pathlib.Path(cfg_dir, path + ".cfg")
                for path in exclude_file.read().strip().split()
            }

        cfg_paths = cfg_paths - exclude_paths

    if num_gpus == 1:
        print("Training with 1 gpu")
        results = [
            train_model(0, model_class, model_dir, out_dir, path)
            for path in sorted(cfg_paths)
        ]
    else:
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        for gpu_id in range(num_gpus):
            queue.put(gpu_id)
        print(
            f"{len(os.sched_getaffinity(0))} CPUs available: {repr(os.sched_getaffinity(0))[1:-1]}"
        )

        print(f"Attempting to allocate a pool of {num_gpus} processes...")

        with multiprocessing.Pool(num_gpus) as p:
            print("Have pool")
            results = p.map(
                functools.partial(train_model, queue, model_class, model_dir, out_dir),
                sorted(cfg_paths),
            )

    # scores = []
    # cfg_paths_used = []
    # for idx, result in enumerate(results):
    #     with open(pathlib.Path(out_dir,
    #                            os.path.splitext(os.path.basename(result.cfg_path))[
    #                                0] + '_train.txt'), 'w',
    #               encoding='utf8') as out_file:
    #         out_file.write(result.output)
    #     scores.append(result.scores())
    #     cfg_paths_used.append(result.cfg_path)
    # df = pd.DataFrame(scores, columns=('best_epoch', 'val_loss', 'val_f1'))
    # df['best_epoch'] = df['best_epoch'].astype(int)
    # df.insert(0, 'cfg_path', cfg_paths_used)
    # print(df.to_csv(index=False))


if __name__ == "__main__":
    grp()
