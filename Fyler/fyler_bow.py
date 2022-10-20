#!/usr/bin/env python3
"""
Adapted from Codes/bow.py

Train a Fyler code encoding model on bag of words.
"""
import functools
import io
import multiprocessing
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import click
import numpy as np
import pandas as pd
import sklearn.metrics

sys.path.append("../Lib/")

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

import os, configparser, random, pickle
import fyler_data, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)

# model and model config locations
MODEL_DIR = "Model"
MODEL_PATH = "model.pt"
CONFIG_PATH = "config.p"


class BagOfWords(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        hidden_units,
        dropout_rate,
        model_dir,
        save_config=True,
    ):
        """Constructor"""

        super(BagOfWords, self).__init__()

        self.hidden = nn.Linear(in_features=input_vocab_size, out_features=hidden_units)

        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(
            in_features=hidden_units, out_features=output_vocab_size
        )

        self.init_weights()

        # save configuration for loading later
        if save_config:
            config = {
                "input_vocab_size": input_vocab_size,
                "output_vocab_size": output_vocab_size,
                "hidden_units": hidden_units,
                "dropout_rate": dropout_rate,
                "model_dir": model_dir,
            }
            pickle_file = open(os.path.join(model_dir, CONFIG_PATH), "wb")
            pickle.dump(config, pickle_file)

    def init_weights(self):
        """Never trust pytorch default weight initialization"""

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self, texts, return_hidden=False):
        """Optionally return hidden layer activations"""

        features = self.hidden(texts)
        output = self.activation(features)
        output = self.dropout(output)
        output = self.classifier(output)

        if return_hidden:
            return features
        else:
            return output


def make_data_loader(model_inputs, model_outputs, batch_size, partition):
    """DataLoader objects for train or dev/test sets"""

    # e.g. transformers take input ids and attn masks
    if type(model_inputs) is tuple:
        tensor_dataset = TensorDataset(*model_inputs, model_outputs)
    else:
        tensor_dataset = TensorDataset(model_inputs, model_outputs)

    # use sequential sampler for dev and test
    if partition == "train":
        sampler = RandomSampler(tensor_dataset)
    else:
        sampler = SequentialSampler(tensor_dataset)

    data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size)

    return data_loader


def fit(
    model,
    cfg,
    train_loader,
    val_loader,
    n_epochs,
    *,
    model_dir: Optional[pathlib.Path] = None,
    device: torch.device = None,
):
    """Training routine"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is None:
        model_dir = MODEL_DIR

    model.to(device)
    print(f"Fitting on {device!r}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.getfloat("model", "lr"))

    best_loss = float("inf")
    optimal_epochs = 0

    for epoch in range(1, n_epochs + 1):

        model.train()
        train_loss, num_train_steps = 0, 0

        for batch in train_loader:

            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            batch_inputs, batch_outputs = batch

            logits = model(batch_inputs)
            loss = criterion(logits, batch_outputs)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_train_steps += 1

        av_tr_loss = train_loss / num_train_steps
        val_loss, val_f1 = evaluate(model, val_loader, device)
        print(
            "ep: %d, steps: %d, tr loss: %.4f, val loss: %.4f, val macro f1: %.4f"
            % (epoch, num_train_steps, av_tr_loss, val_loss, val_f1)
        )

        if val_loss < best_loss:
            print(
                f"loss improved, saving model to {os.path.join(model_dir, MODEL_PATH)}..."
            )
            torch.save(model.state_dict(), os.path.join(model_dir, MODEL_PATH))
            best_loss = val_loss
            optimal_epochs = epoch

    sys.__stdout__.write(f"Best model saved to {os.path.join(model_dir, MODEL_PATH)}\n")
    sys.__stdout__.flush()
    return best_loss, optimal_epochs


def evaluate(model, data_loader, device=None):
    """Evaluation routine"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Evaluating on {device!r}")

    criterion = nn.BCEWithLogitsLoss()
    total_loss, num_steps = 0, 0

    model.eval()

    outputs = []
    y_hats = []

    for batch in data_loader:

        batch = tuple(t.to(device) for t in batch)
        batch_inputs, batch_outputs = batch

        with torch.no_grad():
            logits = model(batch_inputs)
            loss = criterion(logits, batch_outputs)

        outputs.append(batch_outputs.cpu().numpy())
        y_hats.append(np.round(torch.sigmoid(logits).cpu()))

        total_loss += loss.item()
        num_steps += 1

    av_loss = total_loss / num_steps

    output_array = np.concatenate(outputs, axis=0)
    y_hat_array = np.concatenate(y_hats, axis=0)

    f1 = sklearn.metrics.f1_score(
        output_array, y_hat_array, average="macro", zero_division=0
    )

    return av_loss, f1


def main(cfg, model_dir: Optional[pathlib.Path] = None, stdout=None, device=None):
    """My main main"""

    if stdout is not None:
        sys.stdout = stdout

    root = os.path.expandvars(cfg.get("data", "root"))

    notes_path = os.path.join(root, fyler_data.NOTE)
    text_path = os.path.join(root, fyler_data.TEXT)
    notes = fyler_data.open_notes(notes_path)

    dp = fyler_data.FylerDatasetProvider(
        conn=notes,
        note_dir=text_path,
        input_vocab_size=cfg.get("args", "cui_vocab_size"),
        code_vocab_size=cfg.get("args", "code_vocab_size"),
        cfg=cfg["data"],
        tokenizer_dir=model_dir / "tokenizer",
    )

    in_seqs, out_seqs = dp.load_as_sequences()

    tr_in_seqs, val_in_seqs, tr_out_seqs, val_out_seqs = train_test_split(
        in_seqs, out_seqs, test_size=0.10, random_state=2020
    )

    print(
        "loaded %d training and %d validation samples"
        % (len(tr_in_seqs), len(val_in_seqs))
    )

    max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
    print("longest cui sequence:", max_cui_seq_len)

    max_code_seq_len = max(len(seq) for seq in tr_out_seqs)
    print("longest code sequence:", max_code_seq_len)

    train_loader = make_data_loader(
        utils.sequences_to_matrix(tr_in_seqs, len(dp.input_tokenizer.stoi)),
        utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
        cfg.getint("model", "batch"),
        "train",
    )

    val_loader = make_data_loader(
        utils.sequences_to_matrix(val_in_seqs, len(dp.input_tokenizer.stoi)),
        utils.sequences_to_matrix(val_out_seqs, len(dp.output_tokenizer.stoi)),
        cfg.getint("model", "batch"),
        "dev",
    )

    if model_dir is None:
        model_dir = pathlib.Path(MODEL_DIR)

    os.makedirs(model_dir, exist_ok=True)

    model = BagOfWords(
        input_vocab_size=len(dp.input_tokenizer.stoi),
        output_vocab_size=len(dp.output_tokenizer.stoi),
        hidden_units=cfg.getint("model", "hidden"),
        dropout_rate=cfg.getfloat("model", "dropout"),
        model_dir=model_dir,
    )

    best_loss, optimal_epochs = fit(
        model,
        cfg,
        train_loader,
        val_loader,
        cfg.getint("model", "epochs"),
        model_dir=model_dir,
        device=device,
    )
    print("best loss %.4f after %d epochs" % (best_loss, optimal_epochs))

    if stdout is not None:
        sys.stdout = sys.__stdout__


@dataclass()
class Result:
    cfg_path: str
    output: str = field(repr=False)
    best_epoch: int
    val_loss: float
    val_f1: float

    def scores(self):
        return self.best_epoch, self.val_loss, self.val_f1


def parse(output: str):
    lines = output.strip().split("\n")
    epochs = [line for line in lines if line.startswith("ep: ")]  # 1-indexed
    best_epoch_re = re.compile(r"best loss .+ after (\d+) epochs", re.M)
    best_epoch_result = best_epoch_re.search(output)
    if best_epoch_result is None:
        raise RuntimeError(f"Unable to parse output for best epoch number: {output}")
    best_epoch = int(best_epoch_result.group(1))
    score_re = re.compile(r"val loss:\s+(\d*\.\d*),\s+val macro f1:\s+(\d*\.\d*)")
    score_result = score_re.search(epochs[best_epoch - 1])
    if score_result is None:
        raise RuntimeError(f"Unable to parse output for f1 score: {output}")
    return best_epoch, float(score_result.group(1)), float(score_result.group(2))


def train_model(
    gpu,
    model_dir: pathlib.Path,
    out_dir: Optional[pathlib.Path],
    cfg_path: pathlib.Path,
):
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
        out_io = io.StringIO()

        device = torch.device("cuda", gpu_id)

        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        model_dir_extended = pathlib.Path(model_dir, cfg_name)

        main(cfg_obj, model_dir_extended, out_io, device)

        out = out_io.getvalue()
        result = Result(str(cfg_path), out, *parse(out))

        if out_dir is not None:
            with open(
                pathlib.Path(out_dir, cfg_name + "_train.txt"), "w", encoding="utf8"
            ) as out_file:
                out_file.write(result.output)
        else:
            print(result.output)

        print(f"{pid}: Done. Elapsed: {time.time() - init_time:.2f} seconds")
        return result
    finally:
        if hasattr(gpu, "put"):
            gpu.put(gpu_id)


@click.group()
def grp():
    pass


@grp.command()
@click.argument(
    "cfg_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
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
def batch(
    cfg_dir: pathlib.Path,
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    num_gpus: Optional[int],
    out_dir: Optional[pathlib.Path],
    exclude: Optional[pathlib.Path],
):
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    print(f"Have {num_gpus} GPUs available")
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
    cfg_paths = set(cfg_dir.glob("*.cfg"))
    data_paths = set(path.name for path in data_dir.glob("*"))

    if exclude is not None:
        with exclude.open("r", encoding="utf8") as exclude_file:
            exclude_paths = {
                pathlib.Path(cfg_dir, path + ".cfg")
                for path in exclude_file.read().strip().split()
            }

        cfg_paths = cfg_paths - exclude_paths

    paths = [
        path
        for path in cfg_paths
        if os.path.splitext(os.path.basename(path))[0] in data_paths
    ]

    if num_gpus == 1:
        print("Training with 1 gpu")
        results = [train_model(0, model_dir, out_dir, path) for path in paths]
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
                functools.partial(train_model, queue, model_dir, out_dir), paths
            )

    scores = []
    cfg_paths_used = []
    for idx, result in enumerate(results):
        with open(
            pathlib.Path(
                out_dir,
                os.path.splitext(os.path.basename(result.cfg_path))[0] + "_train.txt",
            ),
            "w",
            encoding="utf8",
        ) as out_file:
            out_file.write(result.output)
        scores.append(result.scores())
        cfg_paths_used.append(result.cfg_path)
    df = pd.DataFrame(scores, columns=("best_epoch", "val_loss", "val_f1"))
    df["best_epoch"] = df["best_epoch"].astype(int)
    df.insert(0, "cfg_path", cfg_paths_used)
    print(df.to_csv(index=False))


@grp.command()
@click.argument("cfg", type=click.Path(exists=True, dir_okay=False))
def bow(cfg):
    cfg_obj = configparser.ConfigParser()
    cfg_obj.read(cfg)
    main(cfg)


if __name__ == "__main__":
    grp()
