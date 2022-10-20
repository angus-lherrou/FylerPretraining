#!/usr/bin/env python
"""
Generate data (fyler_data.py) and train models (fyler_bow.py) for many configurations on multiple GPUs.
Note: only the `data` subcommand is currently functional. `train` does not work with the current code in fyler_bow.py.
"""


import concurrent.futures as cf
import multiprocessing, multiprocessing.queues, multiprocessing.managers
import functools
import os
import time
from dataclasses import dataclass, field
import re
import subprocess
import pathlib
from typing import Optional

import pandas as pd
import click
import torch.cuda


@dataclass()
class Result:
    cfg_path: str
    output: str = field(repr=False)
    best_epoch: int
    val_loss: float
    val_f1: float

    def scores(self):
        return self.best_epoch, self.val_loss, self.val_f1


@dataclass()
class DataGen:
    cfg_path: str
    output: str = field(repr=False)


def gen_data(cfg_path: pathlib.Path):
    init_time = time.time()
    pid = multiprocessing.current_process().pid
    print(f"{pid}: Generating data for {cfg_path!s}")
    completed_process = subprocess.run(
        ["python", "fyler_data.py", cfg_path],
        capture_output=True,
        text=True,
    )
    out = completed_process.stdout
    if completed_process.returncode != 0:
        raise RuntimeError(out + completed_process.stderr)
    print(
        f'{time.strftime("%a, %d %b %Y %H:%M:%S")}: {pid}: Done. Elapsed: {time.time() - init_time:.2f} seconds'
    )
    return DataGen(str(cfg_path), out)


def train_model(gpu, cfg_path: pathlib.Path):
    init_time = time.time()
    pid = multiprocessing.current_process().pid
    print(
        f'{time.strftime("%a, %d %b %Y %H:%M:%S")}: {pid}: Generating data for {cfg_path!s}'
    )
    if isinstance(gpu, int):
        gpu_id = gpu
    else:
        gpu_id = gpu.get(timeout=30)
    try:
        completed_process = subprocess.run(
            ["python", "fyler_bow.py", cfg_path],
            capture_output=True,
            text=True,
            env={**os.environ.copy(), "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        out = completed_process.stdout
        if completed_process.returncode != 0:
            raise RuntimeError(out + completed_process.stderr)
        print(f"{pid}: Done. Elapsed: {time.time() - init_time:.2f} seconds")
        return Result(str(cfg_path), out, *parse(out))
    finally:
        if hasattr(gpu, "put"):
            gpu.put(gpu_id)


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
@click.option("--num-gpus", "-g", type=int, default=None)
@click.option(
    "--out-dir", "-o", type=click.Path(file_okay=False, path_type=pathlib.Path)
)
def train(
    cfg_dir: pathlib.Path,
    data_dir: pathlib.Path,
    num_gpus: Optional[int],
    out_dir: pathlib.Path,
):
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    print(f"Have {num_gpus} GPUs available")
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
    cfg_paths = list(cfg_dir.glob("*.cfg"))
    data_paths = set(path.name for path in data_dir.glob("*"))

    paths = [
        path
        for path in cfg_paths
        if os.path.splitext(os.path.basename(path))[0] in data_paths
    ]

    if num_gpus == 1:
        print("Training with 1 gpu")
        results = [train_model(0, path) for path in paths]
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
            results = p.map(functools.partial(train_model, queue), paths)

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
@click.argument(
    "cfg_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--exclude", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option("--processes", "-p", type=int, default=None)
@click.option(
    "--out-dir", "-o", type=click.Path(file_okay=False, path_type=pathlib.Path)
)
def data(
    cfg_dir: pathlib.Path,
    exclude: Optional[pathlib.Path],
    processes: Optional[int],
    out_dir: Optional[pathlib.Path],
):
    if processes is None:
        processes = len(os.sched_getaffinity(0)) - 2
    print("Will run on", processes, "cores")
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

    with cf.ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(gen_data, cfg_path) for cfg_path in cfg_paths]
        for idx, future in enumerate(cf.as_completed(futures)):
            result = future.result()
            if out_dir is not None:
                with open(
                    pathlib.Path(
                        out_dir,
                        os.path.splitext(os.path.basename(result.cfg_path))[0]
                        + "_data.txt",
                    ),
                    "w",
                    encoding="utf8",
                ) as out_file:
                    out_file.write(result.output)
            else:
                print(result.output)


if __name__ == "__main__":
    grp()
