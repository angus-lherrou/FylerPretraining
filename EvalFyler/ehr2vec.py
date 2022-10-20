#!/usr/bin/env python3
import os
import sys
import time
import shutil
from typing import Union, Optional, Tuple, List, Sequence, Dict
from pathlib import Path

import click
import numpy as np
import requests

from tqdm import tqdm

EHR2VEC_URL = "http://nlp-gpu:8000/ehr2vec"


class Ehr2VecConnection:
    def __init__(self, url: str, wait_time: float = 0.01) -> None:
        self.s = requests.Session()
        self.url = url
        self.last_request_time: Optional[float] = None
        self.wait_time = wait_time

    def _extract(self, text: str, timeout: float) -> requests.Response:
        return self.s.post(self.url, json={"note": text}, timeout=timeout)

    def extract(self, text: str, timeout: float = 5.0) -> Union[np.ndarray, str]:
        if (
            self.last_request_time is not None
            and (time_left := self.wait_time - (time.time() - self.last_request_time))
            > 0
        ):
            time.sleep(time_left)
        try:
            r = self._extract(text, timeout)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            return "timeout " + type(e).__name__
        if not r.ok:
            return r.reason
        vector = r.json()["vector"]
        self.last_request_time = time.time()
        return np.array(vector)

    @staticmethod
    def load_saved(directory: Union[Path, str]) -> np.ndarray:
        return np.vstack([np.load(p) for p in directory.glob("*.npy")])

    def extract_many(self, texts: Sequence[str], timeout: float = 5) -> np.ndarray:
        vecs = [self.extract(text, timeout=timeout) for text in tqdm(texts)]
        percent_str = 100 * sum(isinstance(vec, str) for vec in vecs) / len(vecs)
        percent_timeout = (
            100
            * sum(isinstance(vec, str) and vec[:7] == "timeout" for vec in vecs)
            / len(vecs)
        )
        if percent_str:
            print(
                f"Dropped {percent_str:.2f}% of {len(vecs)} notes", file=sys.__stderr__
            )
            print(
                f" * {percent_timeout:.2f}% timeout, {percent_str - percent_timeout:.2f}% HTTP error",
                file=sys.__stderr__,
            )
            print(
                f' * {", ".join({vec[8:] for vec in vecs if isinstance(vec, str) and vec[:7] == "timeout"})}',
                file=sys.__stderr__,
            )
            if percent_str - percent_timeout:
                print(
                    f' * {", ".join({vec for vec in vecs if isinstance(vec, str) and vec[:7] != "timeout"})}',
                    file=sys.__stderr__,
                )
            raise RuntimeError()
        return np.vstack(vecs)

    def extract_sequence(
        self, texts: Sequence[Tuple[str, str]], timeout: float
    ) -> Tuple[List[Tuple[str, List[float]]], List[Tuple[str, str]]]:
        vectors = []
        timeouts = []
        for name, text in tqdm(texts, desc=f"Timeout {timeout:.2f}"):
            try:
                r = self._extract(text, timeout)
                r.raise_for_status()
                vectors.append((name, r.json()["vector"]))
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                timeouts.append((name, text))
        return vectors, timeouts


def save_loop(
    conn: Ehr2VecConnection,
    initial_timeout: float,
    max_timeout: Optional[float],
    out_dir: Path,
    texts: List[Tuple[str, str]],
):
    num_texts = len(texts)
    timeout = initial_timeout
    new_vectors: List[Tuple[str, List[float]]]

    new_vectors, texts = conn.extract_sequence(texts, timeout=timeout)
    num_vectors = len(new_vectors)
    for name, vector in new_vectors:
        (out_dir / name).parent.mkdir(exist_ok=True)
        np.save(out_dir / name, vector)

    while texts:
        timeout *= 2
        if max_timeout is not None and timeout > max_timeout:
            print(
                "Maximum timeout reached. Failed to save",
                ", ".join(name for name, text in texts),
            )
            break
        new_vectors, texts = conn.extract_sequence(texts, timeout=timeout)
        num_vectors += len(new_vectors)
        for name, vector in new_vectors:
            (out_dir / name).parent.mkdir(exist_ok=True)
            np.save(out_dir / name, vector)
    if num_texts != num_vectors:
        print(
            f"Something went wrong. {num_texts} texts processed, {num_vectors} vectors saved."
        )


@click.group()
def grp():
    pass


@grp.command()
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "out_dir", type=click.Path(exists=False, file_okay=False, path_type=Path)
)
@click.option(
    "--url", "-u", type=str, help="URL of ehr2vec server", default=EHR2VEC_URL
)
@click.option(
    "--initial-timeout",
    "-t",
    type=float,
    help="Initial timeout in (fractional) seconds",
    default=2,
)
@click.option(
    "--max-timeout",
    "-m",
    type=float,
    help="Maximum timeout in (fractional) seconds",
    default=None,
)
def save(
    in_dir: Path,
    out_dir: Path,
    url: str,
    initial_timeout: float,
    max_timeout: Optional[float],
):
    """Save ehr2vec vectors for all text files in IN_DIR into OUT_DIR.

    Will continue until all vectors are saved. Press CTRL-C to exit.
    """
    conn = Ehr2VecConnection(url)
    texts = []

    for file in tqdm(list(in_dir.rglob("*.txt")), desc="Loading texts"):
        relative_name = os.path.splitext(file.relative_to(in_dir))[0]
        texts.append((relative_name, file.read_text(encoding="utf8")))

    out_dir.mkdir(exist_ok=True)
    save_loop(conn, initial_timeout, max_timeout, out_dir, texts)


@grp.command()
@click.argument(
    "npy_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "text_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("/dev/null"),
)
def reallocate(npy_dir: Path, text_dir: Path, out_dir: Optional[Path]):
    if out_dir == Path("/dev/null"):
        out_dir = text_dir.parent / (text_dir.name + "_npy")
    print("making", out_dir)
    out_dir.mkdir(exist_ok=True)
    # get all npy paths and key them by the file stem (document index)
    npy_paths: Dict[str, Path] = {
        p.stem: p for p in tqdm(npy_dir.rglob("*.npy"), desc="Reading npy paths")
    }

    # iterate over the texts
    for text_path in tqdm(text_dir.rglob("*.txt"), desc="Copying npys"):
        # get label path
        label_path = text_path.parent.relative_to(text_dir)

        # create new label path
        npy_label_path = out_dir / label_path
        npy_label_path.mkdir(exist_ok=True)

        # copy corresponding npy to new label directory
        npy_path = npy_paths[text_path.stem]
        new_npy_path = npy_label_path / npy_path.name

        shutil.copy(npy_path, new_npy_path)


if __name__ == "__main__":
    grp()
