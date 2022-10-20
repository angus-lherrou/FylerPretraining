#!/usr/bin/env python3
"""
Utilities and classes for downstream classification datasets.
"""
import pathlib
import sys
from typing import Tuple, List

import numpy as np

sys.dont_write_bytecode = True
sys.path.append("../Lib/")
import configparser, os, pickle


def get_labels(corpus_path):
    return os.listdir(corpus_path)


class IntSeqMixin:
    def load_as_int_seqs(self) -> Tuple[List[List[int]], List[int]]:
        raise NotImplementedError


class StrMixin:
    def load_as_strs(self) -> Tuple[List[str], List[int]]:
        raise NotImplementedError


class VecMixin:
    def load_as_vecs(self) -> Tuple[List[np.ndarray], List[int]]:
        raise NotImplementedError


class FylerBaseDatasetProvider:
    def __init__(self, corpus_path):
        self.corpus_path = pathlib.Path(corpus_path)
        self.labels = get_labels(self.corpus_path)
        self.label2int = {
            label.lower(): i for label, i in zip(self.labels, range(len(self.labels)))
        }


class FylerEhr2VecDatasetProvider(FylerBaseDatasetProvider, VecMixin):
    # def __init__(self, corpus_path, vector_path) -> None:
    #     super(FylerEhr2VecDatasetProvider, self).__init__(corpus_path)
    #     self.vector_path = vector_path

    def __init__(self, corpus_path):
        super(FylerEhr2VecDatasetProvider, self).__init__(corpus_path)

        # TODO (angus @ 2021-10-19): factor this out as a config parameter
        self.vector_path = self.corpus_path.parent / (self.corpus_path.name + "_npy")

    def load_as_vecs(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load examples as np.ndarrays"""

        x = []
        y = []

        for d in self.labels:
            label_dir = os.path.join(self.vector_path, d)

            for f in os.listdir(label_dir):
                int_label = self.label2int[d.lower()]
                y.append(int_label)

                # todo: treat tokens as set?
                file_path = os.path.join(label_dir, f)
                arr = np.load(file_path)
                x.append(arr)

        return x, y


class FylerTextDatasetProvider(FylerBaseDatasetProvider, StrMixin):
    def load_as_strs(self) -> Tuple[List[str], List[int]]:
        """Load examples as strs"""

        x = []
        y = []

        for d in self.labels:
            label_dir = os.path.join(self.corpus_path, d)

            for f in os.listdir(label_dir):
                int_label = self.label2int[d.lower()]
                y.append(int_label)

                # todo: treat tokens as set?
                file_path = os.path.join(label_dir, f)
                text = open(file_path).read()
                x.append(text)

        return x, y


class FylerDatasetProvider(FylerBaseDatasetProvider, IntSeqMixin):
    """Read data from files and make keras inputs/outputs"""

    def __init__(self, corpus_path, tokenizer_pickle) -> None:
        super(FylerDatasetProvider, self).__init__(corpus_path)

        with open(tokenizer_pickle, "rb") as pkl:
            self.tokenizer = pickle.load(pkl)

    def load_as_int_seqs(self) -> Tuple[List[List[int]], List[int]]:
        """Convert examples into lists of indices"""

        x = []
        y = []

        for d in self.labels:
            label_dir = os.path.join(self.corpus_path, d)

            for f in os.listdir(label_dir):
                int_label = self.label2int[d.lower()]
                y.append(int_label)

                # todo: treat tokens as set?
                file_path = os.path.join(label_dir, f)
                text = open(file_path).read()
                x.append(text)

        x = self.tokenizer.texts_as_sets_to_seqs(x)

        return x, y


def main():
    cfg = configparser.ConfigParser()
    cfg.read(sys.argv[1])

    base = os.environ["DATA_ROOT"]
    if os.path.isabs(cfg.get("data", "train")):
        data_dir = cfg.get("data", "train")
    else:
        data_dir = os.path.join(base, cfg.get("data", "train"))
    tokenizer_pickle = cfg.get("data", "tokenizer_pickle")

    dp = FylerDatasetProvider(data_dir, tokenizer_pickle)
    x, y = dp.load_as_int_seqs()

    print(x[1])


if __name__ == "__main__":
    main()
