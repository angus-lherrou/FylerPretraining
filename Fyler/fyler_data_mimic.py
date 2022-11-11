#!/usr/bin/env python3
"""
Generate the data for an experiment given a configuration file.
"""

import collections
import configparser
import json
import os
import pathlib
import pickle
import shutil
import sys
import typing
import datetime
import math
import sqlite3

import pandas as pd
import click
from tqdm.auto import tqdm

sys.path.append("../Lib")
from tokenizer import Tokenizer

MODEL_DIR = pathlib.Path("Model/")

NOTE = "notes.sql"
TEXT = "text"

NOTE_COUNT = collections.Counter()

__all__ = [
    "NOTE",
    "TEXT",
    "MODEL_DIR",
    "open_notes",
    "fetch_note",
    "fetch_several_notes",
    "FylerDatasetProvider",
]


def open_notes(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)


def fetch_note(
    text_path: str,
    event_id: int,
    note_key: int,
) -> str:
    full_path = os.path.join(text_path, f"Event-{event_id}_Key-{note_key}.txt")
    if not os.path.exists(full_path):
        NOTE_COUNT["EMPTY"] += 1
        return ""
    with open(
        full_path,
        "r",
        encoding="utf8",
    ) as note_obj:
        text = note_obj.read()
        if text:
            NOTE_COUNT["CONTENTFUL"] += 1
        else:
            NOTE_COUNT["EMPTY"] += 1
        return text


def fetch_several_notes(
    text_path: str,
    event_id_note_key_pairs: typing.Sequence[typing.Tuple[int, int]],
) -> str:

    return "\n".join(
        fetch_note(text_path, event_id, note_key)
        for event_id, note_key in event_id_note_key_pairs
    )


class DataDictifier:
    def __init__(self, code_keys: typing.List[float], num_digits: int):
        code_keys = [str(int(code))[:num_digits] for code in code_keys]
        self.code_keys: typing.List[str] = code_keys
        self.code_key_set: typing.Set[str] = set(code_keys)
        if overlaps := ({"id", "text"} & self.code_key_set):
            raise ValueError(f"Label keys {overlaps} conflict with data schema")

    def metadata(self) -> dict:
        # TODO (10/26/22): finalize
        return {
            "version": "1.0",
            "task": "fyler-pretraining",
            "subtasks": [
                {
                    "task_name": code,
                    "output_mode": "classification",
                }
                for code in self.code_keys
            ],
        }

    def data_dict(self, *, ident: str, text: str, codes: typing.List[str],) -> dict:
        code_set = set(codes)

        return {
            "id": ident,
            "text": text,
            **{
                key: "Y" if key in code_set else "N"
                for key in self.code_keys
            }
        }


class FylerDatasetProvider:  # (DatasetProvider):
    """Notes and Fyler code data"""

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        note_dir: str,
        input_vocab_size: str,
        code_vocab_size: str,
        cfg: configparser.SectionProxy,
        tokenizer_dir: pathlib.Path = MODEL_DIR,
    ):
        """Construct it"""
        regenerate = cfg.get("regenerate")
        self.regenerate = True
        if regenerate is not None:
            assert regenerate.lower() in {"y", "n", "yes", "no", "true", "false"}
            self.regenerate = regenerate.lower() in {"y", "yes", "true"}

        self.conn = conn
        self.fyler_path = os.path.expandvars(cfg["fyler"])
        self.note_dir = note_dir
        self.save_dir = os.path.expandvars(cfg["save_dir"])
        self.tokenizer_dir = tokenizer_dir

        self.input_vocab_size = None if input_vocab_size == "all" else int(input_vocab_size)
        self.code_vocab_size = None if code_vocab_size == "all" else int(code_vocab_size)

        self.notes_df: pd.DataFrame = pd.read_sql_query(
            "SELECT DISTINCT mrn, event_id, note_date, "
            "                GROUP_CONCAT(note_key) AS note_keys "
            "FROM NOTE "
            "GROUP BY mrn, event_id, note_date;",
            self.conn,
        )
        print("loaded notes dataframe")

        self.notes_df["note_date"] = pd.to_datetime(self.notes_df["note_date"])
        self.notes_df["note_keys"] = self.notes_df["note_keys"].str.split(",")

        self.mrns = set(self.notes_df["mrn"].unique())

        redcap_df = pd.read_csv(
            os.path.expandvars(
                "$ACHD/redcap/" "ResultsFromTheBoston_DATA_LABELS_2021-02-17_1050.csv"
            ),
            dtype=str,
        )
        print("loaded redcap dataframe")

        self.redcap_mrns = {
            float(item)
            for (index, item) in redcap_df["Patient MRN"].dropna().iteritems()
            if type(item) is str and item.replace(".", "", 1).isnumeric()
        }

        if os.path.isdir(self.tokenizer_dir):
            shutil.rmtree(self.tokenizer_dir)
        os.makedirs(self.tokenizer_dir)

        # event ids -> Fyler code sets
        self.event2codes = {}

        self.index_codes(
            self.fyler_path,
            "FYLER_CODE",
            num_digits=4,
            fyler_window_size=int(cfg["fyler_window_size"]),
            note_window_size=int(cfg["note_window_size"]),
            fyler_min_count=int(cfg["fyler_min_count"]),
            iteration_window_size=int(cfg["iteration_window_size"]),
            train_samples=int(cfg["train_samples"])
            if cfg.get("train_samples")
            else None,
            align=cfg["align"] if cfg.get("align") else "left",
        )

        # index inputs (cuis)
        self.input_tokenizer = Tokenizer(
            n_words=self.input_vocab_size,
            lower=False,
        )
        self.tokenize_input()

        # index outputs (codes)
        self.output_tokenizer = Tokenizer(
            n_words=self.code_vocab_size,
            lower=False,
        )
        self.tokenize_output()

    def generate_events(
        self,
        code_file,
        code_col,
        *,
        num_digits,
        note_window_size,
        fyler_window_size,
        iteration_window_size,
        fyler_min_count,
        sep="|",
        align="left",
        train_samples=None,
    ):
        if os.path.exists(self.save_dir) and os.listdir(self.save_dir):
            if click.confirm(
                f"{self.save_dir} exists and is not empty. Confirm regeneration",
                err=True,
            ):
                shutil.rmtree(self.save_dir)

        os.makedirs(self.save_dir)

        print(
            f"Generating data source in {self.save_dir} \n"
            f"with parameters: \n"
            f" * {num_digits            = } \n"
            f" * {note_window_size      = } \n"
            f" * {fyler_window_size     = } \n"
            f" * {iteration_window_size = } \n"
            f" * {fyler_min_count       = } \n"
            f" * {align                 = } \n"
        )
        fyler_df = pd.read_csv(code_file, sep=sep)
        print("loaded fyler dataframe")
        fyler_df["MODALITY_EVENT_DATE"] = pd.to_datetime(
            fyler_df["MODALITY_EVENT_DATE"]
        )

        uniq_fyler_mrns = set(fyler_df["MRN"].unique())

        # FIXME: temporarily reduce the number of MRNs
        # pretrain_mrns = (uniq_fyler_mrns & self.mrns) - self.redcap_mrns
        pretrain_mrns = (uniq_fyler_mrns & self.mrns) & self.redcap_mrns
        pretrain_mrns = set(list(pretrain_mrns)[::4])

        note_window = datetime.timedelta(days=note_window_size)
        fyler_window = datetime.timedelta(days=fyler_window_size)

        iteration_window = datetime.timedelta(days=iteration_window_size)

        event_id = 0

        # TODO (10/31/2022): is this the right way to get all fyler codes?
        #  Have to find a way to get N most frequent fyler codes, per tokenizer
        fyler_code_col = fyler_df[fyler_df["MRN"].isin(pretrain_mrns) & fyler_df[code_col].notna()][code_col]

        if self.code_vocab_size is None:
            all_fyler_codes = sorted(fyler_code_col.unique())
        else:
            val_cts = fyler_code_col.value_counts()
            all_fyler_codes = sorted(val_cts.nlargest(self.code_vocab_size).index)

        print(f"Found {len(all_fyler_codes)} fyler codes")

        dictifier = DataDictifier(all_fyler_codes, num_digits)

        with open(os.path.join(self.save_dir, 'metadata.json'), 'w', encoding='utf8') as metadata_fd:
            json.dump(dictifier.metadata(), metadata_fd)

        with open(os.path.join(self.save_dir, 'all.jsonl'), 'w', encoding='utf8') as save_fd:
            for patient in tqdm(list(pretrain_mrns)[:train_samples]):
                patient_df = fyler_df[fyler_df["MRN"] == patient]
                patient_notes = self.notes_df[self.notes_df["mrn"] == patient]
                min_date = max(
                    patient_df["MODALITY_EVENT_DATE"].min(),
                    self.notes_df["note_date"].min(),
                )
                max_date = patient_df["MODALITY_EVENT_DATE"].max()
                num_windows = math.ceil((max_date - min_date) / iteration_window)
                start_dates = [min_date + n * iteration_window for n in range(num_windows)]
                for start_date in start_dates:
                    window_df = patient_df[
                        patient_df["MODALITY_EVENT_DATE"].between(
                            pd.to_datetime(start_date),
                            pd.to_datetime(start_date + fyler_window),
                            inclusive="left",
                        )
                    ].dropna(subset=[code_col])
                    codes = set(
                        str(int(code))[:num_digits] for code in window_df[code_col].unique()
                    )
                    if len(codes) >= fyler_min_count:
                        if align == "left":
                            window_notes = patient_notes[
                                patient_notes["note_date"].between(
                                    pd.to_datetime(start_date),
                                    pd.to_datetime(start_date + note_window),
                                    inclusive="left",
                                )
                            ]
                        elif align == "right":
                            window_notes = patient_notes[
                                patient_notes["note_date"].between(
                                    pd.to_datetime(start_date + fyler_window - note_window),
                                    pd.to_datetime(start_date + fyler_window),
                                    inclusive="right",
                                )
                            ]
                        else:
                            raise NotImplementedError(
                                f'unsupported alignment criterion "{align}"'
                            )
                        note_ids = [
                            (item.event_id, note_key)
                            for item in window_notes.itertuples()
                            for note_key in item.note_keys
                        ]
                        notes = fetch_several_notes(self.note_dir, note_ids)

                        if notes:
                            json.dump(
                                dictifier.data_dict(
                                    ident=str(event_id),
                                    text=notes,
                                    codes=sorted(codes),
                                ),
                                save_fd,
                            )
                            save_fd.write('\n')
                            # with open(
                            #     os.path.join(self.input_dir, f"{event_id}.txt"), "w"
                            # ) as note_txt:
                            #     note_txt.write(notes + "\n")
                            # with open(
                            #     os.path.join(self.output_dir, f"{event_id}.pkl"),
                            #     "wb",
                            # ) as fyler_pkl:
                            #     pickle.dump(codes, fyler_pkl)
                            event_id += 1

        print(f'{NOTE_COUNT["CONTENTFUL"] = }')
        print(f'{NOTE_COUNT["EMPTY"] = }')

    def index_codes(
        self,
        code_file,
        code_col,
        *,
        num_digits,
        note_window_size,
        fyler_window_size,
        iteration_window_size,
        fyler_min_count,
        sep="|",
        align="left",
        train_samples=None,
    ):
        """Map encounters to codes"""

        if (
            self.regenerate
            or not os.path.exists(self.save_dir)
            or not os.listdir(self.save_dir)
        ):
            self.generate_events(
                code_file,
                code_col,
                num_digits=num_digits,
                note_window_size=note_window_size,
                fyler_window_size=fyler_window_size,
                iteration_window_size=iteration_window_size,
                fyler_min_count=fyler_min_count,
                sep=sep,
                align=align,
                train_samples=train_samples,
            )
        # for file_path in pathlib.Path(self.output_dir).glob("*.pkl"):
        #     with open(file_path, "rb") as fyler_pkl:
        #         self.event2codes[file_path.stem] = pickle.load(fyler_pkl)

        # with open(os.path.join(self.save_dir, "metadata.json"), "r", encoding="utf8") as metadata_fd:
        #     metadata: typing.Dict[str, str] = json.load(metadata_fd)

        with open(os.path.join(self.save_dir, "all.jsonl"), "r", encoding="utf8") as all_fd:
            for line in all_fd:
                data: typing.Dict[str, str] = json.loads(line)
                id_ = data.pop("id")
                data.pop("text")
                self.event2codes[id_] = [
                    code
                    for code, yn in data.items()
                    if yn.upper() == "Y"
                ]
        print(f"Found {len(self.event2codes)} events")

    def tokenize_input(self):
        """Read text and map tokens to ints"""

        x = []  # input documents
        with open(os.path.join(self.save_dir, "all.jsonl"), "r", encoding="utf8") as all_fd:
            for line in all_fd:
                data = json.loads(line)
                x.append(data["text"])
        # for file_path in pathlib.Path(self.input_dir).glob("*.txt"):
        #     x.append(file_path.read_text())
        self.input_tokenizer.fit_on_texts(x)

        pickle_file = open(self.tokenizer_dir / "tokenizer.p", "wb")
        pickle.dump(self.input_tokenizer, pickle_file)
        print("input vocab:", len(self.input_tokenizer.stoi))

    def tokenize_output(self):
        """Map codes to ints"""
        all_codes = set()

        y = []  # prediction targets
        for _, codes in self.event2codes.items():
            all_codes.update(set(codes))
            y.append(" ".join(codes))
        self.output_tokenizer.fit_on_texts(y)
        print(f"Found {len(all_codes)} codes")
        print("output vocab:", len(self.output_tokenizer.stoi))
        if len(self.output_tokenizer.stoi) < 10:
            print(self.output_tokenizer.stoi)
            print(self.output_tokenizer.itos)

    def load_as_sequences(self):
        """Make x and y"""

        x = []
        y = []

        # make a list of inputs and outputs to vectorize
        # for file_path in pathlib.Path(self.input_dir).glob("*.txt"):
        #     if file_path.stem not in self.event2codes:
        #         continue
        with open(os.path.join(self.save_dir, "all.jsonl"), "r", encoding="utf8") as all_fd:
            for line in all_fd:
                data = json.loads(line)
                if data["id"] not in self.event2codes:
                    continue

                x.append(data["text"])
                codes_as_string = " ".join(self.event2codes[data["id"]])
                y.append(codes_as_string)

        # make x and y matrices
        x = self.input_tokenizer.texts_as_sets_to_seqs(x, add_cls_token=True)
        y = self.output_tokenizer.texts_to_seqs(y, add_cls_token=False)

        # column zero is empty
        # return x, y[:,1:]
        return x, y


def fyler_data():
    cfg = configparser.ConfigParser()
    cfg.read(sys.argv[1])
    root = os.path.expandvars(cfg.get("data", "root"))

    notes_path = os.path.join(root, NOTE)
    text_path = os.path.join(root, TEXT)
    notes = open_notes(notes_path)

    cfg_path = sys.argv[1]
    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
    tok_dir = pathlib.Path("models", cfg_name, "tokenizer")
    print(tok_dir)

    FylerDatasetProvider(
        conn=notes,
        note_dir=text_path,
        input_vocab_size=cfg.get("args", "cui_vocab_size"),
        code_vocab_size=cfg.get("args", "code_vocab_size"),
        cfg=cfg["data"],
        tokenizer_dir=tok_dir,
    )


if __name__ == "__main__":
    fyler_data()
