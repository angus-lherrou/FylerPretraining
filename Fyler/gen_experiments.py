#!/usr/bin/env python
"""
Generate .cfg files for pretraining or evaluation.
"""

import configparser
import pathlib
import os
import click
from string import Template


def format_filename(**kwargs):
    """Create a descriptive suffix for a config filename given its added parameters."""
    return "_".join(f"{key}-{value}" for key, value in kwargs.items()) + ".cfg"


@click.group()
def grp():
    pass


def prepare(base: pathlib.Path) -> configparser.ConfigParser:
    """Loads a config file."""
    cfg = configparser.ConfigParser()
    cfg.read(base)
    return cfg


@grp.command()
@click.argument(
    "base", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path)
)
@click.argument("out", type=click.Path(file_okay=False, path_type=pathlib.Path))
def fextract(base: pathlib.Path, out: pathlib.Path):
    """
    Generates config files for downstream inference from a base config file at BASE
    with phenotype, window size, note count, and alignment parameters.
    """
    cfg = prepare(base)

    prior_train = Template(cfg["data"]["train"])
    prior_test = Template(cfg["data"]["test"])

    prior_tokenizer_pickle = Template(cfg["data"]["tokenizer_pickle"])
    prior_config_pickle = Template(cfg["data"]["config_pickle"])
    prior_model_file = Template(cfg["data"]["model_file"])

    # for experiment in ['Fontan', 'Eisenmenger', 'd-TGA', 'cc-TGA', 'NYHA-FC']:
    for experiment in ["cyanosis", "PH_1", "PH_2", "AA"]:
        cfg["data"]["train"] = prior_train.substitute(experiment=experiment)
        cfg["data"]["test"] = prior_test.substitute(experiment=experiment)
        (out / experiment).mkdir(parents=True, exist_ok=True)
        for fyler_window_size in [7, 14, 30]:
            for note_window_size in [3, 7, 14]:
                for fyler_min_count in [5, 7, 10, 15, 20]:
                    for align in ["left", "right"]:
                        filename = format_filename(
                            experiment=experiment,
                            fyler_window_size=fyler_window_size,
                            note_window_size=note_window_size,
                            fyler_min_count=fyler_min_count,
                            align=align,
                        )
                        cfg_name = format_filename(
                            fyler_window_size=fyler_window_size,
                            note_window_size=note_window_size,
                            fyler_min_count=fyler_min_count,
                            align=align,
                        )[:-4]
                        cfg["data"][
                            "tokenizer_pickle"
                        ] = prior_tokenizer_pickle.substitute(cfg=cfg_name)
                        cfg["data"]["config_pickle"] = prior_config_pickle.substitute(
                            cfg=cfg_name
                        )
                        cfg["data"]["model_file"] = prior_model_file.substitute(
                            cfg=cfg_name
                        )
                        with (out / experiment / filename).open(
                            "w", encoding="utf8"
                        ) as cfg_out:
                            cfg.write(cfg_out)


@grp.command()
@click.argument(
    "base", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path)
)
@click.argument("out", type=click.Path(file_okay=False, path_type=pathlib.Path))
def pretrain(base: pathlib.Path, out: pathlib.Path):
    """
    Generates config files for pretraining from a base config file at BASE with
    window size, note count, and alignment parameters.
    """
    cfg = prepare(base)

    out.mkdir(exist_ok=True)

    base_save_dir = cfg["data"]["save_dir"]

    for fyler_window_size in [7, 14, 30]:
        for note_window_size in [3, 7, 14]:
            for fyler_min_count in [5, 7, 10, 15, 20]:
                for align in ["left", "right"]:
                    # for cui_vocab_size in ['10000', '20000', '30000', 'all']:
                    #     for code_vocab_size in ['1000', '3000', '5000', 'all']:
                    #         pass
                    filename = format_filename(
                        fyler_window_size=fyler_window_size,
                        note_window_size=note_window_size,
                        fyler_min_count=fyler_min_count,
                        align=align,
                    )
                    cfg["data"]["fyler_window_size"] = str(fyler_window_size)
                    cfg["data"]["note_window_size"] = str(note_window_size)
                    cfg["data"]["fyler_min_count"] = str(fyler_min_count)
                    cfg["data"]["align"] = align
                    cfg["data"]["save_dir"] = os.path.join(
                        base_save_dir, os.path.splitext(filename)[0]
                    )
                    with pathlib.Path(out, filename).open(
                        "w", encoding="utf8"
                    ) as cfg_out:
                        cfg.write(
                            cfg_out
                        )  # ``write`` method from ConfigParser, not file object


@grp.command()
@click.argument(
    "in_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=pathlib.Path))
def downstream_model_aug(in_dir: pathlib.Path, out_dir: pathlib.Path):
    """
    Augments existing config files in IN_DIR with the ``model::model_type`` parameter.

    For every .cfg file in IN_DIR, generates three new .cfg files in OUT_DIR
    with an additional ``model::model_type`` value of "logistic", "svm", or "mlp".
    These model types are used for the downstream classifiers that infer phenotypes
    from the Fyler code vectors.
    """
    out_dir.mkdir(exist_ok=True)

    for cfg_file in in_dir.rglob("*.cfg"):
        cfg = prepare(cfg_file)
        for model_type in ["logistic", "svm", "mlp"]:
            cfg["model"]["model_type"] = model_type
            out_subdir = out_dir / cfg_file.parent.relative_to(in_dir)
            out_subdir.mkdir(parents=True, exist_ok=True)
            with (out_subdir / f"{cfg_file.stem}_{model_type}.cfg").open(
                "w", encoding="utf8"
            ) as cfg_out:
                cfg.write(
                    cfg_out
                )  # ``write`` method from ConfigParser, not file object


if __name__ == "__main__":
    grp()
