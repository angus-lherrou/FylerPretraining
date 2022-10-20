import os

import pandas
from tqdm import tqdm
import click


@click.command()
@click.argument("notes", type=click.Path(dir_okay=False, exists=True))
@click.argument("outpath", type=click.Path(dir_okay=True, file_okay=False))
def collect_notes(notes, outpath):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    elif len(os.listdir(outpath)) > 0:
        click.confirm(
            f"Directory {outpath} is non-empty. Files may be overwritten.\n"
            f"Continue?",
            abort=True,
        )

    print("Importing notes... ", end="", flush=True)
    notes_df = pandas.read_csv(notes)
    print("done.")

    print("Grouping notes by admission ID... ", end="", flush=True)
    groups = (
        notes_df[["HADM_ID", "TEXT"]].groupby("HADM_ID").agg(lambda x: "\n".join(x))
    )
    print("done.")

    for hadm_id, text in tqdm(groups.iterrows()):
        hadm_id = int(hadm_id)
        with open(
            os.path.join(outpath, f"{hadm_id}.txt"), "w", encoding="utf8"
        ) as txt_file:
            txt_file.write(text[0])


if __name__ == "__main__":
    collect_notes()
