import sqlite3
import glob
import re
import pathlib

import pandas as pd
from tqdm.auto import tqdm


def main():
    # load sql database
    conn = sqlite3.connect(
        f"{pathlib.Path.home()}/mnt/ACHD/extracted_notes/15mar2021/notes.sql"
    )
    notes = pd.read_sql_query(
        "SELECT DISTINCT event_id, note_key FROM NOTE;",
        conn,
    )

    # regular expressions to capture event ID and note key fields
    event_re = re.compile(r"Event-(\d+)")
    key_re = re.compile(r"Key-(\d+)")

    # extract event IDs and note keys from text filenames into a dataframe
    files_df = pd.DataFrame(
        [
            {
                "event_id": int(event_re.findall(file)[0]),
                "note_key": int(key_re.findall(file)[0]),
            }
            for file in tqdm(
                glob.glob("~/mnt/ACHD/extracted_notes/15mar2021/text/Event-*_Key-*.txt")
            )
        ]
    )

    outer = notes.merge(
        files_df, on=["event_id", "note_key"], how="outer", indicator=True
    )

    missing_from_files = outer[outer["_merge"] == "left_only"]
    missing_from_sql = outer[outer["_merge"] == "right_only"]
    with open("missing_from_files.txt", "w") as mff_txt:
        for _, event_id, note_key, _ in missing_from_files.itertuples():
            mff_txt.write(f"Event-{event_id}_Key-{note_key}.txt\n")

    with open("missing_from_sql.txt", "w") as sql_txt:
        for _, event_id, note_key, _ in missing_from_sql.itertuples():
            sql_txt.write(f"Event-{event_id}_Key-{note_key}.txt\n")


if __name__ == "__main__":
    main()
