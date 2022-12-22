import json
import sys
import random

from pathlib import Path
from typing import Optional

import click


START = """{
  "data": [
"""

END = """
  ]
}
"""


@click.command()
@click.argument("in_file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--out-path", "-o", type=click.Path(path_type=Path, exists=False, file_okay=False), default=None)
@click.option("--random-seed", "-s", type=int, default=42)
@click.option("--proportion", "-p", type=float, default=.9)
def main(in_file: Path, out_path: Optional[Path], random_seed: int, proportion: float):
    with in_file.open('r', encoding='utf8') as in_fd:
        if out_path is None:
            out_fd = sys.stdout
            out_fd.write(START)
        else:
            train_fd = (out_path / 'train.json').open('w', encoding='utf8')
            dev_fd = (out_path / 'dev.json').open('w', encoding='utf8')
            train_fd.write(START)
            dev_fd.write(START)

        out_has_first_line = False
        train_has_first_line = False
        dev_has_first_line = False

        count = 0

        if out_path is not None:
            random.seed(random_seed)

        for line in in_fd:
            if out_path is None:
                this_fd = out_fd
                has_first_line = out_has_first_line
            else:
                this_fd, has_first_line = (train_fd, train_has_first_line) if random.random() < proportion else (dev_fd, dev_has_first_line)

            if has_first_line:
                # Write rest of lines indented with preceding comma-newline
                this_fd.write(",\n")
            count += 1
            if count % 32 == 0:
                if out_path is None:
                    out_fd.flush()
                else:
                    train_fd.flush()
                    dev_fd.flush()
            this_fd.write("    ")
            this_fd.write(line[:-1])

        if out_path is None:
            out_fd.write(END)
            out_fd.flush()
            out_fd.close()
        else:
            train_fd.write(END)
            train_fd.flush()
            train_fd.close()
            dev_fd.write(END)
            dev_fd.flush()
            dev_fd.close()


if __name__ == '__main__':
    main()
