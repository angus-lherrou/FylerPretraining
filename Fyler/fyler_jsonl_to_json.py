import json
import sys
from pathlib import Path
from typing import Optional

import click


@click.command()
@click.argument("in_file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--out-file", "-o", type=click.Path(path_type=Path, exists=False, dir_okay=False), default=None)
def main(in_file: Path, out_file: Optional[Path]):
    with in_file.open('r', encoding='utf8') as in_fd:
        if out_file is None:
            out_fd = sys.stdout
        else:
            out_fd = out_file.open('w', encoding='utf8')
        # Write preamble
        out_fd.write("""{
  "data": [
""")
        in_fd_iter = iter(in_fd)

        # Write first line indented with no preceding comma-newline
        first_line = next(in_fd_iter)
        out_fd.write("    ")
        out_fd.write(first_line[:-1])
        count = 0
        for line in in_fd_iter:
            # Write rest of lines indented with preceding comma-newline
            out_fd.write(",\n")
            count += 1
            if count % 32 == 0:
                out_fd.flush()
            out_fd.write("    ")
            out_fd.write(line[:-1])

        out_fd.write("""
  ]
}
""")
        out_fd.flush()
        out_fd.close()


if __name__ == '__main__':
    main()
