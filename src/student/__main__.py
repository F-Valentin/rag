"""Entry point for python -m student."""

import fire
from student.cli import main

if __name__ == "__main__":
    fire.Fire(main)