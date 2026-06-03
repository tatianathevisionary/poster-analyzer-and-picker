"""Allow running the package as ``python -m poster_analyzer``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
