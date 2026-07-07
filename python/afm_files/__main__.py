"""Enable ``python -m afm_files`` as an alias for the ``afm-convert`` CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
