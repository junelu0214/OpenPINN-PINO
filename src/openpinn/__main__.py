"""Allow ``python -m openpinn`` to run the unified CLI."""

from .cli import main


raise SystemExit(main())
