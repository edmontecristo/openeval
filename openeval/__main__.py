"""
openeval/__main__.py — Entry point for `python -m openeval`.

Allows OpenEval to be run as a Python module:
    python -m openeval run my_eval.py
"""

from openeval.cli import main

if __name__ == "__main__":
    main()
