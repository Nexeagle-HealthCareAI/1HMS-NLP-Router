#!/usr/bin/env python
# coding: utf-8
"""
location_brain/Pincode_final.py  -- BACKWARD-COMPATIBILITY SHIM
---------------------------------------------------------------
This file is retained so that existing imports continue to work:

    from location_brain.Pincode_final import PincodeFinder, _script_dir

All logic has been moved to the SOLID-compliant modules:
    - location_brain/repositories.py
    - location_brain/fallback_strategies.py
    - location_brain/services.py
    - location_brain/finder.py
    - location_brain/cli.py

Do NOT add new code here. Depend on the new modules directly.
"""

import os

# Re-export the refactored PincodeFinder so callers stay unbroken.
from .finder import PincodeFinder  # noqa: F401

# Re-export CLI helpers for any scripts that still import them from here.
from .cli import interactive_lookup, print_result  # noqa: F401


def _script_dir() -> str:
    """Return the directory of this file (used by main.py for CSV path resolution)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


if __name__ == "__main__":
    from .cli import interactive_lookup

    base_dir = _script_dir()
    cities_file = os.path.join(base_dir, "Indian_Cities_Database.csv")
    pincodes_file = os.path.join(base_dir, "pincode-dataset.csv")

    finder = PincodeFinder(cities_file, pincodes_file)

    print("Welcome to the Offline City & Pincode Finder!")
    while True:
        interactive_lookup(finder)
        again = input("\nLook up another city? [y/N]: ").strip().lower()
        if again != "y":
            break
