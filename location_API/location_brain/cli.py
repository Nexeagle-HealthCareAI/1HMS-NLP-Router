"""
location_brain/cli.py
---------------------
Command-line interface utilities — separated from domain logic (SRP).

These functions are ONLY for interactive terminal usage.
The FastAPI application does NOT import from this module.
"""

from __future__ import annotations

from typing import Dict

from .finder import PincodeFinder


def print_result(result: Dict) -> None:
    """Pretty-print a pincode lookup result to stdout."""
    print()
    if not result["found"]:
        print(f"No pincodes found for '{result['query']}'.")
        if result["message"]:
            print(result["message"])
        if result["suggestions"]:
            print("Did you mean: " + ", ".join(result["suggestions"]) + "?")
        return

    state_note = f" ({', '.join(result['states'])})" if result["states"] else ""
    print(f"Possible pincodes for '{result['matched_city']}'{state_note}:")
    print(", ".join(result["pincodes"]))

    print("\nDetails:")
    for d in result["details"]:
        print(f"  {d['pincode']}  -  {d['district']}, {d['state']}")


def interactive_lookup(finder: PincodeFinder) -> None:
    """Run a single interactive lookup cycle."""
    query = input("\nEnter a city name: ").strip()
    if not query:
        print("Please enter a city name.")
        return

    matches = finder.search_cities(query, limit=10)

    if not matches:
        print(f"No matching locations found for '{query}'.")
        result = finder.find(query)
        if result["suggestions"]:
            print("Did you mean: " + ", ".join(result["suggestions"]) + "?")
        return

    print(f"\nFound {len(matches)} matching location(s):")
    for i, m in enumerate(matches, start=1):
        state_note = f" ({m['state']})" if m["state"] else ""
        print(f"  {i}. {m['city']}{state_note}")

    choice = input(f"\nSelect a location [1-{len(matches)}]: ").strip()
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(matches)):
            raise ValueError
    except ValueError:
        print("Invalid selection.")
        return

    selected_city = matches[idx]["city"]
    selected_state = matches[idx]["state"]
    label = f"{selected_city} ({selected_state})" if selected_state else selected_city
    print(f"\n=== Results for '{label}' ===")

    # 1. Pincodes
    pin_result = finder.find(selected_city, state=selected_state)
    print_result(pin_result)

    # 2. Coordinates
    coord_result = finder.get_coordinates(selected_city)
    print()
    if coord_result["found"]:
        print(f"Coordinates: lat={coord_result['latitude']}, long={coord_result['longitude']}")
    else:
        print(f"Coordinates: {coord_result['message']}")
