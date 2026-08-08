"""
location_brain/interfaces.py
----------------------------
Abstract base classes (interfaces) for all location lookup capabilities.

Each interface is intentionally narrow (ISP) so that consumers depend only
on the methods they actually use.  Concrete implementations (e.g. PincodeFinder)
implement all three; mocks / alternative back-ends can implement just one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ICitySearcher(ABC):
    """Search for city / district names that match a free-text query."""

    @abstractmethod
    def search_cities(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Return up to *limit* best-matching city records.

        Each record is a dict with at least ``{"city": str, "state": str}``.
        """


class IPincodeFinder(ABC):
    """Look up all known pincodes for a city / district name."""

    @abstractmethod
    def find(self, city_query: str, state: Optional[str] = None) -> Dict:
        """
        Return a result dict with keys:
          query, matched_city, states, found, pincodes, details,
          suggestions, message
        """


class ICoordinateFinder(ABC):
    """Look up latitude / longitude for a city name."""

    @abstractmethod
    def get_coordinates(self, city_query: str) -> Dict:
        """
        Return a result dict with keys:
          query, found, matched_city, state, latitude, longitude,
          suggestions, message
        """
