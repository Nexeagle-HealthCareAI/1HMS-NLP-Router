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
    """Search for city / town / district names that match a free-text query."""

    @abstractmethod
    def search_cities(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Return up to *limit* best-matching place records, ranked by match
        quality (exact > prefix > substring > fuzzy typo).

        Each record is a dict with keys:
          name, type ("city" | "town" | "district"), state, district,
          pincodes (capped sample), coordinates (if known).
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


class ISmartLocator(ABC):
    """Resolve a free-form location query -- a pincode, "lat,lon"
    coordinates, or a city/town/district name -- into one unified result,
    without the caller needing to know or specify which kind of query it
    is (that's the whole point: ICitySearcher/IPincodeFinder/
    ICoordinateFinder each require the caller to already know they have a
    city name; this doesn't)."""

    @abstractmethod
    def locate(self, query: str) -> Dict:
        """
        Return a result dict with keys:
          query, queryType, found, matched, district, state, pincodes,
          coordinates, details, suggestions, message
        """
