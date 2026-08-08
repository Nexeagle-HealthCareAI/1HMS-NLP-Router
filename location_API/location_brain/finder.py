"""
location_brain/finder.py
------------------------
Concrete `PincodeFinder` that composes repositories + services and
implements all three narrow interfaces (ICitySearcher, IPincodeFinder,
ICoordinateFinder).

This is the single entry-point that outside consumers (e.g. main.py) can
depend on when they need all three capabilities together.  Each interface
is satisfied through delegation to the appropriate service, so every
service remains individually testable and swappable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .interfaces import ICitySearcher, ICoordinateFinder, IPincodeFinder
from .fallback_strategies import FallbackChain, default_fallback_chain
from .repositories import CitiesRepository, PincodeRepository
from .services import (
    CitySearchService,
    CoordinateLookupService,
    PincodeLookupService,
)


class PincodeFinder(ICitySearcher, IPincodeFinder, ICoordinateFinder):
    """
    Facade that wires repositories, fallback chain, and services together.

    Callers that only need one capability should depend on the matching
    narrow interface (ICitySearcher / IPincodeFinder / ICoordinateFinder)
    rather than on this concrete class directly.
    """

    def __init__(
        self,
        cities_csv: str,
        pincodes_csv: str,
        fallback: Optional[FallbackChain] = None,
    ) -> None:
        cities_repo = CitiesRepository(cities_csv)
        pincode_repo = PincodeRepository(pincodes_csv)
        chain = fallback or default_fallback_chain()

        self._searcher = CitySearchService(cities_repo, pincode_repo)
        self._pincode_svc = PincodeLookupService(cities_repo, pincode_repo, chain)
        self._coord_svc = CoordinateLookupService(cities_repo, pincode_repo)

    # --- ICitySearcher ---

    def search_cities(self, query: str, limit: int = 10) -> List[Dict]:
        return self._searcher.search_cities(query, limit=limit)

    # --- IPincodeFinder ---

    def find(self, city_query: str, state: Optional[str] = None) -> Dict:
        return self._pincode_svc.find(city_query, state=state)

    # --- ICoordinateFinder ---

    def get_coordinates(self, city_query: str) -> Dict:
        return self._coord_svc.get_coordinates(city_query)
