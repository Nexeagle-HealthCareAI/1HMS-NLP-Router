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

from .interfaces import ICitySearcher, ICoordinateFinder, IPincodeFinder, ISmartLocator
from .fallback_strategies import FallbackChain, default_fallback_chain
from .repositories import CitiesRepository, GovPincodeRepository, PincodeRepository, TownsRepository
from .services import (
    CitySearchService,
    CoordinateLookupService,
    PincodeLookupService,
    SmartLocationService,
)


class PincodeFinder(ICitySearcher, IPincodeFinder, ICoordinateFinder, ISmartLocator):
    """
    Facade that wires repositories, fallback chain, and services together.

    Callers that only need one capability should depend on the matching
    narrow interface (ICitySearcher / IPincodeFinder / ICoordinateFinder /
    ISmartLocator) rather than on this concrete class directly.

    `gov_pincodes_csv`/`towns_csv` are optional so existing callers that
    only pass the original two CSVs keep working -- but locate()
    (ISmartLocator) needs both and raises RuntimeError if either was
    omitted, rather than silently returning empty/degraded results.
    search_cities() (ICitySearcher) degrades more gracefully instead: it
    still works with just the original two CSVs, just with narrower
    coverage (see CitySearchService's docstring).
    """

    def __init__(
        self,
        cities_csv: str,
        pincodes_csv: str,
        gov_pincodes_csv: Optional[str] = None,
        towns_csv: Optional[str] = None,
        fallback: Optional[FallbackChain] = None,
    ) -> None:
        cities_repo = CitiesRepository(cities_csv)
        pincode_repo = PincodeRepository(pincodes_csv)
        chain = fallback or default_fallback_chain()

        # Built once here (not duplicated per-service) so CitySearchService
        # and SmartLocationService share the same in-memory repositories.
        towns_repo = TownsRepository(towns_csv) if towns_csv else None
        gov_pincode_repo = GovPincodeRepository(gov_pincodes_csv) if gov_pincodes_csv else None

        self._searcher = CitySearchService(cities_repo, pincode_repo, towns_repo=towns_repo, gov_pincode_repo=gov_pincode_repo)
        self._pincode_svc = PincodeLookupService(cities_repo, pincode_repo, chain)
        self._coord_svc = CoordinateLookupService(cities_repo, pincode_repo)

        self._smart_svc: Optional[SmartLocationService] = None
        if gov_pincode_repo and towns_repo:
            self._smart_svc = SmartLocationService(cities_repo, towns_repo, gov_pincode_repo, chain)

    # --- ICitySearcher ---

    def search_cities(self, query: str, limit: int = 10) -> List[Dict]:
        return self._searcher.search_cities(query, limit=limit)

    # --- IPincodeFinder ---

    def find(self, city_query: str, state: Optional[str] = None) -> Dict:
        return self._pincode_svc.find(city_query, state=state)

    # --- ICoordinateFinder ---

    def get_coordinates(self, city_query: str) -> Dict:
        return self._coord_svc.get_coordinates(city_query)

    # --- ISmartLocator ---

    def locate(self, query: str) -> Dict:
        if self._smart_svc is None:
            raise RuntimeError(
                "PincodeFinder was constructed without gov_pincodes_csv/towns_csv -- "
                "locate() needs both. Pass them to __init__ to enable smart search."
            )
        return self._smart_svc.locate(query)
