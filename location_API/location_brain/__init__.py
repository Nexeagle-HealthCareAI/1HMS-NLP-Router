"""
location_brain
--------------
Offline city / pincode / coordinate lookup for Indian cities.

Public API
~~~~~~~~~~
    from location_brain import PincodeFinder          # full facade
    from location_brain.interfaces import (           # narrow interfaces
        ICitySearcher, IPincodeFinder, ICoordinateFinder
    )
    from location_brain.repositories import (         # data-access layer
        CitiesRepository, PincodeRepository
    )
    from location_brain.services import (             # individual services
        CitySearchService, PincodeLookupService, CoordinateLookupService
    )
    from location_brain.fallback_strategies import (  # extensible fallbacks
        FallbackChain, SubstringFallback, FuzzyFallback, default_fallback_chain
    )
    from location_brain.Pincode_final import _script_dir  # backward-compat
"""

from .finder import PincodeFinder
from .interfaces import ICitySearcher, ICoordinateFinder, IPincodeFinder

__all__ = [
    "PincodeFinder",
    "ICitySearcher",
    "IPincodeFinder",
    "ICoordinateFinder",
]
