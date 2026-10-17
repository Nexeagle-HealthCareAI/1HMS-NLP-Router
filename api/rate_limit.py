"""Shared rate limiter instance. Its own module so both main.py (which wires
it into the app) and routes.py (which uses it as a decorator) can import it
without a circular dependency between the two."""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
