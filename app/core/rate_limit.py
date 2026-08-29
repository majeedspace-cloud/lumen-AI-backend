"""Single shared rate limiter instance.

Must be imported from here everywhere it's used (main.py registers it on
the app, routes.py applies it to endpoints) — two separate Limiter()
instances would track request counts independently instead of sharing
state, silently breaking the actual rate limiting.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
