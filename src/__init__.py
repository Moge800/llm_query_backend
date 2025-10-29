"""srcパッケージのルート"""

from .main import app
from .di_container import DIContainer

__all__ = ["app", "DIContainer"]
