from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app import ServiceContainer

_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    if _container is None:
        raise RuntimeError("ServiceContainer not initialized")
    return _container


def set_container(container: ServiceContainer) -> None:
    global _container
    _container = container
