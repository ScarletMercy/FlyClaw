import pytest


def test_acp_gateway_module_loads():
    from src.gateway import router

    routes = [r.path for r in router.routes]
    assert "/ws/acp" in routes
