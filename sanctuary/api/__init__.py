"""Sanctuary API — External interfaces.

Public API, CLI, and runner for interacting with the Sanctuary cognitive system.

Phase 6: Integration + Validation.

Imports are lazy to allow ``python -m sanctuary.api`` to work even when
optional heavy dependencies (torch, transformers, etc.) are not installed.
"""

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.api.sanctuary_api import SanctuaryAPI
from sanctuary.api.health import HealthServer
from sanctuary.api.resource_monitor import ResourceMonitor

__all__ = [
    "RunnerConfig",
    "SanctuaryRunner",
    "SanctuaryAPI",
    "HealthServer",
    "ResourceMonitor",
]


def __getattr__(name: str):
    if name == "SanctuaryRunner":
        from sanctuary.api.runner import SanctuaryRunner
        return SanctuaryRunner
    if name == "RunnerConfig":
        from sanctuary.api.runner import RunnerConfig
        return RunnerConfig
    if name == "SanctuaryAPI":
        from sanctuary.api.sanctuary_api import SanctuaryAPI
        return SanctuaryAPI
    if name == "HealthServer":
        from sanctuary.api.health import HealthServer
        return HealthServer
    if name == "ResourceMonitor":
        from sanctuary.api.resource_monitor import ResourceMonitor
        return ResourceMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
