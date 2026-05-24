"""Tiny name-based plugin registries. Each pluggable component type (scenario, algorithm,
metric, visualization) registers implementations by name; configs select them by name, so the
suite is extended by writing a class and decorating it, with no changes to the runner."""
from typing import Callable, Dict

SCENARIOS: Dict[str, type] = {}
ALGORITHMS: Dict[str, type] = {}
METRICS: Dict[str, type] = {}
VISUALIZATIONS: Dict[str, Callable] = {}


def _register(reg: dict, name: str):
    def deco(obj):
        if name in reg:
            raise KeyError("'%s' already registered in %s" % (name, reg))
        reg[name] = obj
        return obj
    return deco


def scenario(name): return _register(SCENARIOS, name)
def algorithm(name): return _register(ALGORITHMS, name)
def metric(name): return _register(METRICS, name)
def visualization(name): return _register(VISUALIZATIONS, name)


def get(reg: dict, name: str):
    if name not in reg:
        raise KeyError("'%s' not registered; available: %s" % (name, sorted(reg)))
    return reg[name]
