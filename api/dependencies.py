from functools import lru_cache

from graph.hitl import HITLGate
from graph.graph_builder import mosaic_graph
from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore


@lru_cache(maxsize=1)
def get_hitl_gate() -> HITLGate:
    """Return the shared HITLGate instance."""
    return HITLGate()


@lru_cache(maxsize=1)
def get_episodic_store() -> EpisodicStore:
    """Return the shared EpisodicStore instance."""
    return EpisodicStore()


@lru_cache(maxsize=1)
def get_procedural_store() -> ProceduralStore:
    """Return the shared ProceduralStore instance."""
    return ProceduralStore()


@lru_cache(maxsize=1)
def get_semantic_store() -> SemanticStore:
    """Return the shared SemanticStore instance."""
    return SemanticStore()


def get_graph():
    """Return the pre-compiled MOSAIC graph."""
    return mosaic_graph