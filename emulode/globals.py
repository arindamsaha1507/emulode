"""Module for global constants and enumerations."""

from enum import Enum


class Sampler(Enum):
    """Enumeration of supported samplers."""

    LATIN_HYPERCUBE = "latin_hypercube"
    UNIFORN = "uniform"
