"""Module for global constants and enumerations."""

from enum import Enum


class KernelFunction(Enum):
    """Enum for the kernel function."""

    MATERN = "matern2.5"
    SQUARED_EXPONENTIAL = "sexp"


class Sampler(Enum):
    """Enumeration of supported samplers."""

    LATIN_HYPERCUBE = "latin_hypercube"
    UNIFORN = "uniform"
