"""Shared exception types for the spiking neural network package."""


class ParameterError(Exception):
    """Raised when a configuration value is invalid."""


class DatasetError(Exception):
    """Raised when a dataset file is missing or invalid."""


class EncodingError(Exception):
    """Raised when spike encoding inputs or indices are invalid."""


class ImageError(Exception):
    """Raised when an image cannot be loaded or used in the pipeline."""
