"""Custom exceptions for SENTINEL."""


class IncidentLibraryError(Exception):
    """Raised when the incident library YAML is missing or malformed."""


class CascadeError(Exception):
    """Raised when the Cascade_Engine encounters an invalid root service."""
