"""Errors shared by every Octoqai package."""


class ConfigError(Exception):
    """Raised when an experiment file cannot be loaded or is invalid.

    It is also raised when a language model provider cannot be configured, for example by
    ``provider_for`` or ``OllamaProvider``.
    """
