from __future__ import annotations

from typing import Literal


StaleTimeoutSource = Literal["provider_model", "provider"]


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    resolved = resolve_provider_stale_timeout(provider_id, model)
    return resolved[0] if resolved is not None else None


def resolve_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> tuple[float, StaleTimeoutSource] | None:
    """Return an explicit stale timeout and its provider-config source."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None

    for config_provider_id in _provider_config_ids(provider_id):
        provider_config = providers.get(config_provider_id, {})
        if not isinstance(provider_config, dict):
            continue

        model_config = _get_model_config(provider_config, model)
        if model_config is not None:
            timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
            if timeout is not None:
                return timeout, "provider_model"

        timeout = _coerce_timeout(provider_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout, "provider"
    return None


def _provider_config_ids(provider_id: str) -> tuple[str, ...]:
    """Return config keys for a provider, including named-custom aliases.

    Hermes runs a named custom provider as ``custom`` while retaining its
    requested ``custom:<name>`` identity.  The configuration, however, is
    keyed by the bare provider name.  Accept the canonical runtime spelling
    first, then its config spelling, so named-provider stale settings apply.
    """
    normalized = provider_id.strip().lower()
    if normalized.startswith("custom:"):
        named_provider = normalized.split(":", 1)[1].strip()
        if named_provider:
            return normalized, named_provider
    return (normalized,)


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
