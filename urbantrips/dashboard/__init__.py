from pathlib import Path


def get_dashboard_ctx():
    """Build a read-only StorageContext for dashboard use."""
    from urbantrips.config.config import load_config
    from urbantrips.storage.context import build_storage_context

    config = load_config(Path("configs") / "configuraciones_generales.yaml")
    return build_storage_context(config)
