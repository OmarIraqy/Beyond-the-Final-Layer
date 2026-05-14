"""Loss registry — register losses by name for config-driven composition."""

LOSS_REGISTRY = {}


def register_loss(name: str):
    """Decorator to register a loss class in the global registry."""
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def build_loss(loss_cfg, **kwargs):
    """Instantiate a loss from its config block.

    Extra kwargs (e.g. num_classes, embed_dim, num_train_cams) are
    forwarded to the loss constructor for losses that need them.
    """
    loss_type = loss_cfg.type
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
    cls = LOSS_REGISTRY[loss_type]
    return cls(loss_cfg, **kwargs)
