import jax

from .scan import associative, pallas, sequential

IMPLEMENTATIONS = {
    "associative": associative,
    "pallas": pallas,
    "sequential": sequential,
}


def get_scan_implementation(name: str | None = None):
    if name is None:
        return pallas if accelerated() else associative
    if name not in IMPLEMENTATIONS:
        raise KeyError(
            f"unknown scan implementation {name!r}, have {sorted(IMPLEMENTATIONS)}"
        )
    return IMPLEMENTATIONS[name]


def accelerated() -> bool:
    return jax.default_backend() == "gpu" and any(
        "nvidia" in device.device_kind.lower()
        and float(device.compute_capability) >= 9.0
        for device in jax.local_devices()
    )


__all__ = ["get_scan_implementation", "associative", "pallas", "sequential"]
