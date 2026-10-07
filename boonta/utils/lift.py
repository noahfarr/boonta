from collections.abc import Callable
from typing import Any

import flax.linen as nn


def vmap(
    target: type[nn.Module],
    *,
    variable_axes: dict,
    split_rngs: dict,
    in_axes: Any = 0,
    out_axes: Any = 0,
    axis_size: int | None = None,
) -> Callable[..., nn.Module]:

    def build(*target_args, **target_kwargs) -> nn.Module:
        target_instance = target(*target_args, **target_kwargs)

        class Vmapped(nn.Module):
            target: nn.Module

            def setup(self):
                nn.share_scope(self, self.target)

            def __call__(self, *args, **kwargs):
                keys = tuple(kwargs)
                extended_args = args + tuple(kwargs[k] for k in keys)
                base_in_axes = (
                    in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
                )
                extended_in_axes = base_in_axes + (None,) * len(keys)

                def body(mdl, *a):
                    n = len(keys)
                    call_args = a[: len(a) - n] if n else a
                    kwarg_values = a[len(a) - n :] if n else ()
                    return mdl(*call_args, **dict(zip(keys, kwarg_values)))

                lifted = nn.vmap(
                    body,
                    variable_axes=variable_axes,
                    split_rngs=split_rngs,
                    in_axes=extended_in_axes,
                    out_axes=out_axes,
                    axis_size=axis_size,
                )
                return lifted(self.target, *extended_args)

        return Vmapped(target_instance)

    return build
