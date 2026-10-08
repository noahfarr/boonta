import sys

from hydra_zen import BuildsFn, ZenStore, make_custom_builds_fn


class Builds(BuildsFn):
    @classmethod
    def _get_obj_path(cls, target):
        parts = [part for part in target.__module__.split(".") if not part.startswith("_")]
        for end in range(1, len(parts) + 1):
            module = ".".join(parts[:end])
            if getattr(sys.modules.get(module), target.__name__, None) is target:
                return f"{module}.{target.__name__}"
        return super()._get_obj_path(target)


def keep(node):
    return node


def place(node, path, **kwargs):
    group, _, name = path.rpartition("/")
    store(node, group=group, name=name, **kwargs)


builds = Builds.builds
fbuilds = make_custom_builds_fn(populate_full_signature=True, builds_fn=Builds)
store = ZenStore(name="hangar")(to_config=keep)
