import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

SKIP = ("hydra", "loggers", "artisans", "search_space")


def shapes(params) -> list[tuple[str, tuple[int, ...], int]]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(key, "key", key)) for key in path)
        leaf = np.asarray(leaf)
        rows.append((name.removeprefix("params/"), leaf.shape, leaf.size))
    return rows


def nodes(rows) -> list[tuple[int, str, str, int]]:
    totals, order = {}, []
    for name, shape, size in rows:
        parts = name.split("/")
        for depth in range(len(parts)):
            path = "/".join(parts[: depth + 1])
            if path not in totals:
                totals[path] = 0
                order.append(path)
            totals[path] += size

    leaves = {name: str(shape) for name, shape, _ in rows}
    children = {}
    for path in order:
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        children.setdefault(parent, []).append(path)

    cache = {}

    def signature(path: str) -> tuple:
        if path not in cache:
            cache[path] = tuple(
                sorted(
                    (other[len(path) :], leaves.get(other, ""), totals[other])
                    for other in totals
                    if other == path or other.startswith(f"{path}/")
                )
            )
        return cache[path]

    def replica(left: str, right: str) -> bool:
        stem = left.split("/")[-1].rstrip("0123456789")
        return stem == right.split("/")[-1].rstrip("0123456789") and (
            signature(left) == signature(right)
        )

    rendered = []

    def walk(path: str, depth: int, repeat: int) -> None:
        label, kids = path.split("/")[-1], children.get(path, [])
        while len(kids) == 1 and path not in leaves:
            path = kids[0]
            label = f"{label}/{path.split('/')[-1]}"
            kids = children.get(path, [])
        if repeat > 1:
            label = f"{label}  ×{repeat}"
        rendered.append((depth, label, leaves.get(path, ""), totals[path] * repeat))

        index = 0
        while index < len(kids):
            same = 1
            while index + same < len(kids) and replica(kids[index], kids[index + same]):
                same += 1
            walk(kids[index], depth + 1, same)
            index += same

    roots, index = children.get("", []), 0
    while index < len(roots):
        same = 1
        while index + same < len(roots) and replica(roots[index], roots[index + same]):
            same += 1
        walk(roots[index], 0, same)
        index += same
    return rendered


def span(widths) -> int:
    return sum(width + 2 for width in widths) + len(widths) + 1


def borders(widths) -> set[int]:
    positions, offset = set(), 1
    for width in widths[:-1]:
        offset += width + 2
        positions.add(offset)
        offset += 1
    return positions


def seam(above, below, width, left, right, fill="─", down="┬", cross="┼", up="┴"):
    marks = {index: up for index in above}
    marks.update({index: cross if index in above else down for index in below})
    return left + "".join(marks.get(index, fill) for index in range(1, width - 1)) + right


def sizes(head, body, footer) -> list[int]:
    rows = body + ([footer] if footer else [])
    return [max(len(row[i]) for row in rows + [head]) for i in range(len(head))]


def panel(head, body, footer, aligned, minimum: int = 0) -> list[str]:
    widths = sizes(head, body, footer)
    widths[-1] += max(0, minimum - span(widths))
    width, edges = span(widths), borders(widths)

    def line(cells, edge="│"):
        packed = [
            cell.rjust(size) if index in aligned else cell.ljust(size)
            for index, (cell, size) in enumerate(zip(cells, widths))
        ]
        return f"{edge} " + f" {edge} ".join(packed) + f" {edge}"

    lines = [
        seam(set(), edges, width, "┏", "┓", "━", "┳"),
        line(head, "┃"),
        seam(edges, edges, width, "┡", "┩", "━", cross="╇"),
    ]
    lines.extend(line(row) for row in body)
    if footer:
        lines.append(seam(edges, edges, width, "├", "┤"))
        lines.append(line(footer))
    lines.append(seam(edges, set(), width, "└", "┘"))
    return lines


def table(rows, title="module"):
    body = [
        (f"{'  ' * depth}{name}", shape, f"{size:,}")
        for depth, name, shape, size in nodes(rows)
    ]
    total = sum(size for _, _, size in rows)
    return (title, "shape", "params"), body, ("total", "", f"{total:,}"), {1, 2}


def networks(algorithm_state) -> dict:
    held = {"module": getattr(algorithm_state, "params", algorithm_state)}
    for name, value in getattr(algorithm_state, "__dict__", {}).items():
        if name.endswith("_params"):
            held[name] = value
    return held


def flatten(cfg, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for key, value in cfg.items():
        if key in SKIP or str(key).startswith("_"):
            continue
        name = f"{prefix}{key}"
        if isinstance(value, DictConfig):
            rows.extend(flatten(value, f"{name}."))
        elif not OmegaConf.is_list(value):
            rows.append((name, str(value)))
    return rows


def prune(rows) -> list[tuple[str, str]]:
    seen = {key: value for key, value in rows}
    return [
        (key, value)
        for key, value in rows
        if not any(
            key.endswith(f".{other}") and seen[other] == value
            for other in seen
            if other != key
        )
    ]


def settings(rows) -> list[str]:
    groups = {"run": []}
    for key, value in rows:
        head, _, tail = key.partition(".")
        groups.setdefault(head if tail else "run", []).append((tail or head, value))

    for name, entries in list(groups.items()):
        if name != "run" and len(entries) == 1:
            groups["run"].append((f"{name}.{entries[0][0]}", entries[0][1]))
            del groups[name]

    body = []
    for name, entries in groups.items():
        body.append((name, ""))
        body.extend((f"  {key}", value) for key, value in entries)
    return ("setting", "value"), body, None, set()


def brief(cfg, state) -> None:
    algorithm_state = getattr(state, "algorithm_state", state)
    sections = [
        table(shapes(params), title)
        for title, params in networks(algorithm_state).items()
    ]
    sections.append(settings(prune(flatten(cfg))))
    width = max(span(sizes(head, body, footer)) for head, body, footer, _ in sections)
    lines = [""]
    for head, body, footer, aligned in sections:
        lines.extend(panel(head, body, footer, aligned, width))
        lines.append("")
    print("\n".join(lines), flush=True)
