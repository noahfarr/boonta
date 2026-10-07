import argparse

from PIL import Image

from boonta.environments.peanut_gb import RAM_BASE, ROMS, START_STATES, Pool
from boonta.environments.peanut_gb.pokemon_red import MAP, X, Y

NOOP, UP, DOWN, LEFT, RIGHT, A, B, START = range(8)
OPTIONS = 0xD355
BRISK = 0x81


def moves(pool):
    if pool.read(MAP) == 0:
        return False
    for direction in (LEFT, RIGHT, DOWN, UP):
        before = (pool.read(X), pool.read(Y))
        pool.press(direction, hold=24, frames=32)
        if (pool.read(X), pool.read(Y)) != before:
            return True
    return False


def seat(pool, state):
    _, ram = pool.observe()
    found = state.tobytes().find(ram.tobytes())
    if found < 0:
        raise RuntimeError("could not locate wram inside the emulator state")
    return found


def brisken(pool, state):
    state[seat(pool, state) + OPTIONS - RAM_BASE] = BRISK
    return state


def opening(pool, out, snapshots=None):
    for _ in range(60):
        pool.press(NOOP, hold=0, frames=24)
    for _ in range(20):
        pool.press(START, hold=4, frames=40)
    for i in range(180):
        pool.press(A, hold=5, frames=15)
        if snapshots and i % 30 == 0:
            Image.fromarray(pool.screen()).save(f"{snapshots}/dialog_{i:03d}.png")
    for attempt in range(30):
        for _ in range(6):
            pool.press(B, hold=4, frames=12)
        if moves(pool):
            for _ in range(3):
                pool.press(B, hold=4, frames=12)
            state = brisken(pool, pool.save())
            state.tofile(out)
            if snapshots:
                Image.fromarray(pool.screen()).save(f"{snapshots}/opening.png")
            print(f"wrote {out}: map {pool.read(MAP)} after {attempt} control attempts, {state.size} bytes")
            return True
        for _ in range(8):
            pool.press(A, hold=5, frames=15)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(START_STATES / "pokemon_red.bin"))
    parser.add_argument("--snapshots", default=None)
    args = parser.parse_args()
    pool = Pool(ROMS / "pokemon_red.gb", num_envs=1, num_threads=1)
    if not opening(pool, args.out, args.snapshots):
        Image.fromarray(pool.screen()).save((args.snapshots or ".") + "/stuck.png")
        raise SystemExit("could not reach a controllable state")


if __name__ == "__main__":
    main()
