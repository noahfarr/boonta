import sys

from boonta.environments.mjlab import motions


def main():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

    directory, *_ = sys.argv[1:]
    print(motions.convert(directory, str(G1_XML)))


if __name__ == "__main__":
    main()
