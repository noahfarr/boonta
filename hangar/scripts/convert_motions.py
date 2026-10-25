import sys

from hangar.recipes.super_mimic_mjlab import convert_motions


def main():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

    directory, *_ = sys.argv[1:]
    print(convert_motions(directory, str(G1_XML)))


if __name__ == "__main__":
    main()
