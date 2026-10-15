import importlib
import sys


def main():
    command = sys.argv.pop(1)
    importlib.import_module(f"hangar.{command}").main()
