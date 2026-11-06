import importlib
import os
import sys

COMMANDS = ("train", "evaluate")


def complete(shell):
    """Answer a Hydra shell-completion query for `boonta <command> <overrides>`.

    The completion script hands Hydra the whole line, command included. Hydra
    would read the command as an override, so drop it, and offer the command
    names while it is still being typed. Returns the command to hand over to,
    or None when there is nothing left for Hydra to complete.
    """
    program, _, rest = os.environ["COMP_LINE"].lstrip().partition(" ")
    command, space, overrides = rest.lstrip().partition(" ")
    if not space:
        print(("\n" if shell == "fish" else " ").join(COMMANDS))
        return None
    if command not in COMMANDS:
        return None
    os.environ["COMP_LINE"] = f"{program} {overrides}"
    return command


def main():
    args = sys.argv[1:]
    if args[:1] in (["-sc"], ["--shell-completion"]):
        # `boonta -sc install=bash` has no command, so use the default one.
        action, _, shell = args[1].partition("=") if len(args) > 1 else ("", "", "")
        command = complete(shell) if action == "query" else COMMANDS[0]
        if command is None:
            return
    else:
        command = sys.argv.pop(1)
    importlib.import_module(f"hangar.{command}").main()
