"""JSON-lines process boundary. Pixel bytes are base64 encoded for transport.

Usage: python -m resolveai.server PRIVATE_CASE ASSET_ROOT [--budget 12]
The policy receives stdin/stdout only, never filesystem access to private assets.
Deploy under a separate restricted account/container for adversarial policies.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

from .environment import Environment


def encode(value):
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--budget", type=int, default=12)
    args = parser.parse_args()
    env = Environment(json.loads(args.case.read_text()), args.assets, args.budget)
    branches = {"root": env}
    print(json.dumps(env.observation(), default=encode), flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            branch = branches[command.get("branch", "root")]
            if command.get("type") == "fork":
                name = command["new_branch"]
                if name in branches:
                    raise ValueError("branch already exists")
                branches[name] = branch.fork()
                result = {"branch": name, "observation": branches[name].observation()}
            else:
                result = branch.step(command)
        except (ValueError, KeyError, TypeError):
            # Do not leak exception details containing private paths or IDs.
            result = {"error": "invalid_action"}
        print(json.dumps(result, default=encode), flush=True)


if __name__ == "__main__":
    main()
