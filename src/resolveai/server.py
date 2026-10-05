"""Private JSON-lines environment service; annotation/evaluation never reaches policy.

--conversation uses native function calls and persistent assistant/tool messages.
--allow-forks enables trusted trainer control only; policy tools never include fork.
"""
import argparse
import base64
import json
from pathlib import Path
import sys

from .conversation import Conversation
from .environment import Environment
from .tools import ActionError


def encode(value):
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--grounding-root", type=Path)
    parser.add_argument("--conversation", action="store_true")
    parser.add_argument("--allow-forks", action="store_true")
    args = parser.parse_args()
    from .grounding_tools import FrozenGrounding
    grounding = FrozenGrounding(args.grounding_root) if args.grounding_root else None
    env = Environment(json.loads(args.case.read_text()), args.assets, args.budget, grounding_backend=grounding)
    state = Conversation(env) if args.conversation else env
    branches = {"root": state}
    print(json.dumps(state.public() if args.conversation else env.observation(), default=encode), flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ActionError("invalid_arguments")
            branch = branches[command.get("branch", "root")]
            if command.get("type") == "fork":
                name = command["new_branch"]
                if not args.allow_forks or not isinstance(name, str) or name in branches or len(branches) >= 32:
                    raise ActionError("fork_not_allowed")
                branches[name] = branch.fork()
                result = {"branch": name, "observation": branches[name].public() if args.conversation else branches[name].observation()}
            elif args.conversation:
                result = branch.call(command)
            else:
                result = branch.step({k: v for k, v in command.items() if k != "branch"})
        except ActionError as error:
            result = {"error": error.code}
        except (ValueError, KeyError, TypeError, OSError, RuntimeError):
            result = {"error": "invalid_action"}
        print(json.dumps(result, default=encode), flush=True)


if __name__ == "__main__":
    main()
