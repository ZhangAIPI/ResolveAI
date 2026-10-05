"""Trusted fixed-prefix comparisons of evidence-acquisition actions."""
from itertools import combinations
from .conversation import preference_pair
from .rollout import run


def compare_actions(session, candidates, client, *, max_turns=12, max_context_tokens=8192, allow_proxy=False):
    """Run candidates with the same continuation policy, world, prefix and total turn cap.

    Candidates are real native calls (including a policy-proposed finish). This
    helper is not a policy tool. Only public conversations reach the client.
    Greedy model generation gives matched continuations; stochastic clients must
    externally reset to the same seed per branch and record that seed.
    """
    if max_turns < 1 or len(candidates) < 2 or session._env.finished:
        raise ValueError("require a live prefix, at least two actions and a positive turn cap")
    base = session.fork()
    base.messages[0]["content"] += f"\nEpisode limit: {max_turns} assistant turns; finish within this limit.\n"
    released = set(base._env._released)
    fingerprint = base._env._world.fingerprint
    records = []
    for candidate in candidates:
        branch = base.fork()
        branch.call(candidate)
        action_releases = sorted(branch._env._released - released)
        if branch._env.finished:
            record = branch.record(); record["generation"] = []
        elif max_turns == 1:
            branch.termination = "max_turns"; record = branch.record(); record["generation"] = []
        else:
            record = run(branch, client, max_turns-1, max_context_tokens, add_turn_limit=False)
        if branch._env._world.fingerprint != fingerprint:
            raise RuntimeError("candidate changed hidden world state")
        record["branch_audit"] = {"first_action_released": action_releases,
                                  "trajectory_released": sorted(branch._env._released - released),
                                  "max_total_new_turns": max_turns}
        records.append(record)
    pairs = [pair for left, right in combinations(records, 2)
             if (pair := preference_pair(left, right, allow_proxy=allow_proxy)) is not None]
    return {"records": records, "preferences": pairs}
