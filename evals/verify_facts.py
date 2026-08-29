"""Re-verify the frozen task set against the live MCP tools and pinned DB.

A frozen fact is only trustworthy while the snapshot it was captured from is
unchanged. This script (1) checks the DB file's sha256 against the value pinned
in frozen_tasks.jsonl, and (2) calls the exact tool behind each fact and asserts
the fact still appears in the response. Run it before publishing any result; a
mismatch means the snapshot moved and the facts must be recaptured, not that a
model failed.

    NHS_MCP_CWD=/path/to/nhs-intelligence-mcp \
    NHS_INTEL_DB=/path/to/nhs-intelligence-mcp/data/nhs_intel.db \
        uv run python evals/verify_facts.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path

from nhs_care_access_agent.config import AgentSettings
from nhs_care_access_agent.mcp_client import StdioMcpToolClient

PINNED_SHA256 = "0e5eec15a084639822551b6a66c91e6d617dd3eec41d0f25d23eaff087bc99c6"

# (tool, arguments, facts that must appear in the stringified response). One entry
# per fact-bearing case in frozen_tasks.jsonl; abstention cases carry no fact to
# re-verify here and are exercised by the agent evaluation instead.
CHECKS = [
    ("lookup_wait_time", {"provider": "Bolton NHS Foundation Trust", "specialty": "Cardiology"}, ["25"]),
    ("rank_trusts_by_wait", {"specialty": "Cardiology", "region": "London", "limit": 1}, ["Hillingdon", "23"]),
    ("trust_profile", {"identifier": "Isle of Wight NHS Trust", "specialty": "Cardiology", "by_name": True}, ["31", "Good"]),
    ("trust_profile", {"identifier": "RGT", "specialty": "Cardiology"}, ["22"]),
    ("wait_time_trend", {"provider_code": "RBN", "specialty": "Neurosurgical Service"}, ["21", "worsening"]),
]


def _check_checksum(db_path: Path) -> bool:
    if not db_path.exists():
        print(f"DB not found at {db_path}; set NHS_INTEL_DB.", file=sys.stderr)
        return False
    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    if digest != PINNED_SHA256:
        print(f"DB checksum drift:\n  pinned {PINNED_SHA256}\n  actual {digest}", file=sys.stderr)
        return False
    print("DB checksum matches pinned snapshot.")
    return True


async def _check_facts() -> bool:
    settings = AgentSettings.from_env()
    client = StdioMcpToolClient(settings.mcp_command, settings.mcp_args, settings.mcp_cwd)
    ok = True
    async with client as tools:
        for name, arguments, facts in CHECKS:
            response = str(await tools.call_tool(name, arguments)).casefold()
            missing = [fact for fact in facts if not re.search(re.escape(fact.casefold()), response)]
            if missing:
                ok = False
                print(f"FAIL {name}{arguments}: missing {missing}", file=sys.stderr)
            else:
                print(f"ok   {name}: {', '.join(facts)}")
    return ok


def main() -> None:
    db_path = Path(os.environ.get("NHS_INTEL_DB", "")).expanduser()
    checksum_ok = _check_checksum(db_path) if db_path else True
    if not db_path:
        print("NHS_INTEL_DB unset; skipping checksum check.", file=sys.stderr)
    facts_ok = asyncio.run(_check_facts())
    if not (checksum_ok and facts_ok):
        sys.exit(1)
    print("All frozen facts verified.")


if __name__ == "__main__":
    main()
