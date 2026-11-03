"""Persistent, shared USD reservations; unknown charges block further requests."""
import json
import math
from pathlib import Path
import sqlite3
import uuid


class BudgetStop(RuntimeError):
    pass


def microdollars(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("USD values must be finite and nonnegative")
    return math.ceil(value * 1_000_000)


class ApiBudget:
    """One ledger for every model/stage in a study, including interrupted calls."""

    def __init__(self, path, limit_usd=60):
        limit = microdollars(limit_usd)
        if not 0 < limit <= 60_000_000:
            raise ValueError("This study's shared cap must be at most USD 60")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS budget (cap INTEGER)")
            self.db.execute("CREATE TABLE IF NOT EXISTS stages (name TEXT PRIMARY KEY, cap INTEGER)")
            self.db.execute("CREATE TABLE IF NOT EXISTS sealed (id INTEGER PRIMARY KEY)")
            self.db.execute("""CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, key TEXT UNIQUE, stage TEXT, bound INTEGER,
                charge INTEGER, response TEXT)""")
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT cap FROM budget").fetchone()
            if row is None:
                self.db.execute("INSERT INTO budget VALUES (?)", (limit,))
            elif row[0] != limit:
                raise ValueError("Shared budget cap changed; reuse the original cap")

    def reserve(self, key, bound_usd, *, stage="paper", stage_limit_usd=60):
        bound = microdollars(bound_usd)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute(
                "SELECT response FROM calls WHERE key=?", (key,)).fetchone()
            if old is not None:
                if old[0] is None:
                    raise BudgetStop("Prior request has unknown cost; reconcile before retrying")
                return None, json.loads(old[0])
            if self.db.execute("SELECT 1 FROM sealed").fetchone():
                raise BudgetStop("This model allocation is sealed")
            if self.db.execute("SELECT 1 FROM calls WHERE charge IS NULL").fetchone():
                raise BudgetStop("An unresolved request blocks new spending")
            used = self.db.execute("SELECT COALESCE(SUM(charge),0) FROM calls").fetchone()[0]
            cap = self.db.execute("SELECT cap FROM budget").fetchone()[0]
            stage_cap = microdollars(stage_limit_usd)
            old_cap = self.db.execute("SELECT cap FROM stages WHERE name=?", (stage,)).fetchone()
            if old_cap is not None and old_cap[0] != stage_cap:
                raise ValueError("Stage cap changed")
            self.db.execute("INSERT OR IGNORE INTO stages VALUES (?,?)", (stage,stage_cap))
            stage_used = self.db.execute("SELECT COALESCE(SUM(charge),0) FROM calls WHERE stage=?", (stage,)).fetchone()[0]
            if used + bound > cap or stage_used + bound > stage_cap:
                raise BudgetStop("Next request could exceed the study or stage budget")
            ticket = uuid.uuid4().hex
            self.db.execute("INSERT INTO calls VALUES (?,?,?,?,NULL,NULL)", (ticket,key,stage,bound))
        return ticket, None

    def settle(self, ticket, charge_usd, response):
        charge = microdollars(charge_usd)
        with self.db:
            row = self.db.execute("SELECT bound,charge FROM calls WHERE id=?", (ticket,)).fetchone()
            if row is None or row[1] is not None:
                raise ValueError("Reservation missing or already settled")
            # Leave the reservation unresolved on any unaccounted rate/usage mismatch.
            if charge > row[0]:
                raise BudgetStop("Reported usage exceeded reservation; reconcile billing")
            self.db.execute("UPDATE calls SET charge=?,response=? WHERE id=?",
                            (charge,json.dumps(response),ticket))

    def status(self):
        used, pending, count = self.db.execute(
            "SELECT COALESCE(SUM(charge),0),COALESCE(SUM(CASE WHEN charge IS NULL THEN bound ELSE 0 END),0),COUNT(*) FROM calls"
        ).fetchone()
        cap = self.db.execute("SELECT cap FROM budget").fetchone()[0]
        return {"cap_usd":cap/1e6,"accounted_upper_usd":used/1e6,
                "unresolved_upper_usd":pending/1e6,"requests":count}

    def allocate(self, name, child_path, limit_usd):
        """Prefund a child ledger atomically before its worker can spend."""
        bound=microdollars(limit_usd)
        child_path=Path(child_path).resolve()
        key="allocation/"+name
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old=self.db.execute("SELECT response,bound FROM calls WHERE key=?", (key,)).fetchone()
            if old is not None:
                descriptor=json.loads(old[0])
                if descriptor["ledger"]!=str(child_path) or old[1]!=bound:
                    raise ValueError("Allocation configuration changed")
                if descriptor["state"]=="closed":
                    return None
            else:
                if child_path.exists():
                    raise ValueError("An unfunded child ledger already exists")
                used=self.db.execute("SELECT COALESCE(SUM(COALESCE(charge,bound)),0) FROM calls").fetchone()[0]
                cap=self.db.execute("SELECT cap FROM budget").fetchone()[0]
                if used+bound>cap:
                    raise BudgetStop("Insufficient shared budget to prefund model")
                descriptor={"kind":"allocation","ledger":str(child_path),"state":"funded"}
                self.db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?)",
                                (uuid.uuid4().hex,key,"allocation",bound,bound,json.dumps(descriptor)))
        return ApiBudget(child_path,limit_usd)

    def close_allocation(self, name, child_path):
        """Seal before refunding: the child can never spend returned funds."""
        key="allocation/"+name
        child_path=Path(child_path).resolve()
        old=self.db.execute("SELECT response,bound FROM calls WHERE key=?", (key,)).fetchone()
        if old is None or json.loads(old[0])["ledger"]!=str(child_path):
            raise ValueError("Allocation provenance differs")
        with sqlite3.connect(child_path,timeout=30) as child:
            child.execute("BEGIN IMMEDIATE")
            child.execute("INSERT OR IGNORE INTO sealed VALUES (1)")
            used=child.execute("SELECT COALESCE(SUM(COALESCE(charge,bound)),0) FROM calls").fetchone()[0]
            if used>old[1]:
                raise BudgetStop("Child exceeded prefunded amount")
            # Also make the old configured cap fail on an accidental restart.
            child.execute("UPDATE budget SET cap=?", (used,))
        descriptor={"kind":"allocation","ledger":str(child_path),"state":"closed",
                    "accounted_upper_usd":used/1e6}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("UPDATE calls SET charge=?,response=? WHERE key=?",
                            (used,json.dumps(descriptor),key))
        return descriptor

    def close(self):
        self.db.close()
