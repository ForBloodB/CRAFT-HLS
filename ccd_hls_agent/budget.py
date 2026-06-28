from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .utils import utc_now


BudgetKind = str
UNLIMITED: int | None = None


@dataclass
class BudgetCounter:
    limit: int | None = UNLIMITED
    used: int = 0

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    def can_consume(self, amount: int = 1) -> bool:
        if amount < 0:
            raise ValueError("budget amount must be non-negative")
        return self.limit is None or self.used + amount <= self.limit

    def consume(self, amount: int = 1) -> None:
        if not self.can_consume(amount):
            raise RuntimeError("budget exhausted")
        self.used += amount

    def summary(self) -> dict[str, Any]:
        return {"used": self.used, "limit": self.limit, "remaining": self.remaining}


@dataclass
class BudgetLedger:
    llm_calls: BudgetCounter = field(default_factory=BudgetCounter)
    csim_calls: BudgetCounter = field(default_factory=BudgetCounter)
    synth_calls: BudgetCounter = field(default_factory=BudgetCounter)
    cosim_calls: BudgetCounter = field(default_factory=lambda: BudgetCounter(limit=0))
    static_calls: BudgetCounter = field(default_factory=BudgetCounter)
    unified_credits: BudgetCounter = field(default_factory=BudgetCounter)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_limits(
        cls,
        *,
        llm_calls: int | None,
        csim_calls: int | None = None,
        synth_calls: int | None = None,
        cosim_calls: int | None = 0,
        static_calls: int | None = None,
        unified_credits: int | None = None,
    ) -> "BudgetLedger":
        return cls(
            llm_calls=BudgetCounter(llm_calls),
            csim_calls=BudgetCounter(csim_calls),
            synth_calls=BudgetCounter(synth_calls),
            cosim_calls=BudgetCounter(cosim_calls),
            static_calls=BudgetCounter(static_calls),
            unified_credits=BudgetCounter(unified_credits),
        )

    def _counter(self, kind: BudgetKind) -> BudgetCounter:
        try:
            return getattr(self, kind)
        except AttributeError as exc:
            raise KeyError(f"unknown budget kind: {kind}") from exc

    def can_consume(self, kind: BudgetKind, amount: int = 1, *, unified_amount: int = 1) -> bool:
        return self._counter(kind).can_consume(amount) and self.unified_credits.can_consume(unified_amount)

    def consume(self, kind: BudgetKind, *, stage: str, label: str = "", amount: int = 1, unified_amount: int = 1) -> bool:
        ok = self.can_consume(kind, amount, unified_amount=unified_amount)
        event = {
            "created_at": utc_now(),
            "kind": kind,
            "stage": stage,
            "label": label,
            "amount": amount,
            "unified_amount": unified_amount,
            "accepted": ok,
        }
        if ok:
            self._counter(kind).consume(amount)
            self.unified_credits.consume(unified_amount)
        else:
            event["reason"] = f"budget_exhausted_{kind.removesuffix('_calls')}"
        self.events.append(event)
        return ok

    def exhausted_reason(self, kind: BudgetKind) -> str:
        return f"budget_exhausted_{kind.removesuffix('_calls')}"

    def summary(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls.summary(),
            "csim_calls": self.csim_calls.summary(),
            "synth_calls": self.synth_calls.summary(),
            "cosim_calls": self.cosim_calls.summary(),
            "static_calls": self.static_calls.summary(),
            "unified_credits": self.unified_credits.summary(),
        }

    def model_dump(self) -> dict[str, Any]:
        return {"summary": self.summary(), "events": self.events}
