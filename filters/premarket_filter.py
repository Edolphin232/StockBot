# filters/premarket_filter.py
"""
Pre-market filter — uses calculators, no data fetching.
"""
from dataclasses import dataclass, field
from calculators.gap import calculate_gap, GapResult


@dataclass
class PremarketResult:
    gap: GapResult = None
    vix_level: float = 0.0
    vix_in_range: bool = False
    has_catalyst: bool = False
    catalyst_name: str = ""
    score: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def pass_filter(self) -> bool:
        return self.gap.gap_qualifies or self.has_catalyst


def check_vix(
    vix_level: float,
    vix_min: float = 14.0,
    vix_max: float = 35.0,
) -> dict:
    in_range = vix_min <= vix_level <= vix_max
    if vix_level < vix_min:
        note = "Low VIX — likely chop"
    elif vix_level > vix_max:
        note = "High VIX — unpredictable"
    else:
        note = "VIX in trend-friendly range"
    return {"in_range": in_range, "note": note}


def check_catalyst(
    date: str,
    events: list = None,
    keywords: list = None,
) -> dict:
    """
    Check if today has a high-impact catalyst.

    Args:
        date:     "YYYY-MM-DD"
        events:   [{"date": "2025-02-17", "event": "CPI"}]
        keywords: High impact event keywords to match against
    """
    if not events:
        return {"has_catalyst": False, "catalyst_name": ""}

    keywords = keywords or ["FOMC", "CPI", "NFP", "GDP", "Fed", "jobs"]

    for event in events:
        if event.get("date") == date:
            name = event.get("event", "")
            if any(kw.lower() in name.lower() for kw in keywords):
                return {"has_catalyst": True, "catalyst_name": name}

    return {"has_catalyst": False, "catalyst_name": ""}


def run_premarket_filter(
    current_open: float,
    prev_close: float,
    vix_level: float,
    date: str,
    catalyst_events: list = None,
    # Scoring weights
    score_gap: float = 1.0,
    score_vix: float = 1.0,
    score_catalyst: float = 1.5,
    # Gap thresholds
    min_gap_pct: float = 0.1,
    max_gap_pct: float = 3.0,
    # VIX thresholds
    vix_min: float = 14.0,
    vix_max: float = 35.0,
) -> PremarketResult:
    gap    = calculate_gap(current_open, prev_close, min_gap_pct, max_gap_pct)
    vix    = check_vix(vix_level, vix_min, vix_max)
    cat    = check_catalyst(date, catalyst_events)

    score = 0.0
    if gap.gap_qualifies:
        score += score_gap
    if vix["in_range"]:
        score += score_vix
    if cat["has_catalyst"]:
        score += score_catalyst

    return PremarketResult(
        gap=gap,
        vix_level=vix_level,
        vix_in_range=vix["in_range"],
        has_catalyst=cat["has_catalyst"],
        catalyst_name=cat["catalyst_name"],
        score=score,
        details={"vix": vix, "catalyst": cat},
    )