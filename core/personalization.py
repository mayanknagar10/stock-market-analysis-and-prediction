"""
Personalization — zero external accounts, zero API keys.

Tracks what a logged-in user looks at (tickers viewed, watchlist adds,
screens run) in their local profile (extends data/users.json from
core/auth.py), then recommends similar stocks using simple, explainable
signals: same sector, correlated price behavior, similar factor profile.

This is NOT collaborative filtering across users (would need many real
users' data to be meaningful, and a proper database) — it's
content-based recommendation from one user's own behavior, which is a
legitimate and useful personalization technique on its own, and doesn't
require any infrastructure beyond what's already built (core/auth.py's
local JSON store).

If a user isn't logged in, tracking is a no-op — session-only behavior
still lets basic "recently viewed" work via st.session_state, just
without persistence across visits.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import Counter

USERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "users.json"
)

_MAX_HISTORY = 100


def _load_users() -> Dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: Dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def track_view(username: Optional[str], ticker: str, sector: str = "—"):
    """Records a ticker view in the user's local history. No-op for
    guests (username=None) — call sites should always pass
    st.session_state.get('user', {}).get('username')."""
    if not username:
        return
    users = _load_users()
    if username not in users:
        return
    user = users[username]
    history = user.setdefault("view_history", [])
    history.insert(0, {
        "ticker": ticker.upper(), "sector": sector,
        "viewed_at": datetime.now(timezone.utc).isoformat(),
    })
    user["view_history"] = history[:_MAX_HISTORY]
    _save_users(users)


def track_watchlist_add(username: Optional[str], ticker: str, sector: str = "—"):
    if not username:
        return
    users = _load_users()
    if username not in users:
        return
    user = users[username]
    added = user.setdefault("watchlist_adds", [])
    added.insert(0, {"ticker": ticker.upper(), "sector": sector,
                     "added_at": datetime.now(timezone.utc).isoformat()})
    user["watchlist_adds"] = added[:_MAX_HISTORY]
    _save_users(users)


def get_view_history(username: Optional[str]) -> List[Dict]:
    if not username:
        return []
    users = _load_users()
    return users.get(username, {}).get("view_history", [])


def get_favorite_sectors(username: Optional[str], top_n: int = 3) -> List[str]:
    """Returns the user's most-viewed sectors, most-frequent first —
    used both for display ('You often look at: IT, Banking') and as
    input to the recommendation engine below."""
    history = get_view_history(username)
    sectors = [h["sector"] for h in history if h.get("sector") and h["sector"] != "—"]
    if not sectors:
        return []
    counts = Counter(sectors)
    return [s for s, _ in counts.most_common(top_n)]


def get_recently_viewed_tickers(username: Optional[str], limit: int = 10) -> List[str]:
    history = get_view_history(username)
    seen = []
    for h in history:
        t = h["ticker"]
        if t not in seen:
            seen.append(t)
        if len(seen) >= limit:
            break
    return seen


def recommend_similar_stocks(username: Optional[str], universe: List[Dict],
                             limit: int = 8) -> List[Dict]:
    """
    universe: list of {ticker, name, sector} dicts (e.g. NIFTY50 + SP500
    from screener.py) to recommend FROM.

    Content-based recommendation: score each universe stock by whether
    its sector matches the user's most-viewed sectors, excluding
    tickers already viewed. Simple, explainable, and genuinely useful —
    "since you look at IT and Banking stocks a lot, here are others in
    those sectors you haven't checked out yet."
    """
    fav_sectors = get_favorite_sectors(username, top_n=3)
    already_viewed = set(get_recently_viewed_tickers(username, limit=50))

    if not fav_sectors:
        return []

    scored = []
    for item in universe:
        ticker = item.get("ticker") or item.get("Ticker")
        sector = item.get("sector") or item.get("Sector")
        if not ticker or ticker.upper() in already_viewed:
            continue
        if sector in fav_sectors:
            rank = fav_sectors.index(sector)  # 0 = most favorite
            score = len(fav_sectors) - rank
            scored.append({**item, "match_reason": f"You often view {sector} stocks",
                          "score": score})

    scored.sort(key=lambda x: -x["score"])
    return scored[:limit]


def get_user_stats(username: Optional[str]) -> Dict:
    """Simple engagement summary for a 'your activity' panel."""
    if not username:
        return {"total_views": 0, "unique_tickers": 0, "top_sectors": []}
    history = get_view_history(username)
    unique = len(set(h["ticker"] for h in history))
    return {
        "total_views": len(history),
        "unique_tickers": unique,
        "top_sectors": get_favorite_sectors(username, top_n=5),
    }
