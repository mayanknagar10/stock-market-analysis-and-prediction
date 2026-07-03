"""
Local authentication — zero external accounts, zero API keys.

Uses Python's built-in `hashlib.pbkdf2_hmac` for password hashing (no
third-party crypto library needed) and a local JSON file as the user
store. This is a genuine login system — passwords are salted, hashed,
never stored in plaintext — just self-contained instead of depending on
Clerk/Supabase/Auth0.

IMPORTANT — persistence on Streamlit Cloud:
  Streamlit Cloud's free tier has an EPHEMERAL filesystem: any user who
  self-registers via the Sign Up tab is stored in data/users.json, but
  that file is wiped on every redeploy/restart. Two ways to handle this:

  1. (Quick) Pre-seed accounts by editing data/users.json yourself and
     committing it to git — those accounts persist forever across
     redeploys, exactly like the model checkpoint in models/.
  2. (Durable) When you're ready, swap USERS_FILE-based storage for a
     real database (Supabase/Neon Postgres — see the roadmap). The
     verify_login() / create_user() functions below are the only two
     functions that would need to change; everything else (session
     handling, UI) stays identical.

For local development or a self-hosted deployment (Docker, your own
server), the filesystem is NOT ephemeral, so self-registration works
normally with no caveats.
"""

import os
import json
import hashlib
import secrets
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

USERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "users.json"
)

_PBKDF2_ITERATIONS = 260_000  # OWASP-recommended minimum as of 2023+
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,20}$")


# ─────────────────────────────────────────────────────────────────
# PASSWORD HASHING — stdlib only, no bcrypt/passlib dependency
# ─────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Returns 'salt_hex$hash_hex' — self-contained, no separate salt storage needed."""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
        return secrets.compare_digest(dk, expected)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# USER STORE — local JSON file
# ─────────────────────────────────────────────────────────────────

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


def _seed_demo_account():
    """Ensures a demo/demo account always exists so the login screen is
    never a dead end, even before anyone has registered."""
    users = _load_users()
    if "demo" not in users:
        users["demo"] = {
            "password_hash": _hash_password("demo1234"),
            "name": "Demo User",
            "role": "Free tier",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_users(users)


def validate_username(username: str) -> Tuple[bool, str]:
    if not _USERNAME_RE.match(username):
        return False, "3-20 characters: letters, numbers, underscore only."
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    if len(password) < 8:
        return False, "At least 8 characters."
    return True, ""


def create_user(username: str, password: str, name: str,
                role: str = "Free tier") -> Tuple[bool, str]:
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
    ok, msg = validate_password(password)
    if not ok:
        return False, msg

    users = _load_users()
    if username in users:
        return False, "That username is already taken."

    users[username] = {
        "password_hash": _hash_password(password),
        "name": name.strip() or username,
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    return True, "Account created."


def verify_login(username: str, password: str) -> Optional[Dict]:
    """Returns the user dict (without password_hash) on success, else None."""
    users = _load_users()
    entry = users.get(username)
    if not entry:
        return None
    if not _verify_password(password, entry["password_hash"]):
        return None
    return {"username": username, "name": entry["name"], "role": entry["role"]}


def change_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    users = _load_users()
    entry = users.get(username)
    if not entry or not _verify_password(old_password, entry["password_hash"]):
        return False, "Current password is incorrect."
    ok, msg = validate_password(new_password)
    if not ok:
        return False, msg
    entry["password_hash"] = _hash_password(new_password)
    _save_users(users)
    return True, "Password updated."


def user_count() -> int:
    return len(_load_users())


# Seed the demo account at import time so login always has something to try
_seed_demo_account()
