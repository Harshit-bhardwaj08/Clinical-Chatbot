"""
MediChat Authentication: A simple but secure user management layer.

We store user profiles in a local 'users.json' file and protect passwords
using PBKDF2-HMAC-SHA256 with individual salts.

On the very first run, we'll automatically create an 'admin' account 
(password: admin123) so you can get started immediately. You can 
update this later by editing the JSON file directly.
"""

import hashlib
import json
import secrets
from pathlib import Path

from src.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"
_PBKDF2_ITERATIONS = 200_000   # NIST SP 800-132 recommendation


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hex digest of *password* using *salt*."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    ).hex()


def _load_users() -> dict:
    """
    Load all user records from users.json.

    If the file does not exist, create it with a single default admin account
    so the application starts cleanly without manual setup.
    """
    if not _USERS_FILE.exists():
        log.info("users.json not found — creating default admin account.")
        salt = secrets.token_hex(16)
        default_users = {
            "admin": {
                "password_hash": _hash_password("admin123", salt),
                "salt": salt,
                "display_name": "Admin",
                "role": "admin",
            }
        }
        _USERS_FILE.write_text(json.dumps(default_users, indent=2), encoding="utf-8")
        log.warning(
            "Default account created (admin / admin123). "
            "Update users.json before sharing this deployment."
        )
        return default_users

    try:
        return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("users.json is malformed: %s", exc)
        return {}


def _save_users(users: dict) -> None:
    """Persist the users dict to users.json."""
    _USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


# ── Public API ────────────────────────────────────────────────────────────────

def verify_credentials(username: str, password: str) -> tuple[bool, dict]:
    """
    Verify a username / password pair.

    Args:
        username: Plain-text username (case-insensitive).
        password: Plain-text password (never stored or logged).

    Returns:
        ``(True, user_record)``  on success.
        ``(False, {})``          on failure.
    """
    if not username or not password:
        return False, {}

    users = _load_users()
    key = username.strip().lower()
    user = users.get(key)

    if not user:
        return False, {}

    expected = _hash_password(password, user["salt"])

    # secrets.compare_digest prevents timing-based enumeration attacks.
    if secrets.compare_digest(expected, user["password_hash"]):
        return True, user

    return False, {}


def user_exists(username: str) -> bool:
    """Check if a username already exists in users.json (case-insensitive)."""
    users = _load_users()
    return username.strip().lower() in users


def add_user(username: str, password: str, display_name: str = "", role: str = "user") -> None:
    """
    Add or update a user record. 
    
    This is mostly used for initial setup or admin scripts.
    """
    users = _load_users()
    salt = secrets.token_hex(16)
    users[username.strip().lower()] = {
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "display_name": display_name or username,
        "role": role,
    }
    _save_users(users)
    log.info("User '%s' saved to users.json.", username)
