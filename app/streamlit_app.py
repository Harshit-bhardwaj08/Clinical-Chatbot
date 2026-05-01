"""
MediChat: A high-fidelity clinical assistant designed with a focus on 
safety, medical grounding, and a smooth, ChatGPT-inspired user experience.

This file handles the frontend logic using Streamlit. To get things running,
just use: streamlit run app/streamlit_app.py
"""

import uuid
import time
import re
import json
import requests
import streamlit as st
from datetime import datetime

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediChat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Environment & Path Setup ──
# We need to make sure the project root is in our path so we can import from /src.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import API_URL
from src.auth import (
    verify_credentials,
    add_user,
    user_exists,
    create_session_token,
    validate_session_token,
)

_AUTH_QUERY_KEY = "auth_token"
_CHAT_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "chat_store"


def _safe_username_for_path(username: str) -> str:
    username = (username or "").strip().lower()
    return re.sub(r"[^a-z0-9.-]+", "_", username) or "anonymous"


def _chat_store_path(username: str) -> Path:
    return _CHAT_STORE_DIR / f"{_safe_username_for_path(username)}.json"


def _load_chats_from_disk(username: str) -> tuple[dict, str | None]:
    path = _chat_store_path(username)
    if not path.exists():
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chats = payload.get("chats") or {}
        current_chat = payload.get("current_chat")
        if not isinstance(chats, dict):
            return {}, None
        if current_chat and current_chat not in chats:
            current_chat = None
        return chats, current_chat
    except Exception:
        return {}, None


def _save_chats_to_disk(username: str, chats: dict, current_chat: str | None) -> None:
    _CHAT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "chats": chats,
        "current_chat": current_chat,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    _chat_store_path(username).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _persist_chats_if_authenticated() -> None:
    if not st.session_state.get("authenticated", False):
        return
    username = st.session_state.get("current_user", {}).get("username", "")
    if not username:
        return
    _save_chats_to_disk(
        username,
        st.session_state.get("chats", {}),
        st.session_state.get("current_chat"),
    )


def _get_auth_token_from_url() -> str:
    try:
        return str(st.query_params.get(_AUTH_QUERY_KEY, "")).strip()
    except Exception:
        return ""


def _set_auth_token_in_url(token: str) -> None:
    try:
        if token:
            st.query_params[_AUTH_QUERY_KEY] = token
        elif _AUTH_QUERY_KEY in st.query_params:
            del st.query_params[_AUTH_QUERY_KEY]
    except Exception:
        pass


def _restore_auth_from_url() -> None:
    token = _get_auth_token_from_url()
    if not token:
        return

    ok, user_record, consent_given = validate_session_token(token)
    if not ok:
        _set_auth_token_in_url("")
        return

    st.session_state.authenticated = True
    st.session_state.current_user = user_record
    st.session_state.consent_given = consent_given
    st.session_state.login_error = ""


def _refresh_auth_token(consent_given: bool) -> None:
    username = st.session_state.get("current_user", {}).get("username", "").strip().lower()
    if not username:
        return
    _set_auth_token_in_url(create_session_token(username, consent_given=consent_given))


_SIDEBAR_SEARCH_HTML = """
<div class="mc-search" role="search">
  <span class="mc-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="11" cy="11" r="7.5" stroke="currentColor" stroke-width="2"></circle>
      <line x1="16.6" y1="16.6" x2="21" y2="21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></line>
    </svg>
  </span>
  <input id="mc-input" type="text" autocomplete="off" />
  <button id="mc-clear" type="button" aria-label="Clear search" title="Clear">
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M18 6L6 18M6 6l12 12"></path>
    </svg>
  </button>
</div>
"""

_SIDEBAR_SEARCH_CSS = """
.mc-search{
  width:100%;
  box-sizing:border-box;
  display:flex;
  align-items:center;
  position:relative;
  background:#2f2f36;
  border:1px solid transparent;
  border-radius:12px;
  padding:0;
  margin-top:8px;
}
.mc-search:focus-within{
  border-color: rgba(255,255,255,0.20);
}
.mc-icon{
  position:absolute;
  left:12px;
  display:flex;
  align-items:center;
  justify-content:center;
  width:18px;
  height:18px;
  color:#a8adb7;
  pointer-events:none;
}
.mc-icon svg{ width:18px; height:18px; }
#mc-input{
  width:100%;
  box-sizing:border-box;
  border:none;
  outline:none;
  background:transparent;
  color:#ececec;
  font-size:16px;
  line-height:1.2;
  padding:10px 38px 10px 40px;
  font-family: inherit;
}
#mc-input::placeholder{
  color:#a8adb7;
}
#mc-clear{
  position:absolute;
  right:10px;
  top:50%;
  transform:translateY(-50%);
  width:22px;
  height:22px;
  border:none;
  border-radius:999px;
  background: rgba(255,255,255,0.12);
  color:#ffffff;
  display:flex;
  align-items:center;
  justify-content:center;
  cursor:pointer;
  opacity:0;
  pointer-events:none;
  transition: opacity 0.12s ease, background 0.12s ease;
}
#mc-clear svg{
  width:12px;
  height:12px;
  stroke: currentColor;
  stroke-width: 2.2;
  fill: none;
  stroke-linecap: round;
}
#mc-clear:hover{
  background: rgba(255,255,255,0.22);
  color:#ffffff;
}
"""

_SIDEBAR_SEARCH_JS = """
const _mcInstances = new WeakMap();

export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const input = parentElement.querySelector("#mc-input");
  const clearBtn = parentElement.querySelector("#mc-clear");
  if (!input || !clearBtn) return;

  const placeholder = (data && data.placeholder) ? data.placeholder : "Search chats";
  input.placeholder = placeholder;

  const nextValue = (data && data.value != null) ? String(data.value) : "";
  if (document.activeElement !== input && input.value !== nextValue) {
    input.value = nextValue;
  }

  let inst = _mcInstances.get(parentElement);
  if (!inst) {
    inst = {
      lastSent: nextValue,
      ignoreBlurOnce: false,
    };

    inst.setClearVisibility = () => {
      const has = !!(input.value && input.value.length);
      clearBtn.style.opacity = has ? "1" : "0";
      clearBtn.style.pointerEvents = has ? "auto" : "none";
    };

    inst.applyLocalFilter = (raw) => {
      const term = String(raw ?? "").trim().toLowerCase();
      const sidebar = document.querySelector('section[data-testid="stSidebar"]');
      if (!sidebar) return;
      const hook = sidebar.querySelector(".recent-chats-scroll-hook");
      if (!hook) return;

      const hookContainer = hook.closest('[data-testid="stElementContainer"]');
      const recentsBlock = hookContainer ? hookContainer.parentElement : null;
      if (!recentsBlock) return;

      const rows = recentsBlock.querySelectorAll('[data-testid="stHorizontalBlock"]');
      rows.forEach((row) => {
        const titleBtn = row.querySelector(".stButton button");
        if (!titleBtn) return;
        const txt = String(titleBtn.textContent || "").trim().toLowerCase();
        const show = !term || txt.includes(term);
        row.style.display = show ? "" : "none";
      });
    };

    inst.emit = () => {
      inst.setClearVisibility();
      const val = input.value ?? "";
      if (val === inst.lastSent) return;
      inst.lastSent = val;
      setStateValue("value", val);
    };

    inst.onInput = () => {
      inst.setClearVisibility();
      inst.applyLocalFilter(input.value);
    };

    inst.onKeydown = (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        inst.emit();
      }
    };

    inst.onBlur = () => {
      if (inst.ignoreBlurOnce) {
        inst.ignoreBlurOnce = false;
        return;
      }
      inst.emit();
    };

    // mousedown happens before blur/click; guard blur early to avoid flicker reruns.
    inst.onClearMouseDown = (e) => {
      inst.ignoreBlurOnce = true;
      e.preventDefault();
    };

    inst.onClear = (e) => {
      e.preventDefault();
      input.value = "";
      inst.applyLocalFilter("");
      // Force a single state sync so cleared search survives future reruns.
      inst.lastSent = "__mc_force_clear__";
      inst.emit();
      input.focus();
    };

    input.addEventListener("input", inst.onInput);
    input.addEventListener("keydown", inst.onKeydown);
    input.addEventListener("blur", inst.onBlur);
    clearBtn.addEventListener("mousedown", inst.onClearMouseDown);
    clearBtn.addEventListener("click", inst.onClear);
    _mcInstances.set(parentElement, inst);
  }

  inst.setClearVisibility();
  inst.applyLocalFilter(input.value);

  return () => {
    const cur = _mcInstances.get(parentElement);
    if (!cur) return;
    input.removeEventListener("input", cur.onInput);
    input.removeEventListener("keydown", cur.onKeydown);
    input.removeEventListener("blur", cur.onBlur);
    clearBtn.removeEventListener("mousedown", cur.onClearMouseDown);
    clearBtn.removeEventListener("click", cur.onClear);
    _mcInstances.delete(parentElement);
  };
}
"""

try:
    _sidebar_search_component = st.components.v2.component(
        "medichat_sidebar_search",
        html=_SIDEBAR_SEARCH_HTML,
        css=_SIDEBAR_SEARCH_CSS,
        js=_SIDEBAR_SEARCH_JS,
    )
except Exception:
    _sidebar_search_component = None


def sidebar_search_box(*, key: str = "sidebar_search_box", placeholder: str = "Search chats") -> str:
    if _sidebar_search_component is None:
        return st.text_input(
            "Search chats",
            placeholder=placeholder,
            label_visibility="collapsed",
            key=key,
        )

    state = st.session_state.get(key, {})
    value = state.get("value", "") if isinstance(state, dict) else state
    value = str(value or "")

    result = _sidebar_search_component(
        key=key,
        data={"value": value, "placeholder": placeholder},
        default={"value": value},
        on_value_change=lambda: None,
        width="stretch",
        height="content",
    )
    return str(getattr(result, "value", value) or "")
# ── The Design System ──
# We're injecting custom CSS here to override Streamlit's defaults and achieve
# that dark, minimalist "SaaS" aesthetic.
st.markdown("""
<style>
/* GPT-like visual direction */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-main: #1a1b1e;
    --bg-soft: #202226;
    --bg-input: #2a2d33;
    --bg-input-hover: #31353c;
    --line-soft: rgba(255, 255, 255, 0.09);
    --text-main: #ececec;
    --text-muted: #a8adb7;
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

html, body, .stApp {
    font-family: var(--font-family) !important;
    background: radial-gradient(circle at 20% -10%, #2a2c31 0%, #1a1b1e 45%, #17181a 100%) !important;
    color: var(--text-main) !important;
    height: 100vh !important;
    min-height: 100vh !important;
}

[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    height: 100vh !important;
    min-height: 100vh !important;
}

footer, #MainMenu {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* Main conversation column */
.block-container {
    max-width: 768px !important;
    margin: 0 auto !important;
    padding-top: 2.2rem !important;
    padding-bottom: 8.5rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    position: relative !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    height: 100vh !important;
    min-height: 100vh !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.75rem !important;
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    height: 100vh !important;
    min-height: 100vh !important;
}

/* Target ALL buttons in sidebar for transparency and alignment */
section[data-testid="stSidebar"] .stButton button {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important;
    text-align: left !important;
    padding: 4px 8px !important;
    margin: 0 !important;
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    transition: background 0.12s ease !important;
    min-height: 28px !important;
    height: auto !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] .stButton button div {
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
    width: 100% !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] .stButton button p {
    font-size: 16px !important;
    line-height: 1.4 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background-color: rgba(255,255,255,0.07) !important;
}

/* Search input styling - fully bulletproof */
section[data-testid="stSidebar"] [data-testid="stTextInput"] > div > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #2f2f36 !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    color: #ececec !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8adb7' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: 12px center !important;
    background-size: 18px 18px !important;
    padding: 10px 36px 10px 40px !important;
    margin: 0 !important;
    width: calc(100% - 0px) !important;
    box-sizing: border-box !important;
    font-size: 16px !important;
    outline: none !important;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(255,255,255,0.2) !important;
}

/* Force hide the Press Enter to apply text */
section[data-testid="stSidebar"] [data-testid="stTextInput"] small {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] div.st-emotion-cache-1r6slb0,
section[data-testid="stSidebar"] [data-testid="stTextInput"] div[style*="font-size: 12px"] {
    display: none !important;
}

section[data-testid="stSidebar"] .stTextInput input {
    background-color: transparent !important;
    border: none !important;
    color: #ececec !important;
    /* Injecting a beautiful clean SVG icon */
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8adb7' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: 12px center !important;
    background-size: 18px 18px !important;
    padding: 10px 12px 10px 40px !important;
    margin: 0 !important;
    width: 100% !important;
    font-size: 16px !important;
}

section[data-testid="stSidebar"] .stTextInput input::placeholder {
    font-size: 16px !important;
    color: #a8adb7 !important;
}

.sidebar-label {
    color: #8e8ea0;
    font-size: 13.5px;
    font-weight: 600;
    display: block !important;
    padding: 4px 8px !important;
    margin: 1.0rem 0 0 0 !important;
    text-transform: none;
    letter-spacing: 0.01em;
}

/* Force Streamlit's wrapper to respect the bottom spacing */
section[data-testid="stSidebar"] .element-container:has(.sidebar-label) {
    margin-bottom: 15px !important;
}

/* Sidebar structural tweaks: we're compressing things a bit to keep the UI tight. */

/* Each columns ROW: zero internal gap between title col and dots col */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: center !important;
}

/* Each column cell inside a row: no extra pad */
section[data-testid="stSidebar"] [data-testid="stColumn"] {
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
    min-height: unset !important;
}

/* Top-level vertical block: zero gap — section spacing handled by element margins below */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* Give search input its own top margin so it breathes from New Chat */
section[data-testid="stSidebar"] .stTextInput {
    margin-top: 8px !important;
}

/* Nested vertical block (inside columns): zero gap */
section[data-testid="stSidebar"] [data-testid="stColumn"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* element-container inside column cells only */
section[data-testid="stSidebar"] [data-testid="stColumn"] .element-container {
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove popover trigger extra space */
section[data-testid="stSidebar"] div[data-testid="stPopover"] {
    margin: 0 !important;
    padding: 0 !important;
}

/* This strip is a nice touch—it shows up when the sidebar is collapsed to keep branding visible. */
#collapsed-sidebar-strip {
    display: none; /* hidden by default, shown via JS when sidebar collapses */
    position: fixed;
    top: 0;
    left: 0;
    width: 56px;
    height: 100vh;
    background-color: #000000;
    border-right: 1px solid rgba(255,255,255,0.08);
    z-index: 9999;
    flex-direction: column;
    align-items: center;
    padding-top: 14px;
    gap: 4px;
}

#collapsed-sidebar-strip .cs-logo {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}

#collapsed-sidebar-strip .cs-btn {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    background: transparent;
    border: none;
    color: #a8adb7;
    transition: background 0.13s, color 0.13s;
}

#collapsed-sidebar-strip .cs-btn:hover {
    background: rgba(255,255,255,0.09);
    color: #ffffff;
}

#collapsed-sidebar-strip .cs-btn svg {
    width: 20px;
    height: 20px;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.sidebar-branding {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 2.1rem 0;
    margin-top: -0.80rem;
}

.branding-logo {
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    border-radius: 9px;
}

.branding-logo svg {
    width: 20px;
    height: 20px;
}

.branding-text {
    font-family: "SF Pro Display", sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #f5f7fa;
    letter-spacing: -0.6px;
}

/* Make sidebar content a vertical column so bottom slot can anchor naturally */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    min-height: calc(100vh - 1.2rem) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > [data-testid="stVerticalBlock"] {
    min-height: calc(100vh - 1.2rem) !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Sidebar scroll scope: only Recents rows scroll */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .recent-chats-scroll-hook) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: calc(100vh - 315px) !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    margin-top: 10px !important;
    padding-bottom: 36px !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(255,255,255,0.28) transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .recent-chats-scroll-hook)::-webkit-scrollbar {
    width: 4px !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .recent-chats-scroll-hook)::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.28) !important;
    border-radius: 10px !important;
}

/* Anchor ONLY the footer wrapper (the block that directly contains sidebar-bottom-slot) */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .sidebar-bottom-slot) {
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    background: #000000 !important;
    border-top: 1px solid rgba(255,255,255,0.10) !important;
    padding: 0.7rem 0.5rem 1rem 0.5rem !important;
    z-index: 1000 !important;
    margin: 0 !important;
}

/* Hard pin bottom actions using stable Streamlit key classes */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-bottom: 120px !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .sidebar-bottom-slot) .stButton button {
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    color: #f5f7fa !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 14px !important;
    padding: 0.45rem 0.5rem !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .sidebar-bottom-slot) .stButton button:hover {
    background: rgba(255,255,255,0.08) !important;
}

/* FORCEFULLY REMOVE dropdown chevrons from sidebar popovers (3-dot, Settings, Profile) */
section[data-testid="stSidebar"] [data-testid="stPopover"] button div:first-child > *:last-child:not(:first-child) {
    display: none !important;
}


/* Chat rows */
/* ── 3-dot trigger button ── */
div[data-testid="stPopover"] button svg[data-testid="stIcon"] {
    display: none !important;
}

div[data-testid="stPopover"] button {
    border: none !important;
    background: transparent !important;
    padding: 2px 4px !important;
    margin: 0 !important;
    min-height: unset !important;
    min-width: unset !important;
    width: auto !important;
    box-shadow: none !important;
    border-radius: 6px !important;
    transition: background 0.15s ease !important;
}

div[data-testid="stPopover"] button p {
    color: #a8adb7 !important;
    font-size: 18px !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    letter-spacing: 1px;
}

div[data-testid="stPopover"] button:hover {
    background-color: rgba(255,255,255,0.08) !important;
}

div[data-testid="stPopover"] button:hover p {
    color: #ffffff !important;
}

/* ── ChatGPT-style popover card ── */
/* Target the floating popover panel */
[data-testid="stPopoverBody"] {
    background-color: #2f2f2f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    min-width: 140px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.3) !important;
    overflow: hidden !important;
}

/* Remove vertical gaps inside popovers to reduce height */
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* Also target alternate popover container selectors for robustness */
div[data-testid="stPopoverContainer"] > div:nth-child(2),
div[data-testid="stPopoverContainer"] > div:last-child {
    background-color: #2f2f2f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    min-width: 160px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.3) !important;
    overflow: hidden !important;
}

/* ── Delete button inside popover — ChatGPT red style ── */
section[data-testid="stSidebar"] div[data-testid="stPopover"] .stButton button,
[data-testid="stPopoverBody"] .stButton button {
    border: none !important;
    background-color: transparent !important;
    padding: 8px 12px !important;
    width: 100% !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important; /* Centrally aligned */
    gap: 10px !important;
    transition: background-color 0.12s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stPopover"] .stButton button:hover,
[data-testid="stPopoverBody"]:has(.is-delete-popover) .stButton button:hover {
    background-color: rgba(239, 68, 68, 0.12) !important;
}

section[data-testid="stSidebar"] div[data-testid="stPopover"] .stButton button p,
[data-testid="stPopoverBody"]:has(.is-delete-popover) .stButton button p {
    color: #ef4444 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important; /* Center text within p */
    gap: 10px !important;
    margin: 0 !important;
    padding: 0 !important;
    width: auto !important; /* Allow centering of the group */
    white-space: nowrap !important;
}

/* Red trash icon before the Delete label — matches ChatGPT's bin icon */
section[data-testid="stSidebar"] div[data-testid="stPopover"] .stButton button p::before,
[data-testid="stPopoverBody"]:has(.is-delete-popover) .stButton button p::before {
    content: '';
    display: inline-block;
    flex-shrink: 0;
    width: 16px;
    height: 16px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'%3E%3C/polyline%3E%3Cpath d='M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6'%3E%3C/path%3E%3Cpath d='M10 11v6'%3E%3C/path%3E%3Cpath d='M14 11v6'%3E%3C/path%3E%3Cpath d='M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2'%3E%3C/path%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    vertical-align: middle;
}

/* Base style for OTHER popovers (Settings & Profile) */
[data-testid="stPopoverBody"]:not(:has(.is-delete-popover)) .stButton button p {
    color: #ececec !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: auto !important; /* Allow centering of the group */
    white-space: nowrap !important;
}

/* Fix icon and text alignment inside Streamlit buttons with icons */
[data-testid="stPopoverBody"]:not(:has(.is-delete-popover)) .stButton button > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important; /* Centrally aligned */
    gap: 12px !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stPopoverBody"]:not(:has(.is-delete-popover)) .stButton button span.st-icon {
    font-size: 18px !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="stPopoverBody"]:not(:has(.is-delete-popover)) .stButton button:hover {
    background-color: rgba(255,255,255,0.08) !important;
}

/* Red styling for Logout in Profile popover */
[data-testid="stPopoverBody"]:has(.is-profile-popover) .stButton button p,
[data-testid="stPopoverBody"]:has(.is-profile-popover) .stButton button span.st-icon {
    color: #ef4444 !important;
}
[data-testid="stPopoverBody"]:has(.is-profile-popover) .stButton button:hover {
    background-color: rgba(239, 68, 68, 0.12) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    background-color: #2b2f36 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 18px !important;
    padding: 0.2rem 0.8rem !important;
    width: fit-content !important;
    margin-left: auto !important;
    color: var(--text-main) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 42px !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: var(--text-main) !important;
    max-width: 100% !important;
}

/* Avatar look */
[data-testid="stChatMessageAvatarUser"] {
    background-color: #4c6fff !important;
    margin-left: 10px !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #10a37f 0%, #0f8a6c 100%) !important;
    margin-right: 10px !important;
    border-radius: 10px !important;
}

/* The prompt bar is the heart of the UI. We've rounded it out for that modern look. */
[data-testid="stChatInput"] {
    --chat-input-h: 46px;
    --chat-font-size: 19px;
    width: 100%;
    max-width: min(768px, calc(100vw - 2rem)) !important;
    margin: 0 auto !important;
    padding: 0 0.25rem 0.75rem 0.25rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    transition: bottom 0.22s ease !important;
}

.chat-layout-empty,
.chat-layout-active {
    display: none;
}

/* Default (new chat): keep prompt near center like GPT home screen */
[data-testid="stChatInput"] {
    bottom: 33vh !important;
}

/* As soon as first message exists, dock prompt at bottom */
.stApp:has([data-testid="stChatMessage"]) [data-testid="stChatInput"] {
    bottom: 0.75rem !important;
}

[data-testid="stChatInput"] {
    width: 100% !important;
    min-width: 0 !important;
}

[data-testid="stChatInput"] > div {
    position: relative !important;
    display: flex;
    align-items: flex-end !important;
    gap: 0.35rem;
    min-width: 0 !important;
    width: 100% !important;
    min-height: unset !important;
    height: auto !important;
    max-height: 62px !important;
    overflow: hidden !important;
    padding: 14px 0.3rem 14px 1.0rem !important;
    background: rgba(48, 51, 58, 0.78) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 26px !important;
    box-shadow: none !important;
    margin: 0 !important;
}

[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div > div {
    display: flex !important;
    align-items: stretch !important;
    min-height: unset !important;
    max-height: 34px !important;
    height: auto !important;
    overflow: hidden !important;
    min-width: 0 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    border: none !important;
    border-radius: 0px !important;
    margin-left: 0 !important;
    padding-left: 0.5rem !important;
    padding-right: 2.7rem !important;
    color: #ececec !important;
    font-size: 19px !important;
    line-height: 1.3 !important;
    min-height: 28px !important;
    height: 28px !important;
    max-height: 28px !important;
    padding-top: 3px !important;
    padding-bottom: 1px !important;
    caret-color: #ececec !important;
    outline: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    resize: none !important;
    display: block !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-y: hidden !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
    -ms-overflow-style: none !important;  /* IE and Edge */
    scrollbar-width: none !important;  /* Firefox */
}

[data-testid="stChatInput"] textarea::-webkit-scrollbar {
    display: none !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"] {
    display: block !important;
    height: auto !important;
    font-size: var(--chat-font-size) !important;
    min-width: 0 !important;
    width: 100% !important;
}

[data-testid="stChatInput"] textarea[data-testid="stChatInputTextArea"] {
    font-size: 19px !important;
    caret-color: #ececec !important;
}

[data-testid="stChatInput"] [data-baseweb="textarea"] {
    flex: 1 1 auto;
    min-height: unset !important;
    display: block !important;
    height: auto !important;
    font-size: var(--chat-font-size) !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    min-width: 0 !important;
    max-width: 100% !important;
}

/* BaseWeb injects nested wrappers with default backgrounds; clear all layers */
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"] *,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="base-input"] * {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea:focus {
    background-color: transparent !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(255, 255, 255, 0.26) !important;
    background: rgba(54, 57, 64, 0.88) !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(236, 236, 236, 0.74) !important;
    line-height: 1.3 !important;
    font-size: 19px !important;
}

[data-testid="stChatInputSubmitButton"] {
    border-radius: 999px !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    margin-right: 2px !important;
    margin-bottom: 0 !important;
    background: #ececec !important;
    color: #121316 !important;
    border: none !important;
    position: absolute !important;
    right: 6px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    display: inline-flex !important;
    align-self: center !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 10 !important;
    pointer-events: auto !important;
}

[data-testid="stChatInputSubmitButton"]:hover {
    background: #ffffff !important;
}

[data-testid="stChatInputSubmitButton"]:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.22) !important;
}

[data-testid="stChatInputSubmitButton"] svg {
    width: 16px !important;
    height: 16px !important;
}

/* Change send arrow to a 'stop' square when disabled/thinking */
[data-testid="stChatInputSubmitButton"]:disabled {
    opacity: 1 !important;
    cursor: not-allowed !important;
}

[data-testid="stChatInputSubmitButton"]:disabled svg {
    display: none !important;
}

[data-testid="stChatInputSubmitButton"]:disabled::after {
    content: "";
    display: block !important;
    width: 12px !important;
    height: 12px !important;
    background-color: #121316 !important;
    border-radius: 2px !important;
}

/* Empty state headline */
.hero-title {
    position: absolute;
    top: calc(50vh - 80px);
    left: 0;
    right: 0;
    transform: translateY(-100%);
    width: 100%;
    max-width: min(768px, calc(100vw - 2rem)) !important;
    margin: 0 auto !important;
    text-align: left;
    padding-left: 0.5rem !important;
    font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: clamp(2.0rem, 4vw, 3.0rem) !important;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: #f5f7fa;
    pointer-events: none;
    z-index: 100;
}

/* Hide hero title when chat messages are present */
.stApp:has([data-testid="stChatMessage"]) .hero-title {
    display: none !important;
}


@media (max-width: 768px) {
    .hero-title {
        top: calc(50% - 50px);
    }
}

/* Disclaimer */
.sticky-disclaimer {
    position: fixed;
    bottom: 1.2rem;
    left: 50%;
    transform: translateX(-50%);
    width: auto;
    white-space: nowrap;
    text-align: center;
    font-size: 13px;
    color: #e6c384;
    z-index: 100;
    pointer-events: none;
    transition: left 0.3s cubic-bezier(0.2, 0, 0, 1);
}

/* Shift disclaimer right when sidebar is open to stay centered with prompt bar */
.stApp:has([data-testid="stSidebar"][aria-expanded="true"]) .sticky-disclaimer,
.stApp:has([data-testid="stSidebar"][data-expanded="true"]) .sticky-disclaimer {
    left: calc(50% + 130px);
}


@media (max-width: 980px) {
    .block-container {
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }

    [data-testid="stChatInput"] {
        --chat-input-h: 44px;
        --chat-font-size: 18px;
        max-width: 100% !important;
        padding: 0 0.15rem 0.6rem 0.15rem !important;
    }

    [data-testid="stChatInput"] > div {
        min-height: unset !important;
        border-radius: 24px !important;
        padding-left: 0.75rem;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea::placeholder {
        font-size: 18px !important;
    }

    [data-testid="stChatInput"] {
        bottom: 25vh !important;
    }

    .stApp:has([data-testid="stChatMessage"]) [data-testid="stChatInput"] {
        bottom: 0.55rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ── State Management ──────────────────────────────────────────────────────────
def init_session():
    """Initialize all session state keys used across the application."""

    # ── Authentication ────────────────────────────────────────────────────────
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = {}   # full user record from auth.py
    if "login_error" not in st.session_state:
        st.session_state.login_error = ""
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"
    if not st.session_state.get("authenticated", False):
        _restore_auth_from_url()

    # ── Consent ───────────────────────────────────────────────────────────────
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False

    # ── Chat history ──────────────────────────────────────────────────────────
    if "chats" not in st.session_state:
        default_id = str(uuid.uuid4())
        st.session_state.chats = {
            default_id: {
                "title": "New Chat",
                "messages": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        }
        st.session_state.current_chat = default_id
    if "chats_loaded_for" not in st.session_state:
        st.session_state.chats_loaded_for = ""
    if st.session_state.get("authenticated", False):
        username = st.session_state.get("current_user", {}).get("username", "")
        if username and st.session_state.chats_loaded_for != username:
            disk_chats, disk_current = _load_chats_from_disk(username)
            if disk_chats:
                st.session_state.chats = disk_chats
                st.session_state.current_chat = disk_current or next(iter(disk_chats.keys()))
            st.session_state.chats_loaded_for = username

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "pending_chat_id" not in st.session_state:
        st.session_state.pending_chat_id = None

    # ── Sidebar panel toggles ─────────────────────────────────────────────────
    if "show_profile" not in st.session_state:
        st.session_state.show_profile = False
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    # ── Sidebar panel toggles ─────────────────────────────────────────────────
    if "show_profile" not in st.session_state:
        st.session_state.show_profile = False
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

# Helper to get active chat
def get_current_chat():
    chat_id = st.session_state.current_chat
    if chat_id not in st.session_state.chats:
        chat_id = list(st.session_state.chats.keys())[0]
        st.session_state.current_chat = chat_id
    return st.session_state.chats[chat_id]


# ── API Integration ───────────────────────────────────────────────────────────
def call_rag_pipeline(question: str, history: list):
    """Call backend API and pass full history for context awareness."""
    try:
        # Format history to match expected backend schema
        api_history = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        
        payload = {
            "question": question,
            "history": api_history
        }
        res = requests.post(API_URL, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": f"System Error: {str(e)}"}


# ── UI Modules ────────────────────────────────────────────────────────────────

# ── Authentication & Onboarding Pages ────────────────────────────────────────

def render_login_page() -> None:
    """Render a clean, Apple-style centered login page."""
    st.markdown("""
    <style>
    /* Hide sidebar completely on login/consent pages */
    [data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stSidebarContent"] { display: none !important; }
    [data-testid="stChatInput"] { display: none !important; }
    [data-testid="stChatInputContainer"] { display: none !important; }
    [data-testid="stBottomBlockContainer"] { display: none !important; }
    .stChatFloatingInputContainer { display: none !important; }
    [data-testid="stChatInput"] * { display: none !important; }
    .sticky-disclaimer { display: none !important; }
    /* Center login content without HTML wrappers that Streamlit breaks */
    .login-shell {
        max-width: 920px;
        margin: 9vh auto 0 auto;
        text-align: center;
    }
    .login-logo {
        width: 52px; height: 52px;
        background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.1rem auto;
    }
    .login-title,
    .stApp .login-shell .login-title,
    .stApp .login-shell p.login-title {
        text-align: center !important;
        font-size: clamp(46px, 7.5vw, 92px) !important;
        font-weight: 900 !important;
        color: #f5f7fa !important;
        margin: 0 0 0.25rem 0 !important;
        line-height: 1.0 !important;
        letter-spacing: -0.015em !important;
    }
    .login-sub {
        text-align: center;
        font-size: 14px; color: #8e8ea0;
        margin: 0 0 1.8rem 0;
    }
    /* Style Streamlit inputs on this page */
    .stApp:has(.login-title) [data-testid="stTextInput"],
    .stApp:has(.login-title) .stButton {
        max-width: 940px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stApp:has(.login-title) .stTextInput input {
        background: #2a2d33 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #ececec !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 15px !important;
    }
    .stApp:has(.login-title) .stTextInput input:focus {
        border-color: rgba(13,148,136,0.6) !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.15) !important;
    }
    .stApp:has(.login-title) .stButton button {
        width: 100% !important;
        background: linear-gradient(135deg, #0d9488, #0891b2) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #fff !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 0.65rem !important;
        margin-top: 0.4rem !important;
        transition: opacity 0.15s !important;
    }
    .stApp:has(.login-title) .stButton button:hover { opacity: 0.88 !important; }
    .login-error {
        background: rgba(220,38,38,0.12);
        border: 1px solid rgba(220,38,38,0.3);
        border-radius: 8px;
        color: #fca5a5;
        font-size: 13.5px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Logo
    st.markdown("""
    <div class='login-shell'>
        <div class='login-logo'>
            <svg viewBox='0 0 24 24' fill='none' width='28' height='28'>
                <path d='M12 4V20M4 12H20' stroke='white' stroke-width='3' stroke-linecap='round'/>
            </svg>
        </div>
        <p class='login-title'>MediChat</p>
        <p class='login-sub'>AI-powered clinical assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Error banner
    if st.session_state.login_error:
        st.markdown(
            f"<div class='login-error'>{st.session_state.login_error}</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.auth_mode == "login":
        username = st.text_input(
            "Username", placeholder="Enter your username",
            label_visibility="collapsed", key="login_username",
        )
        password = st.text_input(
            "Password", placeholder="Enter your password",
            type="password", label_visibility="collapsed", key="login_password",
        )

        if st.button("Sign In", use_container_width=True, key="login_btn"):
            ok, user_record = verify_credentials(username, password)
            if ok:
                normalized_username = username.strip().lower()
                session_user = {
                    "username": normalized_username,
                    "display_name": user_record.get("display_name", normalized_username),
                    "role": user_record.get("role", "user"),
                }
                st.session_state.authenticated = True
                st.session_state.current_user = session_user
                st.session_state.login_error = ""
                st.session_state.consent_given = False
                _set_auth_token_in_url(
                    create_session_token(normalized_username, consent_given=False)
                )
                st.rerun()
            else:
                st.session_state.login_error = "Incorrect username or password."
                st.rerun()
        
        st.markdown("<div style='text-align:center; margin-top:1rem;'>", unsafe_allow_html=True)
        if st.button("Don't have an account? Register", key="switch_to_reg"):
            st.session_state.auth_mode = "register"
            st.session_state.login_error = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        full_name = st.text_input(
            "Full Name", placeholder="Enter your full name",
            label_visibility="collapsed", key="reg_fullname",
        )
        username = st.text_input(
            "Username", placeholder="Choose a username",
            label_visibility="collapsed", key="reg_username",
        )
        password = st.text_input(
            "Password", placeholder="Create a password",
            type="password", label_visibility="collapsed", key="reg_password",
        )
        confirm_password = st.text_input(
            "Confirm Password", placeholder="Confirm your password",
            type="password", label_visibility="collapsed", key="reg_confirm_password",
        )

        if st.button("Create Account", use_container_width=True, key="reg_btn"):
            if not username or not password or not full_name or not confirm_password:
                st.session_state.login_error = "Please fill in all fields."
                st.rerun()
            elif password != confirm_password:
                st.session_state.login_error = "Passwords do not match."
                st.rerun()
            elif user_exists(username):
                st.session_state.login_error = "Username already exists."
                st.rerun()
            else:
                add_user(username, password, display_name=full_name)
                # Auto-login after registration
                ok, user_record = verify_credentials(username, password)
                if ok:
                    normalized_username = username.strip().lower()
                    session_user = {
                        "username": normalized_username,
                        "display_name": user_record.get("display_name", normalized_username),
                        "role": user_record.get("role", "user"),
                    }
                    st.session_state.authenticated = True
                    st.session_state.current_user = session_user
                    st.session_state.login_error = ""
                    st.session_state.auth_mode = "login"
                    st.session_state.consent_given = False
                    _set_auth_token_in_url(
                        create_session_token(normalized_username, consent_given=False)
                    )
                    st.rerun()
        
        st.markdown("<div style='text-align:center; margin-top:1rem;'>", unsafe_allow_html=True)
        if st.button("Already have an account? Sign In", key="switch_to_login"):
            st.session_state.auth_mode = "login"
            st.session_state.login_error = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def render_consent_page() -> None:
    """Full-screen consent page shown after login and before the chatbot."""
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stChatInput"] { display: none !important; }
    [data-testid="stChatInputContainer"] { display: none !important; }
    [data-testid="stBottomBlockContainer"] { display: none !important; }
    .stChatFloatingInputContainer { display: none !important; }
    [data-testid="stChatInput"] * { display: none !important; }
    .sticky-disclaimer { display: none !important; }
    .consent-shell {
        max-width: 940px;
        margin: 9vh auto 0 auto;
        text-align: center;
    }
    .consent-logo {
        width: 52px; height: 52px;
        background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.1rem auto;
    }
    .consent-title,
    .stApp .consent-shell .consent-title,
    .stApp .consent-shell p.consent-title {
        text-align: center !important;
        font-size: clamp(46px, 7.5vw, 92px) !important;
        font-weight: 900 !important;
        color: #f5f7fa !important;
        margin: 0 0 0.2rem 0 !important;
        line-height: 1.0 !important;
        letter-spacing: -0.015em !important;
    }
    .consent-sub {
        text-align: center; font-size: 14px;
        color: #8e8ea0; margin: 0 0 1.6rem 0;
    }
    .consent-body {
        max-width: 940px;
        margin: 0 auto 1.4rem auto;
        text-align: left;
        font-size: 14.5px; 
        color: #f1e5ac; /* Light golden/beige text */
        line-height: 1.7;
        background: rgba(255, 215, 0, 0.05); /* Subtle gold background tint */
        border-radius: 10px; 
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255, 215, 0, 0.25); /* Elegant gold border */
    }
    .stApp:has(.consent-title) [data-testid="stCheckbox"] {
        color: #f1e5ac !important;
        font-weight: 500 !important;
    }
    .stApp:has(.consent-title) [data-testid="stCheckbox"],
    .stApp:has(.consent-title) .stButton {
        max-width: 940px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stApp:has(.consent-title) .stButton button {
        width: 100% !important;
        background: linear-gradient(135deg, #0d9488, #0891b2) !important;
        border: none !important; border-radius: 10px !important;
        color: #fff !important; font-size: 15px !important;
        font-weight: 600 !important; padding: 0.65rem !important;
        margin-top: 0.5rem !important;
        transition: opacity 0.15s !important;
    }
    .stApp:has(.consent-title) .stButton button:hover { opacity: 0.88 !important; }
    .stApp:has(.consent-title) .stButton button:disabled {
        background: rgba(255,255,255,0.08) !important;
        color: #555 !important; cursor: not-allowed !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='consent-shell'>
        <div class='consent-logo'>
            <svg viewBox='0 0 24 24' fill='none' width='28' height='28'>
                <path d='M12 4V20M4 12H20' stroke='white' stroke-width='3' stroke-linecap='round'/>
            </svg>
        </div>
        <p class='consent-title'>MediChat</p>
        <p class='consent-sub'>AI-powered medical assistant</p>
        <div class='consent-body'>
            This chatbot draws answers from curated medical literature and is intended
            for <strong>informational and educational purposes only</strong>.
            It is <strong>not</strong> a substitute for professional medical advice,
            diagnosis, or treatment. Always consult a qualified healthcare professional
            before making any health-related decision.
            <br><br>
            By continuing, you agree to use this tool responsibly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    consent_controls = st.empty()
    with consent_controls.container():
        agreed = st.checkbox(
            "I understand and agree",
            key="consent_checkbox_full",
        )

        continue_btn = st.button(
            "Continue to MediChat",
            disabled=not agreed,
            use_container_width=True,
            key="consent_continue_btn",
        )

    if continue_btn and agreed:
        st.session_state.consent_given = True
        st.session_state.consent_transitioning = True
        _refresh_auth_token(consent_given=True)
        consent_controls.empty()
        st.empty()
        time.sleep(0.08)
        st.rerun()

# ── Sidebar Profile & Settings Helpers ────────────────────────────────────────






def render_sidebar():
    """Renders the left sidebar containing ChatGPT-like controls."""
    with st.sidebar:
        # Branding
        st.markdown("""
            <div class='sidebar-branding'>
                <div class='branding-logo'>
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 4V20M4 12H20" stroke="white" stroke-width="3" stroke-linecap="round"/>
                    </svg>
                </div>
                <span class='branding-text'>MediChat</span>
            </div>
        """, unsafe_allow_html=True)

        # New Chat Button
        if st.button("New chat", icon=":material/edit_square:", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {
                "title": "New Chat",
                "messages": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.current_chat = new_id
            _persist_chats_if_authenticated()
            st.rerun()

        # Search Box (real-time; no Enter required)
        search_val = sidebar_search_box(key="sidebar_search_box", placeholder="Search chats")

        # Filter and Sort chats
        sorted_chats = sorted(
            [(cid, cdata) for cid, cdata in st.session_state.chats.items()
             if cdata.get("messages") and len(cdata["messages"]) > 0
             and (not search_val or search_val.strip().lower() in cdata["title"].lower())],
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )

        # Chat History List
        st.markdown("<div class='sidebar-label'>Recents</div>", unsafe_allow_html=True)
        recents_scroll_container = st.container()
        with recents_scroll_container:
            st.markdown("<div class='recent-chats-scroll-hook'></div>", unsafe_allow_html=True)
            for chat_id, chat_data in sorted_chats:

                col1, col2 = st.columns([0.88, 0.12])
                with col1:
                    is_active = st.session_state.current_chat == chat_id
                    btn_label = f"{chat_data['title']}"
                    if st.button(btn_label, key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat = chat_id
                        st.rerun()
                with col2:
                    with st.popover("⋯"):
                        st.markdown("<div class='is-delete-popover'></div>", unsafe_allow_html=True)
                        if st.button("Delete", key=f"del_{chat_id}", use_container_width=True):
                            del st.session_state.chats[chat_id]
                            if st.session_state.current_chat == chat_id:
                                if st.session_state.chats:
                                    remaining = sorted(st.session_state.chats.items(), key=lambda x: x[1]['timestamp'], reverse=True)
                                    st.session_state.current_chat = remaining[0][0]
                                else:
                                    new_id = str(uuid.uuid4())
                                    st.session_state.chats[new_id] = {"title": "New Chat", "messages": [], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                                    st.session_state.current_chat = new_id
                            _persist_chats_if_authenticated()
                            st.rerun()

        # ── Sidebar bottom: Profile & Settings ────────────────────────────────
        footer_container = st.container()
        with footer_container:
            st.markdown("<div class='sidebar-bottom-slot'></div>", unsafe_allow_html=True)
            
            display_name = st.session_state.current_user.get("display_name", "User")
            initials = ''.join([part[0] for part in display_name.split()][:2]).upper() if display_name else "U"

            # Settings Popover
            with st.popover("Settings", icon=":material/settings:", key="sidebar_settings_popover", use_container_width=True):
                st.markdown("<div class='is-settings-popover'></div>", unsafe_allow_html=True)
                if st.button("Reset Chat", icon=":material/refresh:", key="reset_chat_pop_btn", use_container_width=True):
                    new_id = str(uuid.uuid4())
                    st.session_state.chats = {
                        new_id: {
                            "title": "New Chat",
                            "messages": [],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                    }
                    st.session_state.current_chat = new_id
                    _persist_chats_if_authenticated()
                    st.rerun()

            # Profile Popover
            with st.popover(f"{display_name[:30]}", icon=":material/account_circle:", key="sidebar_profile_popover", use_container_width=True):
                st.markdown("<div class='is-profile-popover'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
                    <div style="width:34px;height:34px;border-radius:50%;background:#e57339;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;font-size:14px;flex-shrink:0;">
                        {initials}
                    </div>
                    <div style="line-height:1.2;">
                        <p style="font-size:14px;font-weight:600;color:#f5f7fa;margin:0;">{display_name}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Log out", icon=":material/logout:", key="logout_pop_btn", use_container_width=True):
                    _set_auth_token_in_url("")
                    st.session_state.clear()
                    st.rerun()


def render_chat():
    """Renders the main chat history view using native st.chat_message."""
    current_chat = get_current_chat()
    
    if len(current_chat["messages"]) == 0:
        import random
        titles = ["Care starts here.", "Support, when it matters!", "With you, Always"]
        hero_title = random.choice(titles)
        st.markdown(f"<h2 class='hero-title'>{hero_title}</h2>", unsafe_allow_html=True)
        
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)
            if msg.get("role") == "assistant" and msg.get("confidence"):
                _render_confidence_badge(str(msg["confidence"]))


# Confidence badge colours mapped to pipeline confidence levels.
_CONF_STYLE = {
    "high":   ("#16a34a", "High confidence"),
    "medium": ("#d97706", "Medium confidence"),
    "low":    ("#dc2626", "Low confidence - verify with a professional"),
}


def _render_confidence_badge(confidence: str) -> None:
    """Render a small, unobtrusive confidence indicator below an answer."""
    colour, label = _CONF_STYLE.get(
        confidence.lower(),
        ("#6b7280", f"Confidence: {confidence}"),
    )
    st.markdown(
        f'<p style="font-size:12px;color:{colour};margin-top:4px;">{label}</p>',
        unsafe_allow_html=True,
    )


def _render_consent_gate() -> bool:
    """Show a one-time medical disclaimer consent checkbox.

    Returns True once the user has ticked the box, False otherwise.
    The gate is only shown once per browser session.
    """
    st.markdown(
        """
        <div style="
            background:#1e2025;border:1px solid rgba(255,255,255,0.1);
            border-radius:14px;padding:1.4rem 1.6rem;max-width:580px;
            margin:6vh auto 0 auto;">
          <p style="font-size:17px;font-weight:600;color:#f5f7fa;margin:0 0 0.6rem 0;">
            Before you continue
          </p>
          <p style="font-size:14px;color:#a8adb7;line-height:1.6;margin:0 0 1rem 0;">
            MediChat provides information from medical literature for
            <strong>educational purposes only</strong>. It is <strong>not</strong>
            a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare professional before making any
            health-related decisions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    agreed = st.checkbox(
        "I understand this tool is for informational purposes only and does not "
        "replace professional medical advice.",
        key="consent_checkbox",
    )
    if agreed:
        st.session_state.consent_given = True
        st.rerun()
    return False


def handle_input():
    """Handles the sticky chat input at the bottom and streaming response."""
    # Authentication and consent are enforced by main() before this is called.
    current_chat = get_current_chat()

    # st.chat_input natively sticks to the bottom
    user_query = st.chat_input("How can I help with your health today?")
    if user_query:
        # Auto-name chat based on first query
        if len(current_chat["messages"]) == 0:
            current_chat["title"] = user_query[:30]

        # Append User Message
        current_chat["messages"].append({"role": "user", "content": user_query})
        _persist_chats_if_authenticated()
        with st.chat_message("user"):
            st.markdown(user_query, unsafe_allow_html=False)

        # Fetch and Render Assistant Response
        with st.chat_message("assistant"):
            normalized_query = re.sub(r"\s+", " ", user_query.strip().lower())
            is_identity_query = normalized_query in {
                "who are you",
                "who are you?",
                "what are you",
                "what are you?",
            }

            if is_identity_query:
                full_response = (
                    "I am MediChat, an AI-powered medical assistant designed to provide "
                    "informational guidance from curated medical literature. I do not "
                    "replace professional medical advice, diagnosis, or treatment."
                )
                st.markdown(full_response, unsafe_allow_html=False)
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "confidence": "high",
                })
                _persist_chats_if_authenticated()
                st.rerun()
                return

            # Shortcut for simple greetings to avoid backend calls
            is_greeting_query = re.sub(r"[^a-z\s]", "", normalized_query).strip() in {
                "hi", "hello", "hey", "hii", "heyy", 
                "good morning", "good afternoon", "good evening"
            }
            if is_greeting_query:
                full_response = (
                    "Hello, I am MediChat, your medical assistant. I'm here to provide "
                    "clear and professional guidance on various health topics. "
                    "What can I help you with today?"
                )
                st.markdown(full_response, unsafe_allow_html=False)
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "confidence": "high",
                })
                _persist_chats_if_authenticated()
                st.rerun()
                return

            with st.spinner("Thinking..."):
                # Pass full history up to (but excluding) the assistant's new turn
                result = call_rag_pipeline(user_query, current_chat["messages"][:-1])

            if "error" in result:
                err_msg = f"**Error:** {result['error']}"
                st.error(err_msg)
                current_chat["messages"].append({"role": "assistant", "content": err_msg})
                _persist_chats_if_authenticated()
            else:
                ans = result.get("answer", "No response.")
                conf = result.get("confidence", "low")

                # Streaming simulation — preserves markdown line breaks
                message_placeholder = st.empty()
                full_response = ""
                for chunk in re.split(r'(\s+)', ans):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=False)
                    time.sleep(0.005)
                message_placeholder.markdown(full_response, unsafe_allow_html=False)

                # Save to state
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "confidence": conf,
                })
                _persist_chats_if_authenticated()
                st.rerun()


# ── The Main Event ──
# This is where the app actually kicks off. We handle the routing through
# the login and consent gates before showing the chatbot.

def main():
    init_session()

    # ── Navigation Gate ───────────────────────────────────────────────────────
    # Flow: Login → Consent → Chatbot
    if not st.session_state.get("authenticated", False):
        render_login_page()
        return

    if not st.session_state.get("consent_given", False):
        render_consent_page()
        return

    if st.session_state.pop("consent_transitioning", False):
        st.markdown(
            "<style>"
            ".consent-shell,"
            ".st-key-consent_checkbox_full,"
            ".st-key-consent_continue_btn{display:none !important;}"
            "</style>",
            unsafe_allow_html=True,
        )
        st.empty()
        time.sleep(0.08)
        st.rerun()

    # ── Full chatbot (authenticated + consented) ──────────────────────────────
    st.markdown("""
    <script>
    (function lockChatInput() {
        var FIXED_PX = 24;
        var FIXED_H  = FIXED_PX + 'px';

        function applyLock(ta) {
            /* 1. Spoof scrollHeight so Streamlit's own auto-resize always reads 24 */
            try {
                Object.defineProperty(ta, 'scrollHeight', {
                    get: function() { return FIXED_PX; },
                    configurable: true
                });
            } catch(e) {}

            /* 2. Force inline styles (beats CSS and Streamlit's resize handler) */
            ta.style.setProperty('height',        FIXED_H,  'important');
            ta.style.setProperty('min-height',    FIXED_H,  'important');
            ta.style.setProperty('max-height',    FIXED_H,  'important');
            ta.style.setProperty('overflow-y',    'hidden', 'important');
            ta.style.setProperty('overflow-x',    'hidden', 'important');
            ta.style.setProperty('resize',        'none',   'important');
            ta.style.setProperty('white-space',   'nowrap', 'important');
            ta.style.setProperty('overflow-wrap', 'normal', 'important');
            ta.style.setProperty('word-break',    'normal', 'important');
            ta.style.setProperty('font-size',     '19px',   'important');
            ta.style.setProperty('line-height',   '1.3',    'important');
            ta.style.setProperty('padding-top',   '3px',    'important');
        }

        function init() {
            var ta = document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (!ta) { setTimeout(init, 80); return; }

            applyLock(ta);

            /* 3. Re-lock after every keystroke (input fires AFTER Streamlit's resize) */
            ta.addEventListener('input', function() { applyLock(ta); }, true);

            /* 4. MutationObserver with guard flag to prevent infinite loop */
            var busy = false;
            var obs = new MutationObserver(function() {
                if (busy) return;
                busy = true;
                applyLock(ta);
                busy = false;
            });
            obs.observe(ta, { attributes: true, attributeFilter: ['style'] });
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    })();
    </script>
    """, unsafe_allow_html=True)

    render_sidebar()
    render_chat()
    handle_input()

    # ── Collapsed sidebar icon strip (GPT style) ──────────────────────────────
    st.markdown("""
    <div id="collapsed-sidebar-strip">
        <!-- Logo -->
        <div class="cs-logo">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                <path d="M12 4V20M4 12H20" stroke="white" stroke-width="3" stroke-linecap="round"/>
            </svg>
        </div>
        <!-- New Chat -->
        <button class="cs-btn" title="New chat" onclick="openSidebar()">
            <svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
        </button>
        <!-- Search -->
        <button class="cs-btn" title="Search chats" onclick="openSidebar()">
            <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </button>
        <!-- Chats -->
        <button class="cs-btn" title="Chats" onclick="openSidebar()">
            <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        </button>
    </div>

    <script>
    (function collapsedStrip() {
        if (window.__medichatSidebarEnhancerLoaded) return;
        window.__medichatSidebarEnhancerLoaded = true;

        // Open sidebar: try every known Streamlit expand button selector
        function openSidebar() {
            var btn =
                document.querySelector('[data-testid="collapsedControl"]') ||
                document.querySelector('button[kind="header"]') ||
                document.querySelector('[aria-label="open sidebar"]') ||
                document.querySelector('[aria-label="Open sidebar"]');
            if (btn) { btn.click(); return; }
            // Fallback: find any button containing ">>" text near viewport edge
            document.querySelectorAll('button').forEach(function(b) {
                if (b.innerText && b.innerText.trim() === '>>') b.click();
            });
        }
        window.openSidebar = openSidebar;

        function isSidebarCollapsed() {
            // If Streamlit's native expand button is visible, sidebar is definitely collapsed
            var expandBtn = document.querySelector('[data-testid="collapsedControl"]') || document.querySelector('button[kind="header"]');
            if (expandBtn && window.getComputedStyle(expandBtn).display !== 'none') return true;

            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            // If the sidebar element is completely gone from the DOM, it means it's closed!
            if (!sidebar) return true;

            // Check if sidebar is squished OR pushed off the left side of the screen
            var rect = sidebar.getBoundingClientRect();
            if (rect.width <= 60 || rect.right <= 60) return true;

            // Fallback: aria-expanded attribute
            var expanded = sidebar.getAttribute('aria-expanded');
            if (expanded === 'false') return true;
            
            return false;
        }

        var strip = null;
        function syncStrip() {
            if (!strip) strip = document.getElementById('collapsed-sidebar-strip');
            if (!strip) return;
            strip.style.display = isSidebarCollapsed() ? 'flex' : 'none';
        }

        // Cleanup any legacy inline styles from previous spacing hacks that might still be running
        function cleanupLegacyStyles() {
            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            var rows = sidebar.querySelectorAll('[data-testid="stHorizontalBlock"]');
            for (var i = 0; i < rows.length; i++) {
                var el = rows[i];
                var vblock = el.closest('[data-testid="stVerticalBlock"]');
                if (vblock && el.parentElement !== vblock) {
                    el = el.parentElement;
                }
                el.style.removeProperty('margin-top');
            }
        }

        function injectSearchClear() {
            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;

            // If custom component exists, it already owns clear behavior.
            var customSearch = sidebar.querySelector('#mc-input');
            if (customSearch) return;

            var searchInput = sidebar.querySelector('.stTextInput input');
            if (!searchInput) return;

            var container = searchInput.closest('[data-testid="stTextInput"]') || searchInput.parentElement;
            if (!container) return;

            var clearBtn = container.querySelector('.search-clear');
            if (!clearBtn) {
                clearBtn = document.createElement('button');
                clearBtn.type = 'button';
                clearBtn.className = 'search-clear';
                clearBtn.setAttribute('aria-label', 'Clear search');
                clearBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6L6 18M6 6l12 12"></path></svg>';
                container.appendChild(clearBtn);

                Object.assign(clearBtn.style, {
                    position: 'absolute',
                    right: '10px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: '22px',
                    height: '22px',
                    border: 'none',
                    borderRadius: '999px',
                    background: 'rgba(255,255,255,0.12)',
                    color: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    opacity: '0',
                    pointerEvents: 'none',
                    zIndex: '20',
                    transition: 'opacity 0.12s ease, background 0.12s ease'
                });

                var icon = clearBtn.querySelector('svg');
                if (icon) {
                    icon.style.width = '12px';
                    icon.style.height = '12px';
                    icon.style.stroke = 'currentColor';
                    icon.style.strokeWidth = '2.2';
                    icon.style.fill = 'none';
                    icon.style.strokeLinecap = 'round';
                }

                clearBtn.addEventListener('mouseenter', function() {
                    clearBtn.style.background = 'rgba(255,255,255,0.22)';
                    clearBtn.style.color = '#ffffff';
                });
                clearBtn.addEventListener('mouseleave', function() {
                    clearBtn.style.background = 'rgba(255,255,255,0.12)';
                    clearBtn.style.color = '#ffffff';
                });

                clearBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    setter.call(searchInput, '');
                    searchInput.__mcLastSent = '';
                    searchInput.dispatchEvent(new Event('input', { bubbles: true }));
                    searchInput.dispatchEvent(new Event('change', { bubbles: true }));
                    searchInput.focus();
                });
            }

            container.style.position = 'relative';
            searchInput.style.paddingRight = '38px';
            searchInput.style.position = 'relative';
            searchInput.style.zIndex = '1';

            var syncClearVisibility = function() {
                var hasValue = !!(searchInput.value && searchInput.value.length);
                clearBtn.style.opacity = hasValue ? '1' : '0';
                clearBtn.style.pointerEvents = hasValue ? 'auto' : 'none';
            };

            if (!searchInput.dataset.mcRealtimeBound) {
                searchInput.__mcLastSent = searchInput.value || '';
                searchInput.__mcRealtimeTimer = null;

                searchInput.addEventListener('input', function() {
                    syncClearVisibility();
                    if (searchInput.__mcRealtimeTimer) clearTimeout(searchInput.__mcRealtimeTimer);
                    searchInput.__mcRealtimeTimer = setTimeout(function() {
                        var nextValue = searchInput.value || '';
                        if (nextValue !== searchInput.__mcLastSent) {
                            searchInput.__mcLastSent = nextValue;
                            searchInput.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }, 120);
                });
                searchInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') e.preventDefault();
                });
                searchInput.addEventListener('blur', function() {
                    if (searchInput.__mcRealtimeTimer) {
                        clearTimeout(searchInput.__mcRealtimeTimer);
                        searchInput.__mcRealtimeTimer = null;
                    }
                    var nextValue = searchInput.value || '';
                    if (nextValue !== searchInput.__mcLastSent) {
                        searchInput.__mcLastSent = nextValue;
                        searchInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                });
                searchInput.dataset.mcRealtimeBound = '1';
            }

            syncClearVisibility();
        }

        function fixSidebarFooter() {
            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;

            var marker = sidebar.querySelector('.sidebar-bottom-slot');
            if (!marker) return;

            // Find the element container that holds our footer
            var footerElement = marker.closest('[data-testid="stElementContainer"]');
            if (!footerElement) return;

            // Find the main vertical block that contains everything in the sidebar
            var parentBlock = footerElement.parentElement;
            if (!parentBlock) return;

            var sidebarRect = sidebar.getBoundingClientRect();

            var contentHost = sidebar.querySelector('[data-testid="stSidebarUserContent"]')
                || sidebar.querySelector('[data-testid="stSidebarContent"]')
                || sidebar.firstElementChild;

            if (contentHost) {
                contentHost.style.setProperty('display', 'flex', 'important');
                contentHost.style.setProperty('flex-direction', 'column', 'important');
                contentHost.style.setProperty('height', sidebarRect.height + 'px', 'important');
                contentHost.style.setProperty('padding-bottom', '0', 'important');
            }

            // Make the main vertical block a flex container to push the footer down
            parentBlock.style.setProperty('display', 'flex', 'important');
            parentBlock.style.setProperty('flex-direction', 'column', 'important');
            parentBlock.style.setProperty('height', '100%', 'important');
            parentBlock.style.setProperty('min-height', sidebarRect.height + 'px', 'important');
            parentBlock.style.setProperty('gap', '0', 'important');

            // Move the footer to the absolute bottom using margin-top: auto
            footerElement.style.setProperty('margin-top', 'auto', 'important');
            footerElement.style.setProperty('position', 'sticky', 'important');
            footerElement.style.setProperty('bottom', '14px', 'important');
            footerElement.style.setProperty('z-index', '1000', 'important');
            footerElement.style.setProperty('background', 'black', 'important');
        }

        function killChevrons() {
            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            var popoverButtons = sidebar.querySelectorAll('[data-testid="stPopover"] button');
            popoverButtons.forEach(function(btn) {
                var innerDiv = btn.querySelector('div:first-child');
                if (innerDiv && innerDiv.children.length > 1) {
                    var lastChild = innerDiv.lastElementChild;
                    // Check if the last child is likely the chevron (it's an SVG or contains one)
                    if (lastChild && (lastChild.tagName === 'SVG' || lastChild.querySelector('svg') || lastChild.getAttribute('data-testid') === 'stIcon')) {
                        lastChild.style.setProperty('display', 'none', 'important');
                    }
                }
            });
        }

        function runSidebarEnhancements() {
            try { syncStrip(); } catch(e){}
            try { cleanupLegacyStyles(); } catch(e){}
            try { injectSearchClear(); } catch(e){}
            try { killChevrons(); } catch(e){}
        }

        runSidebarEnhancements();

        var footerAttempts = 0;
        var footerFixTimer = setInterval(function() {
            footerAttempts += 1;
            try { fixSidebarFooter(); } catch(e){}
            if (footerAttempts >= 20) clearInterval(footerFixTimer);
        }, 120);

        // Fast bootstrap so clear button/search behavior is ready before first typing.
        var searchBootAttempts = 0;
        var searchBootTimer = setInterval(function() {
            searchBootAttempts += 1;
            try { injectSearchClear(); } catch(e){}
            if (searchBootAttempts >= 25) clearInterval(searchBootTimer);
        }, 100);

        // Lightweight maintenance loop
        setInterval(runSidebarEnhancements, 900);

        // Also fire immediately once DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                runSidebarEnhancements();
                try { fixSidebarFooter(); } catch(e){}
            });
        } else {
            runSidebarEnhancements();
            try { fixSidebarFooter(); } catch(e){}
        }
    })();
    </script>
    """, unsafe_allow_html=True)

    # Persistent disclaimer — always visible, compliant with ACS/ACM ethics requirements.
    st.markdown("""
    <div class="sticky-disclaimer">
        Not medical advice &mdash; always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
