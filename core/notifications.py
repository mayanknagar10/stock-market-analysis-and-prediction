"""
Notifications — zero external accounts, zero API keys.

Two delivery channels, both genuinely free with no signup:

  1. In-app notification center — a persistent list in session_state,
     rendered as a bell icon with unread badge + dropdown. Works exactly
     like a real notification centre; survives page navigation within
     the same browser session.

  2. Browser desktop notifications — uses the native Web Notification
     API via a small embedded JavaScript snippet (st.components.v1.html).
     This is NOT a third-party push service — it's the browser's own
     built-in notification system, the same one every website uses when
     it asks "Allow notifications?". Zero account, zero backend, 100%
     client-side. Limitation (must be stated honestly): only fires while
     the browser tab is open — there's no way to alert you when the app
     isn't loaded without a real push-notification backend (which is a
     Phase-2-plus item requiring a service like Firebase Cloud Messaging).

Email alerts (Resend, SMTP, etc.) are intentionally NOT implemented here
since every real email path requires SOME account (even Gmail SMTP needs
an app password). See EMAIL_ALERTS_README below for the ready-to-activate
path once you're ready to add one.
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timezone
from typing import Dict, List, Optional
import html as _html

_MAX_NOTIFICATIONS = 50


# ─────────────────────────────────────────────────────────────────
# IN-APP NOTIFICATION CENTER
# ─────────────────────────────────────────────────────────────────

def _ensure_store():
    if "notifications" not in st.session_state:
        st.session_state["notifications"] = []
    if "notified_keys" not in st.session_state:
        # Dedup key set so the same alert condition doesn't spam a new
        # notification on every single Streamlit rerun (which happens on
        # nearly every widget interaction).
        st.session_state["notified_keys"] = set()


def push_notification(title: str, message: str, level: str = "info",
                      dedup_key: Optional[str] = None) -> bool:
    """
    Add a notification to the in-app center.

    dedup_key: if given, this notification only fires ONCE per session
    for that key (e.g. f"target_hit:{ticker}:{target_price}") — without
    this, a watchlist alert condition that's still true on every rerun
    would otherwise flood the notification list.

    Returns True if a new notification was actually added.
    """
    _ensure_store()
    if dedup_key:
        if dedup_key in st.session_state["notified_keys"]:
            return False
        st.session_state["notified_keys"].add(dedup_key)

    entry = {
        "title": title, "message": message, "level": level,
        "timestamp": datetime.now(timezone.utc), "read": False,
    }
    st.session_state["notifications"].insert(0, entry)
    st.session_state["notifications"] = st.session_state["notifications"][:_MAX_NOTIFICATIONS]
    return True


def get_notifications() -> List[Dict]:
    _ensure_store()
    return st.session_state["notifications"]


def unread_count() -> int:
    _ensure_store()
    return sum(1 for n in st.session_state["notifications"] if not n["read"])


def mark_all_read():
    _ensure_store()
    for n in st.session_state["notifications"]:
        n["read"] = True


def clear_notifications():
    _ensure_store()
    st.session_state["notifications"] = []
    st.session_state["notified_keys"] = set()


_LEVEL_COLORS = {
    "info": "#58A6FF", "success": "#3FB950",
    "warning": "#E3B341", "danger": "#F85149",
}
_LEVEL_ICONS = {"info": "ℹ️", "success": "🎯", "warning": "⚠️", "danger": "🛑"}


def notification_bell():
    """
    Renders a bell icon with unread badge in the sidebar, expanding into
    the full notification list on click. Call this once, typically right
    under sidebar_brand().
    """
    _ensure_store()
    n_unread = unread_count()
    label = f"🔔 Notifications ({n_unread})" if n_unread else "🔔 Notifications"

    with st.expander(label, expanded=False):
        notes = get_notifications()
        if not notes:
            st.caption("No notifications yet.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Mark all read", use_container_width=True, key="notif_read_all"):
                    mark_all_read()
                    st.rerun()
            with col_b:
                if st.button("Clear all", use_container_width=True, key="notif_clear_all"):
                    clear_notifications()
                    st.rerun()

            for i, n in enumerate(notes[:20]):
                color = _LEVEL_COLORS.get(n["level"], "#8B949E")
                icon = _LEVEL_ICONS.get(n["level"], "•")
                age = _format_age(n["timestamp"])
                opacity = "1" if not n["read"] else "0.55"
                st.markdown(
                    f'<div style="background:#161B22;border-left:3px solid {color};'
                    f'border-radius:4px;padding:8px 10px;margin-bottom:6px;opacity:{opacity}">'
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                    f'font-weight:600;color:#C9D1D9">{icon} {_html.escape(n["title"])}</div>'
                    f'<div style="font-size:11px;color:#8B949E;margin-top:2px">'
                    f'{_html.escape(n["message"])}</div>'
                    f'<div style="font-size:9px;color:#6E7681;margin-top:3px;'
                    f'font-family:\'IBM Plex Mono\',monospace">{age}</div>'
                    f'</div>', unsafe_allow_html=True)


def _format_age(ts: datetime) -> str:
    delta = datetime.now(timezone.utc) - ts
    secs = delta.total_seconds()
    if secs < 60: return "just now"
    if secs < 3600: return f"{int(secs//60)}m ago"
    if secs < 86400: return f"{int(secs//3600)}h ago"
    return f"{int(secs//86400)}d ago"


# ─────────────────────────────────────────────────────────────────
# BROWSER DESKTOP NOTIFICATIONS — native Web Notification API, no backend
# ─────────────────────────────────────────────────────────────────

def browser_notify(title: str, body: str, tag: Optional[str] = None):
    """
    Fires a real OS-level desktop notification via the browser's native
    Notification API. Pure client-side JavaScript — no service, no
    account, no server component. Requires the user to grant permission
    once (browser will prompt automatically on first call).

    tag: optional — browser notifications with the same tag replace each
    other instead of stacking, useful for "price update" style alerts.
    """
    safe_title = title.replace("`", "'").replace("\\", "")
    safe_body  = body.replace("`", "'").replace("\\", "")
    tag_js = f'"{tag}"' if tag else "undefined"

    components.html(
        f"""
        <script>
        (function() {{
            function fire() {{
                try {{
                    new Notification(`{safe_title}`, {{
                        body: `{safe_body}`,
                        tag: {tag_js},
                        icon: "https://cdn-icons-png.flaticon.com/512/2830/2830284.png"
                    }});
                }} catch (e) {{ /* Notification API unavailable — fail silent */ }}
            }}
            if (typeof Notification === "undefined") {{
                // Browser doesn't support it — nothing to do
            }} else if (Notification.permission === "granted") {{
                fire();
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(perm) {{
                    if (perm === "granted") fire();
                }});
            }}
        }})();
        </script>
        """,
        height=0, width=0,
    )


def notification_permission_prompt():
    """
    Renders a one-time visible button to request browser notification
    permission. Browsers block silent/automatic permission prompts on
    page load, so this needs an explicit user click to work reliably.
    """
    components.html(
        """
        <div style="font-family:'IBM Plex Mono',monospace">
        <button id="sp-notif-btn" style="
            background:#161B22;color:#3FB950;border:1px solid #3FB950;
            border-radius:6px;padding:8px 14px;font-size:12px;cursor:pointer;
            font-family:'IBM Plex Mono',monospace;width:100%;">
            🔔 Enable Desktop Alerts
        </button>
        <div id="sp-notif-status" style="font-size:10px;color:#8B949E;margin-top:4px"></div>
        </div>
        <script>
        const btn = document.getElementById('sp-notif-btn');
        const status = document.getElementById('sp-notif-status');
        function updateStatus() {
            if (typeof Notification === "undefined") {
                status.textContent = "Not supported in this browser.";
                btn.disabled = true;
            } else if (Notification.permission === "granted") {
                status.textContent = "✓ Enabled — alerts will pop up while this tab is open.";
                btn.textContent = "🔔 Alerts Enabled";
            } else if (Notification.permission === "denied") {
                status.textContent = "Blocked — enable in browser site settings.";
            }
        }
        updateStatus();
        btn.onclick = function() {
            Notification.requestPermission().then(function(perm) { updateStatus(); });
        };
        </script>
        """,
        height=70,
    )


# ─────────────────────────────────────────────────────────────────
# EMAIL ALERTS — not active, ready-to-wire when you have credentials
# ─────────────────────────────────────────────────────────────────

EMAIL_ALERTS_README = """
Email alerts need SOME account (even Gmail SMTP needs an "app password"),
so this is intentionally not wired up yet. When you're ready:

  Option A — Gmail SMTP (free, just needs an app password, no new signup):
    1. Google Account -> Security -> 2-Step Verification -> App passwords
    2. Generate a 16-character app password
    3. Add to core/notifications.py:

       import smtplib
       from email.mime.text import MIMEText

       def send_email_alert(to_addr, subject, body):
           msg = MIMEText(body)
           msg["Subject"] = subject
           msg["From"] = "your_email@gmail.com"
           msg["To"] = to_addr
           with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
               server.login("your_email@gmail.com", "your_16_char_app_password")
               server.send_message(msg)

  Option B — Resend (free tier, 3000 emails/month, from the roadmap):
    Sign up at resend.com, install `resend` package, then:

       import resend
       resend.api_key = "re_xxxxx"
       resend.Emails.send({
           "from": "alerts@yourdomain.com", "to": to_addr,
           "subject": subject, "html": body,
       })

Either way, call it from watchlist.py alongside push_notification() and
browser_notify() so all three channels fire together.
"""
