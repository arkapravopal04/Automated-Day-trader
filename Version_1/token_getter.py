"""
token_getter.py — run this once every morning before training or going live.

What it does:
    1. Opens the Upstox login page in your browser
    2. You log in and get redirected to a URL
    3. You paste that URL here
    4. It fetches your access token and writes it directly into upstox_data.py

Run:
    python token_getter.py

You only need your API_KEY and API_SECRET once — paste them below.
Get them from: https://account.upstox.com/developer/apps
"""

import re
import os
import sys
import webbrowser
import requests

# ── paste these once, they never change ───────────────────────────────────────
API_KEY    = "YOUR_API_KEY_HERE"
API_SECRET = "YOUR_API_SECRET_HERE"
REDIRECT   = "http://localhost/"     # must exactly match your Upstox app settings

# path to your data file — update if yours is in a different location
UPSTOX_DATA_FILE = "live_data.py"

# ─────────────────────────────────────────────────────────────────────────────

def get_token() -> str:
    # step 1 — open login page
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code&client_id={API_KEY}&redirect_uri={REDIRECT}"
    )
    print("\n[1] Opening Upstox login in your browser...")
    print(f"    If it doesn't open automatically, go to:\n    {auth_url}\n")
    webbrowser.open(auth_url)

    # step 2 — user pastes redirect URL
    print("[2] After logging in, you'll be redirected to a URL like:")
    print("    http://localhost/?code=XXXXXXXXXXXXXXXX\n")
    redirect_url = input("    Paste the full redirect URL here: ").strip()

    # step 3 — extract code from URL
    match = re.search(r"[?&]code=([^&]+)", redirect_url)
    if not match:
        print("\n[ERROR] Could not find 'code' in the URL. Make sure you copied the full URL.")
        sys.exit(1)
    code = match.group(1)
    print(f"\n[3] Authorization code extracted: {code[:8]}...")

    # step 4 — exchange code for token
    print("[4] Fetching access token...")
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        data={
            "code"         : code,
            "client_id"    : API_KEY,
            "client_secret": API_SECRET,
            "redirect_uri" : REDIRECT,
            "grant_type"   : "authorization_code",
        }
    )

    if resp.status_code != 200:
        print(f"\n[ERROR] Token request failed: {resp.status_code} — {resp.text}")
        sys.exit(1)

    token = resp.json().get("access_token")
    if not token:
        print(f"\n[ERROR] No access_token in response: {resp.json()}")
        sys.exit(1)

    print(f"[4] Token received: {token[:12]}...")
    return token


def write_token(token: str):
    if not os.path.exists(UPSTOX_DATA_FILE):
        print(f"\n[ERROR] {UPSTOX_DATA_FILE} not found in current directory.")
        print(f"        Run token_getter.py from the same folder as {UPSTOX_DATA_FILE}")
        sys.exit(1)

    with open(UPSTOX_DATA_FILE, "r") as f:
        content = f.read()

    # replace whatever is currently in ACCESS_TOKEN = "..."
    updated = re.sub(
        r'ACCESS_TOKEN\s*=\s*"[^"]*"',
        f'ACCESS_TOKEN = "{token}"',
        content
    )

    if updated == content:
        print(f"\n[ERROR] Could not find ACCESS_TOKEN = \"...\" in {UPSTOX_DATA_FILE}")
        print(f"        Make sure the line exists exactly as: ACCESS_TOKEN = \"...\"")
        sys.exit(1)

    with open(UPSTOX_DATA_FILE, "w") as f:
        f.write(updated)

    print(f"[5] Token written to {UPSTOX_DATA_FILE} — you're good to go!\n")


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Open token_getter.py and paste your API_KEY and API_SECRET first.")
        sys.exit(1)

    token = get_token()
    write_token(token)
    print("Done. Run train.py or upstox_live_runner.py now.")