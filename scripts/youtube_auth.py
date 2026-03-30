#!/usr/bin/env python3
"""
scripts/youtube_auth.py
─────────────────────────────────────────────────────────────────────────────
One-time YouTube OAuth 2.0 authorisation flow.

Run this ONCE on your local machine (not in Docker) to generate the
youtube_token.json file that the autonomous upload agent uses at runtime.

Prerequisites:
  pip install google-auth-oauthlib google-auth-httplib2

Steps:
  1. Download OAuth 2.0 client credentials from Google Cloud Console:
       APIs & Services → Credentials → Create Credentials → OAuth client ID
       Application type: Desktop app
       Save as: secrets/youtube_client_secrets.json

  2. Run this script:
       python scripts/youtube_auth.py

  3. A browser window will open. Log in with the YouTube channel account
     and grant the requested permissions.

  4. The token is saved to secrets/youtube_token.json
     Mount this file into the Docker container at /secrets/youtube_token.json

The token contains a refresh_token and never expires as long as it is used
at least once every 6 months. The upload agent auto-refreshes the access
token on each run.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

SECRETS_DIR   = Path(__file__).parent.parent / "secrets"
CLIENT_FILE   = SECRETS_DIR / "youtube_client_secrets.json"
TOKEN_FILE    = SECRETS_DIR / "youtube_token.json"


def main() -> None:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: Run:  pip install google-auth-oauthlib")
        sys.exit(1)

    if not CLIENT_FILE.exists():
        print(f"ERROR: Client secrets not found at {CLIENT_FILE}")
        print("Download from: Google Cloud Console → APIs & Services → Credentials")
        sys.exit(1)

    print(f"Starting OAuth flow using: {CLIENT_FILE}")
    print("A browser window will open — log in and grant permissions.\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)

    token_data = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes),
    }

    SECRETS_DIR.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
    os.chmod(TOKEN_FILE, 0o600)   # Owner read/write only

    print(f"\n✅ Token saved to: {TOKEN_FILE}")
    print("Mount this file into your Docker container:")
    print(f"  {TOKEN_FILE} → /secrets/youtube_token.json")
    print("\nThe upload agent will auto-refresh the access token on each run.")


if __name__ == "__main__":
    main()
