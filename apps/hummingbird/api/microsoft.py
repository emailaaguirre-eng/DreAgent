import os
import time
from typing import Optional

import httpx
import msal
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

from core.utils.config import get_settings

settings = get_settings()
router = APIRouter()

# Expected env vars in /home/codrex/lea/.env:
# MICROSOFT_CLIENT_ID=
# MICROSOFT_CLIENT_SECRET=
# MICROSOFT_TENANT_ID=common   (or your tenant GUID)
# MICROSOFT_REDIRECT_URI=http(s)://<your-domain>/api/microsoft/callback

TOKEN_PATH = settings.memory_path / "microsoft_token.json"
_reports_base = getattr(settings, "data_path", Path("data"))
if isinstance(_reports_base, str):
    _reports_base = Path(_reports_base)
REPORTS_DIR = _reports_base / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


SCOPES = [
    "User.Read",
    "Mail.Read",
    "Calendars.Read",
]

def _confidential_client():
    client_id = os.getenv("MICROSOFT_CLIENT_ID")
    client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
    tenant = os.getenv("MICROSOFT_TENANT_ID", "common")

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Microsoft OAuth not configured (missing client id/secret).")

    authority = f"https://login.microsoftonline.com/{tenant}"
    return msal.ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=authority,
    )

def _load_token() -> Optional[dict]:
    try:
        if TOKEN_PATH.exists():
            import json
            return json.loads(TOKEN_PATH.read_text())
    except Exception:
        return None
    return None

def _save_token(token: dict) -> None:
    import json
    TOKEN_PATH.write_text(json.dumps(token, indent=2))


def _refresh_token_if_possible() -> Optional[dict]:
    token = _load_token()
    if not token or "refresh_token" not in token:
        return None

    cca = _confidential_client()
    refreshed = cca.acquire_token_by_refresh_token(
        refresh_token=token["refresh_token"],
        scopes=SCOPES,
    )
    if refreshed and "access_token" in refreshed:
        _save_token(refreshed)
        return refreshed
    return None

@router.get("/status")
async def status():
    token = _load_token()
    return {"connected": bool(token), "token_saved": bool(token)}

@router.get("/login")
async def login():
    redirect_uri = os.getenv("MICROSOFT_REDIRECT_URI")
    if not redirect_uri:
        raise HTTPException(status_code=500, detail="MICROSOFT_REDIRECT_URI not set.")

    cca = _confidential_client()
    auth_url = cca.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        prompt="select_account",
    )
    return RedirectResponse(auth_url)

@router.get("/callback")
async def callback(request: Request, code: Optional[str] = None, error: Optional[str] = None, error_description: Optional[str] = None):
    if error:
        return JSONResponse({"ok": False, "error": error, "error_description": error_description}, status_code=400)

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    redirect_uri = os.getenv("MICROSOFT_REDIRECT_URI")
    cca = _confidential_client()

    result = cca.acquire_token_by_authorization_code(
        code=code,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )

    if "access_token" not in result:
        return JSONResponse({"ok": False, "result": result}, status_code=400)

    # Store token payload (includes refresh token for confidential client flows where applicable)
    _save_token(result)
    return {"ok": True, "message": "Microsoft connected. Token saved.", "saved_at": int(time.time())}

async def _graph_get(path: str) -> dict:
    token = _load_token()
    if not token or "access_token" not in token:
        raise HTTPException(status_code=401, detail="Not connected to Microsoft.")

    headers = {"Authorization": f"Bearer {token['access_token']}"}
    url = f"https://graph.microsoft.com/v1.0{path}"

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=headers)

    if r.status_code == 401:
        refreshed = _refresh_token_if_possible()
        if refreshed and "access_token" in refreshed:
            headers = {"Authorization": f"Bearer {refreshed['access_token']}"}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url, headers=headers)
        else:
            raise HTTPException(status_code=401, detail="Microsoft token expired or invalid. Reconnect.")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()

@router.get("/inbox")
async def inbox(top: int = 10):
    # Read-only: most recent messages
    data = await _graph_get(
	f"/me/mailFolders/inbox/messages?$top={top}"
	f"&$select=id,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,bodyPreview"
	f"&$orderby=receivedDateTime desc"
)
    return data

@router.get("/calendar")
async def calendar(top: int = 10):
    # Read-only: next events
    data = await _graph_get(f"/me/events?$top={top}&$select=subject,organizer,start,end,location")
    return data
def _flatten_email_row(e: dict) -> dict:
    sender = ((e.get("from") or {}).get("emailAddress") or {}).get("address", "")
    sender_name = ((e.get("from") or {}).get("emailAddress") or {}).get("name", "")
    subject = e.get("subject", "")
    received = e.get("receivedDateTime", "")
    is_read = e.get("isRead", False)
    preview = (e.get("bodyPreview", "") or "").replace("\n", " ").replace("\r", " ")
    return {
        "receivedDateTime": received,
        "from": sender,
        "fromName": sender_name,
        "subject": subject,
        "isRead": is_read,
        "bodyPreview": preview[:500],
    }


@router.get("/report/email")
async def generate_email_report(top: int = 100, unread_only: bool = False):
    filt = "&$filter=isRead eq false" if unread_only else ""
    data = await _graph_get(
        f"/me/mailFolders/inbox/messages?$top={top}"
        f"&$select=id,subject,from,receivedDateTime,isRead,bodyPreview"
        f"&$orderby=receivedDateTime desc{filt}"
    )
    items = (data or {}).get("value", [])

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"email_report_{ts}.csv"

    fieldnames = ["receivedDateTime", "from", "fromName", "subject", "isRead", "bodyPreview"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in items:
            writer.writerow(_flatten_email_row(e))

    return {
        "ok": True,
        "rows": len(items),
        "file": out.name,
        "download_url": f"/api/microsoft/report/email/download/{out.name}",
    }


@router.get("/report/email/download/{filename}")
async def download_email_report(filename: str):
    safe_name = Path(filename).name
    file_path = REPORTS_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=str(file_path),
        media_type="text/csv",
        filename=safe_name,
    )
