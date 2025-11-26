"""
Outlook Integration using Microsoft Graph API
Provides access to Outlook email, calendar, and user profile
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logger first, before it's used
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
PROJECT_DIR = Path("F:/Dre_Programs/LeaAssistant")
env_file = PROJECT_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=True)
    logger.info(f"Loaded .env file from: {env_file}")
else:
    load_dotenv(override=True)
    logger.warning(f".env file not found at {env_file}, trying fallback locations")

try:
    from msal import ConfidentialClientApplication, PublicClientApplication
    import requests
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False
    logger.warning("msal or requests not installed. Install with: pip install msal requests")


class OutlookClient:
    """Microsoft Graph API client for Outlook operations"""
    
    def __init__(self):
        self.client_id = os.getenv("OUTLOOK_CLIENT_ID")
        self.client_secret = os.getenv("OUTLOOK_CLIENT_SECRET")  # Optional - for manual login
        self.tenant_id = os.getenv("OUTLOOK_TENANT_ID", "common")
        
        # Debug logging (without exposing secrets)
        env_file_path = PROJECT_DIR / ".env"
        if not self.client_id:
            logger.error("❌ OUTLOOK_CLIENT_ID not found in environment variables")
            logger.error(f"   Checked .env file at: {env_file_path}")
            logger.error(f"   .env file exists: {env_file_path.exists()}")
        else:
            logger.info(f"✅ OUTLOOK_CLIENT_ID found (length: {len(self.client_id)})")
        
        if not self.tenant_id or self.tenant_id == "common":
            logger.warning(f"⚠️  OUTLOOK_TENANT_ID not found, using default: 'common'")
            logger.warning(f"   Checked .env file at: {env_file_path}")
        else:
            logger.info(f"✅ OUTLOOK_TENANT_ID found: {self.tenant_id[:10]}...")
        
        if not self.client_secret:
            logger.info("ℹ️  OUTLOOK_CLIENT_SECRET not found (will use interactive login)")
        else:
            logger.info("✅ OUTLOOK_CLIENT_SECRET found")
        
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        # Use delegated permissions for interactive login (not .default)
        self.scope = ["https://graph.microsoft.com/Mail.Read", 
                     "https://graph.microsoft.com/Calendars.Read",
                     "https://graph.microsoft.com/User.Read"]
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
        
        self.app = None
        self.token = None
        self.token_cache = {}
        
        if MSAL_AVAILABLE and self.client_id:
            if self.client_secret:
                # Confidential client (app with secret)
                self.app = ConfidentialClientApplication(
                    client_id=self.client_id,
                    client_credential=self.client_secret,
                    authority=self.authority
                )
                # For confidential clients, use .default scope
                self.scope = ["https://graph.microsoft.com/.default"]
            else:
                # Public client (no secret - for interactive/manual login)
                self.app = PublicClientApplication(
                    client_id=self.client_id,
                    authority=self.authority
                )
                # For public clients, use specific delegated permissions
                self.scope = ["https://graph.microsoft.com/Mail.Read",
                             "https://graph.microsoft.com/Mail.ReadWrite",
                             "https://graph.microsoft.com/Calendars.Read",
                             "https://graph.microsoft.com/User.Read",
                             "https://graph.microsoft.com/User.ReadWrite"]
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        if not self.app:
            return False
        
        # Try to get token silently
        try:
            accounts = self.app.get_accounts()
            if accounts:
                result = self.app.acquire_token_silent(self.scope, account=accounts[0])
                if result and "access_token" in result:
                    self.token = result["access_token"]
                    return True
        except Exception as e:
            logger.debug(f"Silent token acquisition failed: {e}")
        
        return self.token is not None
    
    def authenticate(self, interactive: bool = True) -> bool:
        """Authenticate with Microsoft Graph"""
        if not self.app:
            logger.error("Outlook client not initialized. Check OUTLOOK_CLIENT_ID in .env")
            return False
        
        try:
            accounts = self.app.get_accounts()
            
            if accounts:
                # Try silent authentication first
                result = self.app.acquire_token_silent(self.scope, account=accounts[0])
                if result and "access_token" in result:
                    self.token = result["access_token"]
                    logger.info("✅ Authenticated silently")
                    return True
            
            # Interactive authentication
            if interactive:
                if self.client_secret:
                    # Client credentials flow (app-only)
                    result = self.app.acquire_token_for_client(scopes=self.scope)
                else:
                    # Public client - use interactive browser flow (best for manual login)
                    try:
                        # Try interactive browser flow first (opens browser window)
                        result = self.app.acquire_token_interactive(
                            scopes=self.scope,
                            # Optional: specify redirect URI if configured in Azure AD
                            # redirect_uri="http://localhost"  # Uncomment if you set this in Azure AD
                        )
                    except Exception as browser_error:
                        logger.info(f"Browser flow not available, trying device code flow: {browser_error}")
                        # Fallback to device code flow if browser flow fails
                        try:
                            flow = self.app.initiate_device_flow(scopes=self.scope)
                            if "user_code" in flow:
                                print(f"\n🔐 Please visit: {flow['verification_uri']}")
                                print(f"   Enter code: {flow['user_code']}\n")
                                result = self.app.acquire_token_by_device_flow(flow)
                            else:
                                logger.error("Device code flow initialization failed")
                                return False
                        except Exception as device_error:
                            logger.error(f"Device code flow failed: {device_error}")
                            return False
                
                if result and "access_token" in result:
                    self.token = result["access_token"]
                    logger.info("✅ Authentication successful")
                    return True
                else:
                    error = result.get("error_description", result.get("error", "Unknown error"))
                    logger.error(f"❌ Authentication failed: {error}")
                    return False
            else:
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make a request to Microsoft Graph API"""
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        url = f"{self.graph_endpoint}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                # Token expired, try to re-authenticate
                self.token = None
                if self.authenticate():
                    headers["Authorization"] = f"Bearer {self.token}"
                    response = requests.request(method, url, headers=headers, **kwargs)
                    response.raise_for_status()
                    return response.json() if response.content else {}
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    def get_emails(self, max_results: int = 50, folder: str = "inbox") -> List[Dict]:
        """Get emails from Outlook inbox"""
        endpoint = f"/me/mailFolders/{folder}/messages"
        params = {
            "$top": min(max_results, 100),
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,from,receivedDateTime,isRead,bodyPreview,body"
        }
        
        result = self._make_request("GET", endpoint, params=params)
        if not result:
            return []
        
        emails = []
        for msg in result.get("value", []):
            emails.append({
                "id": msg.get("id"),
                "subject": msg.get("subject", "(No Subject)"),
                "from": msg.get("from", {}).get("emailAddress", {}).get("address", "Unknown"),
                "received": msg.get("receivedDateTime"),
                "isRead": msg.get("isRead", False),
                "preview": msg.get("bodyPreview", ""),
                "body": msg.get("body", {}).get("content", "")
            })
        
        return emails
    
    def create_draft(self, subject: str, body: str, to: str = "", cc: str = "", bcc: str = "") -> Optional[str]:
        """Create a draft email in Outlook"""
        endpoint = "/me/messages"
        
        # Build recipients
        message = {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body
            }
        }
        
        if to:
            message["toRecipients"] = [{"emailAddress": {"address": addr.strip()}} for addr in to.split(",")]
        if cc:
            message["ccRecipients"] = [{"emailAddress": {"address": addr.strip()}} for addr in cc.split(",")]
        if bcc:
            message["bccRecipients"] = [{"emailAddress": {"address": addr.strip()}} for addr in bcc.split(",")]
        
        result = self._make_request("POST", endpoint, json=message)
        if result and "id" in result:
            return result["id"]
        return None
    
    def get_calendar_events(self, days_ahead: int = 30) -> List[Dict]:
        """Get calendar events for the next N days"""
        start_date = datetime.now().isoformat() + "Z"
        end_date = (datetime.now() + timedelta(days=days_ahead)).isoformat() + "Z"
        
        endpoint = "/me/calendar/calendarView"
        params = {
            "startDateTime": start_date,
            "endDateTime": end_date,
            "$select": "id,subject,start,end,location,organizer,attendees,bodyPreview"
        }
        
        result = self._make_request("GET", endpoint, params=params)
        if not result:
            return []
        
        events = []
        for event in result.get("value", []):
            events.append({
                "id": event.get("id"),
                "subject": event.get("subject", "(No Subject)"),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime"),
                "location": event.get("location", {}).get("displayName", ""),
                "organizer": event.get("organizer", {}).get("emailAddress", {}).get("address", ""),
                "preview": event.get("bodyPreview", "")
            })
        
        return events
    
    def create_organization_plan(self, folder: str = "inbox", rules: Dict = None) -> Dict:
        """Create a plan for organizing inbox"""
        # Get current email stats
        emails = self.get_emails(max_results=100, folder=folder)
        
        unread_count = sum(1 for e in emails if not e.get("isRead", False))
        total_count = len(emails)
        
        plan = {
            "folder": folder,
            "total_emails": total_count,
            "unread_emails": unread_count,
            "suggestions": []
        }
        
        if unread_count > 0:
            plan["suggestions"].append(f"Mark {unread_count} unread emails as read")
        
        if total_count > 50:
            plan["suggestions"].append("Archive old emails (older than 30 days)")
        
        return plan
    
    def organize_inbox(self, folder: str = "inbox", rules: Dict = None) -> Dict:
        """Organize inbox based on rules"""
        # This is a placeholder - implement actual organization logic
        result = {
            "folder": folder,
            "actions_taken": [],
            "emails_processed": 0
        }
        
        # Get emails
        emails = self.get_emails(max_results=100, folder=folder)
        
        # Apply organization rules
        if rules:
            # Implement custom rules
            pass
        
        return result
    
    def get_user_profile(self) -> Dict:
        """Get user profile information"""
        endpoint = "/me"
        result = self._make_request("GET", endpoint)
        
        if result:
            return {
                "id": result.get("id"),
                "displayName": result.get("displayName", ""),
                "mail": result.get("mail", ""),
                "userPrincipalName": result.get("userPrincipalName", ""),
                "jobTitle": result.get("jobTitle", ""),
                "officeLocation": result.get("officeLocation", ""),
                "department": result.get("department", "")
            }
        return {}
    
    def update_user_profile(self, updates: Dict) -> Dict:
        """Update user profile information"""
        endpoint = "/me"
        result = self._make_request("PATCH", endpoint, json=updates)
        return result if result else {}


def get_outlook_client() -> Optional[OutlookClient]:
    """Get or create Outlook client instance"""
    if not MSAL_AVAILABLE:
        logger.warning("msal not installed. Install with: pip install msal requests")
        return None
    
    client_id = os.getenv("OUTLOOK_CLIENT_ID")
    if not client_id:
        logger.error("❌ OUTLOOK_CLIENT_ID not found in .env file")
        logger.error(f"   Expected .env file location: {env_file}")
        logger.error(f"   .env file exists: {env_file.exists()}")
        logger.error("   Please ensure OUTLOOK_CLIENT_ID is set in your .env file")
        return None
    
    client = OutlookClient()
    
    # Log authentication method
    if client.client_secret:
        logger.info("Using confidential client (with secret)")
    else:
        logger.info("Using public client (manual/interactive login - no secret required)")
    
    return client

