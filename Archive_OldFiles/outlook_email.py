"""
Outlook/Microsoft Graph Email Integration
Uses MSAL (Microsoft Authentication Library) for OAuth2 authentication
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from msal import ConfidentialClientApplication, PublicClientApplication
    MSAL_AVAILABLE = True
except ImportError as e:
    MSAL_AVAILABLE = False
    import sys
    error_msg = f"MSAL not available. Install with: pip install msal\nPython path: {sys.executable}\nError: {e}"
    logging.warning(error_msg)
    # Only print once to avoid spam
    if not hasattr(sys, '_msal_warning_shown'):
        print(f"⚠️ {error_msg}")
        sys._msal_warning_shown = True

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available. Install with: pip install requests")

# Configuration
CLIENT_ID = os.getenv("OUTLOOK_CLIENT_ID")
# For personal Microsoft accounts (Outlook.com, Hotmail, etc.), use "common" or "consumers"
# For work/school accounts, use the specific tenant ID
TENANT_ID = os.getenv("OUTLOOK_TENANT_ID", "common")
# If tenant ID looks like a GUID but user has personal account, default to "common"
if TENANT_ID and len(TENANT_ID) == 36 and TENANT_ID.count('-') == 4:
    # It's a GUID - but for personal accounts, we should use "common"
    # User can override by explicitly setting OUTLOOK_TENANT_ID=common
    pass  # Keep the user's setting, but they may need to change it
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["https://graph.microsoft.com/Mail.Read", "https://graph.microsoft.com/Mail.Send"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# Token cache file
TOKEN_CACHE_FILE = Path(__file__).parent / "outlook_token_cache.json"


class OutlookEmailClient:
    """Microsoft Graph API email client using OAuth2"""
    
    def __init__(self, client_id: Optional[str] = None, tenant_id: Optional[str] = None):
        self.client_id = client_id or CLIENT_ID
        self.tenant_id = tenant_id or TENANT_ID
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scopes = SCOPES
        self.token_cache_file = TOKEN_CACHE_FILE
        self.access_token = None
        self.account = None
        
        if not MSAL_AVAILABLE:
            raise ImportError("MSAL library not available. Install with: pip install msal")
        
        if not self.client_id:
            raise ValueError("OUTLOOK_CLIENT_ID not found in environment variables")
        
        # Initialize MSAL app (PublicClientApplication for desktop apps)
        self.app = PublicClientApplication(
            client_id=self.client_id,
            authority=self.authority
        )
        
        # Load token cache
        self._load_token_cache()
    
    def _load_token_cache(self):
        """Load token cache from file"""
        if self.token_cache_file.exists():
            try:
                with open(self.token_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.app.token_cache.deserialize(json.dumps(cache_data))
            except Exception as e:
                logging.warning(f"Could not load token cache: {e}")
    
    def _save_token_cache(self):
        """Save token cache to file"""
        try:
            cache_data = json.loads(self.app.token_cache.serialize())
            with open(self.token_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save token cache: {e}")
    
    def authenticate(self, interactive: bool = True) -> bool:
        """
        Authenticate with Microsoft Graph
        
        Args:
            interactive: If True, will open browser for login if needed
        
        Returns:
            True if authenticated successfully
        """
        # Try to get token from cache first
        accounts = self.app.get_accounts()
        if accounts:
            # Try silent token acquisition
            result = self.app.acquire_token_silent(self.scopes, account=accounts[0])
            if result and "access_token" in result:
                self.access_token = result["access_token"]
                self.account = accounts[0]
                logging.info("✅ Authenticated using cached token")
                return True
        
        # If no cached token or silent acquisition failed, do interactive login
        if interactive:
            logging.info("🔐 Opening browser for Microsoft login...")
            print("\n" + "="*60)
            print("OUTLOOK EMAIL AUTHENTICATION")
            print("="*60)
            print("A browser window should open shortly.")
            print("Please sign in with your Microsoft/Outlook account.")
            print("After signing in, you can close the browser window.")
            print("="*60 + "\n")
            
            try:
                result = self.app.acquire_token_interactive(scopes=self.scopes)
            except Exception as e:
                logging.error(f"❌ Error during authentication: {e}")
                print(f"\n❌ Authentication error: {e}")
                print("\nTroubleshooting:")
                print("1. Make sure OUTLOOK_CLIENT_ID is set in .env")
                print("2. Check that the Azure app is registered correctly")
                print("3. Verify internet connection")
                return False
            
            if result and "access_token" in result:
                self.access_token = result["access_token"]
                self.account = result.get("account")
                self._save_token_cache()
                logging.info("✅ Authentication successful!")
                print("\n✅ Authentication successful! You can now use email features.\n")
                return True
            else:
                error = result.get("error_description", result.get("error", "Unknown error"))
                error_code = result.get("error")
                logging.error(f"❌ Authentication failed: {error}")
                print(f"\n❌ Authentication failed: {error}")
                
                if error_code == "invalid_client":
                    print("\n⚠️  Invalid client ID. Check OUTLOOK_CLIENT_ID in .env file.")
                elif error_code == "access_denied":
                    print("\n⚠️  Access denied. Make sure you granted permissions in Azure AD.")
                elif error_code == "invalid_grant":
                    print("\n⚠️  Invalid grant. Try authenticating again.")
                else:
                    print(f"\n⚠️  Error code: {error_code}")
                    print("Please check your Azure app registration and try again.")
                
                return False
        
        return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make a request to Microsoft Graph API"""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not available")
        
        url = f"{GRAPH_API_ENDPOINT}{endpoint}"
        headers = self._get_headers()
        
        try:
            # Add timeout to prevent hanging
            timeout = kwargs.pop('timeout', 30)  # Default 30 second timeout
            response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logging.error(f"Request timed out after {timeout} seconds: {url}")
            raise ValueError(f"Request timed out. The email service may be slow or unavailable. Please try again.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to refresh
                logging.warning("Token expired, attempting to refresh...")
                if self.authenticate(interactive=False):
                    # Retry request with timeout
                    timeout = kwargs.pop('timeout', 30)
                    headers = self._get_headers()
                    response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
                    response.raise_for_status()
                    return response.json()
                else:
                    raise ValueError("Authentication expired and refresh failed. Please re-authenticate.")
            elif e.response.status_code == 429:
                # Rate limiting
                logging.warning("Rate limit exceeded. Waiting before retry...")
                import time
                time.sleep(2)  # Wait 2 seconds
                timeout = kwargs.pop('timeout', 30)
                response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
                response.raise_for_status()
                return response.json()
            else:
                error_detail = ""
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get('error', {}).get('message', str(e))
                except:
                    error_detail = str(e)
                logging.error(f"API request failed: {e.response.status_code} - {error_detail}")
                raise ValueError(f"API request failed: {error_detail}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error making API request: {e}")
            raise ValueError(f"Network error: {str(e)}. Check your internet connection.")
        except Exception as e:
            logging.error(f"Error making API request: {e}")
            raise
    
    def check_email(self, max_results: int = 10, unread_only: bool = True, get_all: bool = False) -> List[Dict]:
        """
        Check email inbox
        
        Args:
            max_results: Maximum number of emails to retrieve (ignored if get_all=True)
            unread_only: If True, only get unread emails
            get_all: If True, retrieve ALL emails (uses pagination)
        
        Returns:
            List of email dictionaries
        """
        if not self.access_token:
            if not self.authenticate():
                raise ValueError("Authentication required")
        
        # Build filter
        filter_str = "$filter=isRead eq false" if unread_only else ""
        select_str = "$select=id,subject,sender,receivedDateTime,bodyPreview,isRead,importance,hasAttachments"
        orderby_str = "$orderby=receivedDateTime desc"
        
        all_emails = []
        page_size = 999  # Microsoft Graph API max per page
        
        if get_all:
            # Use pagination to get all emails
            next_link = None
            page_count = 0
            
            while True:
                if next_link:
                    # Use the next link from previous response
                    # Extract the query part from the full URL
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(next_link)
                    query = parsed.path + "?" + parsed.query
                else:
                    # First page
                    top_str = f"$top={page_size}"
                    query = f"/me/mailFolders/inbox/messages?{select_str}&{orderby_str}&{top_str}"
                    if filter_str:
                        query += f"&{filter_str}"
                
                try:
                    data = self._make_request("GET", query)
                    emails = data.get("value", [])
                    
                    if not emails:
                        break
                    
                    # Format emails for easier use
                    for email in emails:
                        all_emails.append({
                            "id": email.get("id"),
                            "subject": email.get("subject", "(No Subject)"),
                            "sender": email.get("sender", {}).get("emailAddress", {}).get("address", "Unknown"),
                            "sender_name": email.get("sender", {}).get("emailAddress", {}).get("name", "Unknown"),
                            "received": email.get("receivedDateTime"),
                            "preview": email.get("bodyPreview", ""),
                            "is_read": email.get("isRead", False),
                            "importance": email.get("importance", "normal"),
                            "has_attachments": email.get("hasAttachments", False)
                        })
                    
                    page_count += 1
                    logging.info(f"Retrieved {len(all_emails)} emails so far (page {page_count})...")
                    
                    # Check if there are more pages
                    next_link = data.get("@odata.nextLink")
                    if not next_link or len(emails) < page_size:
                        break
                    
                except Exception as e:
                    logging.error(f"Error retrieving email page: {e}")
                    break
            
            logging.info(f"Total emails retrieved: {len(all_emails)}")
            return all_emails
        else:
            # Get limited results (original behavior)
            top_str = f"$top={max_results}"
            query = f"/me/mailFolders/inbox/messages?{select_str}&{orderby_str}&{top_str}"
            if filter_str:
                query += f"&{filter_str}"
            
            try:
                data = self._make_request("GET", query)
                emails = data.get("value", [])
                
                # Format emails for easier use
                formatted_emails = []
                for email in emails:
                    formatted_emails.append({
                        "id": email.get("id"),
                        "subject": email.get("subject", "(No Subject)"),
                        "sender": email.get("sender", {}).get("emailAddress", {}).get("address", "Unknown"),
                        "sender_name": email.get("sender", {}).get("emailAddress", {}).get("name", "Unknown"),
                        "received": email.get("receivedDateTime"),
                        "preview": email.get("bodyPreview", ""),
                        "is_read": email.get("isRead", False),
                        "importance": email.get("importance", "normal"),
                        "has_attachments": email.get("hasAttachments", False)
                    })
                
                return formatted_emails
        except Exception as e:
            logging.error(f"Error checking email: {e}")
            raise
    
    def get_email(self, email_id: str) -> Dict:
        """
        Get full email content by ID
        
        Args:
            email_id: Email message ID
        
        Returns:
            Email dictionary with full content
        """
        if not self.access_token:
            if not self.authenticate():
                raise ValueError("Authentication required")
        
        try:
            email = self._make_request("GET", f"/me/messages/{email_id}")
            
            return {
                "id": email.get("id"),
                "subject": email.get("subject", "(No Subject)"),
                "sender": email.get("sender", {}).get("emailAddress", {}).get("address", "Unknown"),
                "sender_name": email.get("sender", {}).get("emailAddress", {}).get("name", "Unknown"),
                "received": email.get("receivedDateTime"),
                "body": email.get("body", {}).get("content", ""),
                "body_type": email.get("body", {}).get("contentType", "text"),
                "to_recipients": [r.get("emailAddress", {}).get("address") for r in email.get("toRecipients", [])],
                "cc_recipients": [r.get("emailAddress", {}).get("address") for r in email.get("ccRecipients", [])],
                "is_read": email.get("isRead", False),
                "importance": email.get("importance", "normal"),
                "has_attachments": email.get("hasAttachments", False)
            }
        except Exception as e:
            logging.error(f"Error getting email: {e}")
            raise
    
    def send_email(self, to: str, subject: str, body: str, body_type: str = "HTML", 
                   cc: Optional[List[str]] = None) -> bool:
        """
        Send an email
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            body_type: "HTML" or "Text"
            cc: Optional list of CC recipients
        
        Returns:
            True if sent successfully
        """
        if not self.access_token:
            if not self.authenticate():
                raise ValueError("Authentication required")
        
        # Build message
        message = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": body_type,
                    "content": body
                },
                "toRecipients": [
                    {"emailAddress": {"address": addr}}
                    for addr in to.split(",") if addr.strip()
                ]
            }
        }
        
        if cc:
            message["message"]["ccRecipients"] = [
                {"emailAddress": {"address": addr}}
                for addr in cc if addr.strip()
            ]
        
        try:
            self._make_request("POST", "/me/sendMail", json=message)
            logging.info(f"✅ Email sent to {to}")
            return True
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            raise
    
    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read"""
        if not self.access_token:
            if not self.authenticate():
                raise ValueError("Authentication required")
        
        try:
            self._make_request("PATCH", f"/me/messages/{email_id}", json={"isRead": True})
            return True
        except Exception as e:
            logging.error(f"Error marking email as read: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated"""
        return self.access_token is not None


# Global client instance
_email_client = None

def get_email_client() -> Optional[OutlookEmailClient]:
    """Get or create the global email client"""
    global _email_client
    
    if not MSAL_AVAILABLE:
        import sys
        error_msg = f"MSAL library not available.\nPython: {sys.executable}\nInstall with: {sys.executable} -m pip install msal"
        logging.warning(error_msg)
        if not hasattr(sys, '_msal_error_shown'):
            print(f"⚠️ {error_msg}")
            sys._msal_error_shown = True
        return None
    
    if not REQUESTS_AVAILABLE:
        logging.warning("requests library not available. Install with: pip install requests")
        print("⚠️ requests library not available. Install with: pip install requests")
        return None
    
    if not CLIENT_ID:
        error_msg = """❌ OUTLOOK_CLIENT_ID not found in environment variables.

To fix this:
1. Open your .env file in the LeaAssistant directory
2. Add this line: OUTLOOK_CLIENT_ID=your_client_id_here
3. Get your Client ID from Azure Portal (https://portal.azure.com):
   - Go to Azure Active Directory > App registrations
   - Select or create your app
   - Copy the "Application (client) ID"
   - Paste it after OUTLOOK_CLIENT_ID=

See OUTLOOK_EMAIL_SETUP.md for detailed instructions."""
        logging.warning(error_msg)
        print(error_msg)
        return None
    
    if _email_client is None:
        try:
            _email_client = OutlookEmailClient()
            logging.info("✅ Email client created successfully")
        except ValueError as ve:
            # This is the expected error when CLIENT_ID is missing
            logging.error(f"Failed to create email client: {ve}")
            print(f"❌ {ve}")
            return None
        except Exception as e:
            logging.error(f"Failed to create email client: {e}")
            print(f"❌ Failed to create email client: {e}")
            return None
    
    return _email_client


def check_email_setup() -> Dict[str, Any]:
    """Check email setup and return diagnostic information"""
    diagnostics = {
        "msal_available": MSAL_AVAILABLE,
        "requests_available": REQUESTS_AVAILABLE,
        "client_id_set": bool(CLIENT_ID),
        "client_id_value": CLIENT_ID if CLIENT_ID else None,
        "tenant_id": TENANT_ID,
        "ready": False,
        "issues": []
    }
    
    if not MSAL_AVAILABLE:
        diagnostics["issues"].append("MSAL library not installed. Run: pip install msal")
    
    if not REQUESTS_AVAILABLE:
        diagnostics["issues"].append("requests library not installed. Run: pip install requests")
    
    if not CLIENT_ID:
        diagnostics["issues"].append("OUTLOOK_CLIENT_ID not set in .env file")
    
    if diagnostics["msal_available"] and diagnostics["requests_available"] and diagnostics["client_id_set"]:
        diagnostics["ready"] = True
    
    return diagnostics

