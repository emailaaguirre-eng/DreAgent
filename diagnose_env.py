"""
Diagnostic script to check .env file status and Outlook credentials
Run this to see why Outlook credentials might be disappearing
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path("F:/Dre_Programs/LeaAssistant")
ENV_FILE = PROJECT_DIR / ".env"

def diagnose_env():
    """Diagnose .env file issues"""
    print("=" * 70)
    print("ENV FILE DIAGNOSTIC - Outlook Credentials Check")
    print("=" * 70)
    print()
    
    # Check if file exists
    print(f"1. Checking .env file location: {ENV_FILE}")
    print(f"   File exists: {ENV_FILE.exists()}")
    print()
    
    if not ENV_FILE.exists():
        print("[ERROR] .env file does not exist!")
        print(f"   Please create it at: {ENV_FILE}")
        return
    
    # Read file content
    print("2. Reading .env file content...")
    try:
        with open(ENV_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        print(f"   File size: {len(content)} bytes")
        print(f"   Number of lines: {len(lines)}")
        print()
        
        # Check for Outlook credentials in file
        print("3. Checking for Outlook credentials in file content...")
        has_client_id = False
        has_tenant_id = False
        has_client_secret = False
        
        client_id_line = None
        tenant_id_line = None
        client_secret_line = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                continue
            
            if "OUTLOOK_CLIENT_ID" in line.upper():
                has_client_id = True
                client_id_line = (i, line)
                print(f"   [OK] Found OUTLOOK_CLIENT_ID on line {i}: {line[:60]}...")
            
            if "OUTLOOK_TENANT_ID" in line.upper():
                has_tenant_id = True
                tenant_id_line = (i, line)
                print(f"   [OK] Found OUTLOOK_TENANT_ID on line {i}: {line[:60]}...")
            
            if "OUTLOOK_CLIENT_SECRET" in line.upper():
                has_client_secret = True
                client_secret_line = (i, line)
                print(f"   [OK] Found OUTLOOK_CLIENT_SECRET on line {i}: {line[:60]}...")
        
        if not has_client_id:
            print("   [ERROR] OUTLOOK_CLIENT_ID NOT FOUND in file")
        if not has_tenant_id:
            print("   [WARN] OUTLOOK_TENANT_ID NOT FOUND in file (will use 'common')")
        if not has_client_secret:
            print("   [INFO] OUTLOOK_CLIENT_SECRET NOT FOUND (optional, will use interactive login)")
        print()
        
        # Check syntax errors
        print("4. Checking for syntax errors...")
        syntax_errors = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            if '=' not in stripped and any(key in stripped.upper() for key in ["OUTLOOK_CLIENT_ID", "OUTLOOK_TENANT_ID", "OUTLOOK_CLIENT_SECRET"]):
                syntax_errors.append((i, line, "Missing '=' sign"))
            elif '=' in stripped:
                parts = stripped.split('=', 1)
                if len(parts) != 2:
                    syntax_errors.append((i, line, "Invalid format"))
                elif not parts[0].strip():
                    syntax_errors.append((i, line, "Empty key name"))
        
        if syntax_errors:
            print("   [ERROR] Found syntax errors:")
            for line_num, line_content, error in syntax_errors:
                print(f"      Line {line_num}: {error}")
                print(f"      Content: {line_content[:80]}")
        else:
            print("   [OK] No syntax errors found")
        print()
        
        # Load environment variables
        print("5. Loading environment variables with python-dotenv...")
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        
        # Check if values are loaded
        print("6. Checking if values are loaded into environment...")
        client_id_loaded = os.getenv("OUTLOOK_CLIENT_ID")
        tenant_id_loaded = os.getenv("OUTLOOK_TENANT_ID")
        client_secret_loaded = os.getenv("OUTLOOK_CLIENT_SECRET")
        
        if has_client_id:
            if client_id_loaded:
                print(f"   [OK] OUTLOOK_CLIENT_ID loaded: {client_id_loaded[:20]}... (length: {len(client_id_loaded)})")
            else:
                print("   [ERROR] OUTLOOK_CLIENT_ID found in file but NOT loaded!")
                print("      This indicates a syntax error. Check the line format:")
                if client_id_line:
                    print(f"      Line {client_id_line[0]}: {client_id_line[1]}")
        else:
            print("   [ERROR] OUTLOOK_CLIENT_ID not in file and not loaded")
        
        if has_tenant_id:
            if tenant_id_loaded:
                print(f"   [OK] OUTLOOK_TENANT_ID loaded: {tenant_id_loaded[:20]}...")
            else:
                print("   [ERROR] OUTLOOK_TENANT_ID found in file but NOT loaded!")
                print("      This indicates a syntax error. Check the line format:")
                if tenant_id_line:
                    print(f"      Line {tenant_id_line[0]}: {tenant_id_line[1]}")
        else:
            if tenant_id_loaded:
                print(f"   [WARN] OUTLOOK_TENANT_ID not in file but loaded from elsewhere: {tenant_id_loaded}")
            else:
                print("   [WARN] OUTLOOK_TENANT_ID not found (will use default 'common')")
        
        if has_client_secret:
            if client_secret_loaded:
                print(f"   [OK] OUTLOOK_CLIENT_SECRET loaded (length: {len(client_secret_loaded)})")
            else:
                print("   [WARN] OUTLOOK_CLIENT_SECRET found in file but NOT loaded (this is optional)")
        else:
            print("   [INFO] OUTLOOK_CLIENT_SECRET not found (optional)")
        
        print()
        
        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if client_id_loaded and tenant_id_loaded:
            print("[OK] Outlook credentials are properly configured!")
        elif has_client_id and not client_id_loaded:
            print("[ERROR] OUTLOOK_CLIENT_ID is in file but not loading - syntax error likely")
            print("   Common issues:")
            print("   - Missing '=' sign")
            print("   - Extra spaces around '='")
            print("   - Quotes around the value (may cause issues)")
            print("   - Line continuation issues")
        elif not has_client_id:
            print("[ERROR] OUTLOOK_CLIENT_ID is missing from .env file")
            print("   Add this line to your .env file:")
            print("   OUTLOOK_CLIENT_ID=your_client_id_here")
        else:
            print("[WARN] Partial configuration - check details above")
        
        print()
        print("=" * 70)
        
    except Exception as e:
        print(f"[ERROR] Error reading .env file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_env()

