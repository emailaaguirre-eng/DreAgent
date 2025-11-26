"""
Protect Outlook credentials in .env file
Ensures OUTLOOK_CLIENT_ID and OUTLOOK_TENANT_ID are preserved
"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

PROJECT_DIR = Path("F:/Dre_Programs/LeaAssistant")
ENV_FILE = PROJECT_DIR / ".env"

def protect_outlook_credentials():
    """Ensure Outlook credentials are preserved in .env file"""
    
    print("=" * 70)
    print("PROTECT OUTLOOK CREDENTIALS")
    print("=" * 70)
    print()
    
    if not ENV_FILE.exists():
        print(f"❌ .env file not found at {ENV_FILE}")
        print("   Creating new .env file...")
        ENV_FILE.touch()
        print("   ✅ Created .env file")
        print("   Please add your Outlook credentials manually")
        return False
    
    # Load current .env
    load_dotenv(ENV_FILE)
    
    # Read file to check current state
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    client_id = os.getenv("OUTLOOK_CLIENT_ID")
    tenant_id = os.getenv("OUTLOOK_TENANT_ID")
    
    print("Current status:")
    print(f"  OUTLOOK_CLIENT_ID: {'✅ Found' if client_id else '❌ Missing'}")
    print(f"  OUTLOOK_TENANT_ID: {'✅ Found' if tenant_id else '❌ Missing'}")
    print()
    
    # Check if they exist in file but not loaded (syntax error)
    has_client_id_in_file = "OUTLOOK_CLIENT_ID" in content.upper()
    has_tenant_id_in_file = "OUTLOOK_TENANT_ID" in content.upper()
    
    if has_client_id_in_file and not client_id:
        print("⚠️  OUTLOOK_CLIENT_ID found in file but not loading!")
        print("   This suggests a syntax error. Fixing...")
        
        # Try to extract and fix
        for line in content.split('\n'):
            if "OUTLOOK_CLIENT_ID" in line.upper() and '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"').strip("'")
                    # Remove the problematic line and add correct one
                    new_content = []
                    for l in content.split('\n'):
                        if "OUTLOOK_CLIENT_ID" not in l.upper():
                            new_content.append(l)
                    new_content.append(f"OUTLOOK_CLIENT_ID={value}")
                    with open(ENV_FILE, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_content))
                    print(f"   ✅ Fixed OUTLOOK_CLIENT_ID line")
                    client_id = value
    
    if has_tenant_id_in_file and not tenant_id:
        print("⚠️  OUTLOOK_TENANT_ID found in file but not loading!")
        print("   This suggests a syntax error. Fixing...")
        
        # Try to extract and fix
        for line in content.split('\n'):
            if "OUTLOOK_TENANT_ID" in line.upper() and '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"').strip("'")
                    # Remove the problematic line and add correct one
                    new_content = []
                    for l in content.split('\n'):
                        if "OUTLOOK_TENANT_ID" not in l.upper():
                            new_content.append(l)
                    new_content.append(f"OUTLOOK_TENANT_ID={value}")
                    with open(ENV_FILE, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_content))
                    print(f"   ✅ Fixed OUTLOOK_TENANT_ID line")
                    tenant_id = value
    
    # If missing, prompt to add
    if not client_id:
        print("\n❌ OUTLOOK_CLIENT_ID is missing")
        response = input("Enter your OUTLOOK_CLIENT_ID (or press Enter to skip): ").strip()
        if response:
            set_key(ENV_FILE, "OUTLOOK_CLIENT_ID", response)
            print("✅ Added OUTLOOK_CLIENT_ID to .env file")
            client_id = response
    
    if not tenant_id:
        print("\n⚠️  OUTLOOK_TENANT_ID is missing")
        response = input("Enter your OUTLOOK_TENANT_ID (or press Enter to use 'common'): ").strip()
        if response:
            set_key(ENV_FILE, "OUTLOOK_TENANT_ID", response)
            print(f"✅ Added OUTLOOK_TENANT_ID={response} to .env file")
        else:
            set_key(ENV_FILE, "OUTLOOK_TENANT_ID", "common")
            print("✅ Added OUTLOOK_TENANT_ID=common to .env file")
    
    # Verify final state
    load_dotenv(ENV_FILE, override=True)
    final_client_id = os.getenv("OUTLOOK_CLIENT_ID")
    final_tenant_id = os.getenv("OUTLOOK_TENANT_ID")
    
    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    
    if final_client_id and final_tenant_id:
        print("✅ Outlook credentials are properly configured!")
        print(f"   OUTLOOK_CLIENT_ID: {final_client_id[:20]}... (length: {len(final_client_id)})")
        print(f"   OUTLOOK_TENANT_ID: {final_tenant_id}")
        return True
    else:
        print("❌ Configuration incomplete")
        if not final_client_id:
            print("   OUTLOOK_CLIENT_ID is still missing")
        if not final_tenant_id:
            print("   OUTLOOK_TENANT_ID is still missing")
        return False

if __name__ == "__main__":
    protect_outlook_credentials()

