"""
Protect .env file from being overwritten
This script ensures OUTLOOK_TENANT_ID doesn't disappear
"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

PROJECT_DIR = Path("F:/Dre_Programs/LeaAssistant")
ENV_FILE = PROJECT_DIR / ".env"

def protect_tenant_id():
    """Ensure OUTLOOK_TENANT_ID is preserved in .env file"""
    
    if not ENV_FILE.exists():
        print(f".env file not found at {ENV_FILE}")
        return False
    
    # Load current .env
    load_dotenv(ENV_FILE)
    
    # Check if tenant_id exists
    tenant_id = os.getenv("OUTLOOK_TENANT_ID")
    
    if not tenant_id:
        print("⚠️ OUTLOOK_TENANT_ID is missing from .env file")
        print("\nTo fix this:")
        print("1. Open the .env file in a text editor")
        print("2. Add this line:")
        print("   OUTLOOK_TENANT_ID=common")
        print("   (or your specific tenant ID if you have one)")
        print("3. Save the file")
        return False
    else:
        print(f"✅ OUTLOOK_TENANT_ID is present: {tenant_id}")
        return True

def add_tenant_id_if_missing(default_value="common"):
    """Add OUTLOOK_TENANT_ID to .env if it's missing"""
    
    if not ENV_FILE.exists():
        print(f".env file not found at {ENV_FILE}")
        return False
    
    # Load current .env
    load_dotenv(ENV_FILE)
    
    # Check if tenant_id exists
    tenant_id = os.getenv("OUTLOOK_TENANT_ID")
    
    if not tenant_id:
        print(f"Adding OUTLOOK_TENANT_ID={default_value} to .env file...")
        try:
            # Read current file
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it's already there but commented or malformed
            if "OUTLOOK_TENANT_ID" in content:
                print("⚠️ OUTLOOK_TENANT_ID found in file but not being read - check for syntax errors")
                return False
            
            # Append tenant_id
            with open(ENV_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\nOUTLOOK_TENANT_ID={default_value}\n")
            
            print(f"✅ Added OUTLOOK_TENANT_ID={default_value} to .env file")
            return True
        except Exception as e:
            print(f"❌ Error adding tenant_id: {e}")
            return False
    else:
        print(f"✅ OUTLOOK_TENANT_ID already exists: {tenant_id}")
        return True

if __name__ == "__main__":
    print("="*60)
    print("PROTECT .ENV FILE - Check OUTLOOK_TENANT_ID")
    print("="*60)
    print()
    
    if protect_tenant_id():
        print("\n✅ .env file is protected")
    else:
        print("\n❌ .env file needs attention")
        response = input("\nWould you like to add OUTLOOK_TENANT_ID=common? (y/n): ")
        if response.lower() == 'y':
            add_tenant_id_if_missing()
        else:
            print("\nPlease manually add OUTLOOK_TENANT_ID to your .env file")

