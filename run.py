#!/usr/bin/env python3
"""
=============================================================================
HUMMINGBIRD-LEA - Startup Script
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Run this script to start Hummingbird-LEA.

Usage:
    python run.py
    
Or make it executable:
    chmod +x run.py
    ./run.py
=============================================================================
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_requirements():
    """Check if required packages are installed"""
    required = ['fastapi', 'uvicorn', 'httpx', 'pydantic']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:", ", ".join(missing))
        print("\nInstall them with:")
        print("    pip install -r requirements.txt")
        sys.exit(1)


def check_env_file():
    """Check if .env file exists"""
    env_file = PROJECT_ROOT / ".env"
    example_file = PROJECT_ROOT / "config" / ".env.example"
    
    if not env_file.exists():
        print("⚠️  No .env file found!")
        print("\nCreating .env from template...")
        
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            print("✅ Created .env file. Please edit it with your settings.")
            print(f"\n   Edit: {env_file}")
            print("\n   Important: Change ADMIN_PASSWORD and SECRET_KEY!")
        else:
            print("❌ Template not found. Please create .env manually.")
            sys.exit(1)


def check_ollama():
    """Check if Ollama is available"""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            print(f"✅ Ollama connected. Models: {', '.join(models) or 'None'}")
            
            # Check for required models
            required_models = ["llama3.1:8b", "deepseek-coder"]
            missing = [m for m in required_models if not any(m in model for model in models)]
            
            if missing:
                print(f"\n⚠️  Recommended models not found: {', '.join(missing)}")
                print("   Pull them with:")
                for m in missing:
                    print(f"      ollama pull {m}")
        else:
            print("⚠️  Ollama responded but may have issues")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("   AI features will be limited.")
        print("\n   To install Ollama:")
        print("      curl -fsSL https://ollama.com/install.sh | sh")
        print("   Then pull models:")
        print("      ollama pull llama3.1:8b")


def create_directories():
    """Ensure data directories exist"""
    directories = [
        PROJECT_ROOT / "data" / "knowledge",
        PROJECT_ROOT / "data" / "memory",
        PROJECT_ROOT / "data" / "uploads",
        PROJECT_ROOT / "data" / "templates",
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point"""
    print("=" * 60)
    print("🐦 HUMMINGBIRD-LEA")
    print("   Powered by CoDre-X | B & D Servicing LLC")
    print("=" * 60)
    print()
    
    # Run checks
    print("Running startup checks...\n")
    
    check_requirements()
    check_env_file()
    create_directories()
    check_ollama()
    
    print()
    print("=" * 60)
    print("Starting server...")
    print("=" * 60)
    print()
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=True)
    except ImportError:
        pass
    
    # Get settings
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📖 API Docs: http://{host}:{port}/docs")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Run the server
    import uvicorn
    uvicorn.run(
        "apps.hummingbird.main:app",
        host=host,
        port=port,
        reload=debug,
    )


if __name__ == "__main__":
    main()
