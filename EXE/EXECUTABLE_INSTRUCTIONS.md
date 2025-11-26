# Lea Assistant - Executable Setup Instructions

## For Recipients of the Executable

This executable contains a fully functional AI assistant that can be customized with your own name and preferences.

## First-Time Setup

### Step 1: Run the Executable
1. Double-click `LeaAssistant.exe` to start
2. On first run, you'll see an installation dialog

### Step 2: Customize Your Assistant
The installer will prompt you for:
- **Agent Name**: What you want to call your assistant (e.g., "Jack", "Alex", "Lea")
- **Your Name**: Your name (e.g., "Lu", "Sarah", "Dre")
- **Personality Description** (optional): How you want your assistant to behave

Click "Install" when done.

### Step 3: Create Your .env File
Create a file named `.env` in the same folder as `LeaAssistant.exe` with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
OUTLOOK_CLIENT_ID=your_outlook_client_id (optional - for email features)
OUTLOOK_CLIENT_SECRET=your_outlook_secret (optional - for email features)
OUTLOOK_TENANT_ID=your_tenant_id (optional - for email features)
```

**Important**: 
- You need your own OpenAI API key to use the assistant
- Get one at: https://platform.openai.com/api-keys
- Outlook credentials are only needed if you want email functionality

### Step 4: Run Again
Run `LeaAssistant.exe` again - it will use your customizations!

## What Gets Created

When you run the executable, it creates these files in the same folder:
- `agent_config.json` - Your customization settings
- `lea_settings.json` - Your preferences (voice, microphone, etc.)
- `lea_history.json` - Your conversation history
- `memory/` - Conversation memory database
- `downloads/` - Files downloaded by the assistant
- `backups/` - Automatic backups

## Features

The assistant includes:
- ✅ Multiple specialized modes (General, IT Support, Executive Assistant, etc.)
- ✅ Voice input/output (optional)
- ✅ Email management (if Outlook credentials provided)
- ✅ Screen automation for work tasks
- ✅ File operations and task automation
- ✅ Conversation memory

## Requirements

- **Windows 10/11** (64-bit)
- **No Python installation needed** - everything is bundled!
- **Internet connection** (for OpenAI API calls)
- **OpenAI API key** (required)

## Troubleshooting

### "OpenAI API key not found"
- Make sure you created a `.env` file in the same folder as the executable
- Check that the file is named exactly `.env` (not `.env.txt`)
- Verify your API key is correct

### Executable won't start
- Make sure you're on Windows 10/11 (64-bit)
- Try running as Administrator
- Check Windows Defender isn't blocking it

### First run is slow
- This is normal - the executable extracts files on first run
- Subsequent runs will be faster

### Want to change your customization
- Delete `agent_config.json` and run the executable again
- You'll be prompted to customize again

## Sharing Your Customized Version

If you want to share your customized assistant with someone else:
1. They need to create their own `.env` file with their API keys
2. They can customize it with their own names when they first run it
3. Your conversation history and settings stay on your computer

## Support

For issues or questions, contact the person who shared this executable with you.

