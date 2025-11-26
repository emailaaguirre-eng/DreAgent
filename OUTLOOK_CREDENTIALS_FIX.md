# Outlook Credentials Disappearing - Fix Guide

## Problem
You keep adding `OUTLOOK_CLIENT_ID` and `OUTLOOK_TENANT_ID` to your `.env` file, but they keep disappearing or not being recognized.

## Root Causes

The most common reasons this happens:

1. **Syntax Errors in .env File** - The credentials are in the file but have syntax errors that prevent them from loading:
   - Missing `=` sign
   - Extra spaces around `=`
   - Quotes around values (sometimes causes issues)
   - Line continuation problems

2. **File Encoding Issues** - The .env file might have encoding problems

3. **Multiple .env Files** - There might be multiple .env files in different locations

4. **File Not Being Saved Properly** - The file might not be saving correctly

## Solutions

### Step 1: Run the Diagnostic Script

Run this to see exactly what's wrong:

```bash
python diagnose_env.py
```

This will show you:
- If the .env file exists
- If the credentials are in the file
- If they're loading correctly
- Any syntax errors

### Step 2: Fix the Issue

If the diagnostic shows issues, run:

```bash
python protect_outlook_credentials.py
```

This script will:
- Check your current .env file
- Fix common syntax errors
- Prompt you to add missing credentials
- Verify everything is working

### Step 3: Manual Fix

If the scripts don't work, manually check your `.env` file:

1. Open `F:\Dre_Programs\LeaAssistant\.env` in a text editor
2. Make sure your credentials are formatted correctly:

```
OUTLOOK_CLIENT_ID=your-client-id-here
OUTLOOK_TENANT_ID=your-tenant-id-here
```

**Important formatting rules:**
- No spaces around the `=` sign
- No quotes around the values (unless the value itself contains spaces)
- Each credential on its own line
- No trailing spaces

**Example of CORRECT format:**
```
OUTLOOK_CLIENT_ID=12345678-1234-1234-1234-123456789abc
OUTLOOK_TENANT_ID=af487047-8abc-1234-5678-123456789abc
```

**Example of INCORRECT format:**
```
OUTLOOK_CLIENT_ID = 12345678-1234-1234-1234-123456789abc  # ❌ Spaces around =
OUTLOOK_CLIENT_ID="12345678-1234-1234-1234-123456789abc"  # ❌ Quotes (may cause issues)
OUTLOOK_CLIENT_ID: 12345678-1234-1234-1234-123456789abc   # ❌ Using : instead of =
```

## What I've Fixed

I've made several improvements to help prevent this issue:

1. **Better Error Detection** - The code now detects when credentials exist in the file but aren't loading (syntax error)

2. **Enhanced Logging** - More detailed error messages showing exactly what's wrong

3. **Startup Validation** - The app now checks Outlook credentials on startup and warns you if there are issues

4. **Diagnostic Tools** - Two new scripts to help diagnose and fix the problem

## Prevention

To prevent this from happening again:

1. **Always use the diagnostic script** before reporting issues
2. **Check the logs** - The app now logs detailed information about credential loading
3. **Use the protection script** if you suspect issues
4. **Don't edit .env with programs that might reformat it** - Use a plain text editor

## Testing

After fixing, test by running:

```bash
python -c "from outlook_integration import OutlookClient; client = OutlookClient(); print('✅ Client initialized successfully' if client.client_id else '❌ Client ID missing')"
```

Or simply try to use the Outlook email check feature in Lea.

## Still Having Issues?

If the credentials still disappear:

1. Check if you have multiple .env files:
   ```bash
   dir /s F:\Dre_Programs\LeaAssistant\.env
   ```

2. Check file permissions - make sure the file isn't read-only

3. Check if any backup/restore process is overwriting it

4. Check the logs in `lea_crash.log` for detailed error messages

