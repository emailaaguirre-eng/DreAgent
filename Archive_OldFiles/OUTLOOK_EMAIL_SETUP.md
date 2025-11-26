# Outlook Email Integration Setup Guide

This guide will help you set up Outlook email integration for Lea Assistant, allowing you to check, read, and send emails through Microsoft Outlook (including Outlook.com and Office 365).

## Prerequisites

1. **Install required Python packages:**
   ```bash
   pip install msal requests python-dotenv
   ```

2. **A Microsoft account** (Outlook.com, Office 365, or Microsoft 365)

## Step 1: Register an Azure AD Application

Since you use Outlook online, you need to register an app in Azure AD to get a Client ID:

1. **Go to Azure Portal:**
   - Visit: https://portal.azure.com
   - Sign in with your Microsoft account
   - **For personal accounts:** If you don't have an Azure account, sign up for free at https://azure.microsoft.com/free/
   - **Important:** Make sure you're signed in with the SAME account you use for Outlook.com

2. **Navigate to App Registrations:**
   - Click on "Azure Active Directory" (or search for it)
   - If you see a tenant/organization selector, make sure you're in the correct one:
     - For personal accounts: You should see your personal account or "Default Directory"
     - If you see an organization you don't belong to, click your account icon and switch directories
   - Click on "App registrations" in the left menu
   - Click "+ New registration"

3. **Register the Application:**
   - **Name:** Enter a name like "Lea Assistant Email"
   - **Supported account types:** 
     - **IMPORTANT:** Select "Accounts in any organizational directory and personal Microsoft accounts"
     - This makes it multi-tenant and allows use of the `/common` endpoint
     - If you select "Accounts in this organizational directory only", you cannot use `OUTLOOK_TENANT_ID=common`
   - **Redirect URI:**
     - Platform: Select "Public client/native (mobile & desktop)"
     - Redirect URI: Enter `http://localhost`
   - Click "Register"

4. **Copy the Application (Client) ID:**
   - After registration, you'll see the "Overview" page
   - Copy the **Application (client) ID** - you'll need this for the .env file

5. **Configure API Permissions:**
   - Click on "API permissions" in the left menu
   - Click "+ Add a permission"
   - Select "Microsoft Graph"
   - Select "Delegated permissions"
   - Add these permissions:
     - `Mail.Read` - Read mail in all mailboxes
     - `Mail.Send` - Send mail as the user
   - Click "Add permissions"
   - **Important:** Click "Grant admin consent" if you see that option (for work accounts)

6. **Configure Authentication:**
   - Click on "Authentication" in the left menu
   - **CRITICAL:** Under "Supported account types", make sure it says:
     - "Accounts in any organizational directory and personal Microsoft accounts"
     - If it says something else, click "Edit" and change it
   - Under "Advanced settings", make sure "Allow public client flows" is set to **Yes**
   - Click "Save"
   
   **Note:** If your app was created after 10/15/2018 and is not multi-tenant, you'll get an error when using `OUTLOOK_TENANT_ID=common`. You MUST configure it as multi-tenant.

## Step 2: Configure Environment Variables

1. **Create or edit the `.env` file** in the LeaAssistant directory:
   ```
   OUTLOOK_CLIENT_ID=your_client_id_here
   ```

   Replace `your_client_id_here` with the Application (Client) ID you copied from Azure Portal.

2. **Tenant ID Configuration:**
   - **For personal Microsoft accounts (Outlook.com, Hotmail, etc.):**
     - Use `OUTLOOK_TENANT_ID=common` (or leave it out - it defaults to "common")
     - **Important:** You need a free Azure account to register apps with a personal account
     - Sign up at https://azure.microsoft.com/free/ if you don't have one
     - When registering the app, select "Accounts in any organizational directory and personal Microsoft accounts"
   
   - **For work/school accounts:**
     - You can use the specific tenant ID:
       ```
       OUTLOOK_TENANT_ID=your_tenant_id
       ```
     - Or use `common` to allow both work and personal accounts
     - You can find your Tenant ID in Azure AD > Overview

## Step 3: Test the Integration

1. **Start Lea Assistant**

2. **Try checking your email:**
   - Ask Lea: "Check my Outlook email" or "Check my inbox"
   - The first time, a browser window will open asking you to sign in
   - Sign in with your Microsoft/Outlook account
   - Grant the requested permissions
   - After signing in, you can close the browser window
   - Lea should now be able to check your email

## Troubleshooting

### "Email client not available"
- Make sure you've installed: `pip install msal requests python-dotenv`
- Check that `OUTLOOK_CLIENT_ID` is set in your `.env` file
- Verify the Client ID is correct (no extra spaces or quotes)

### "Authentication failed" or "Invalid client"
- Double-check your Client ID in the `.env` file
- Make sure "Allow public client flows" is set to Yes in Azure AD
- Verify the redirect URI is set to `http://localhost`

### "Access denied" or Permission errors
- Make sure you've added `Mail.Read` and `Mail.Send` permissions in Azure AD
- For work accounts, make sure admin consent has been granted
- Try removing and re-adding the permissions

### Browser doesn't open for login
- Check your internet connection
- Make sure no firewall is blocking the authentication
- Try running Lea Assistant as administrator (if on Windows)

### "Token expired" errors
- The token cache should be saved automatically
- If you keep getting this error, delete `outlook_token_cache.json` and try again

## Security Notes

- The Client ID is not a secret - it's safe to have in your `.env` file
- Your login credentials are never stored - only an access token
- The token is cached locally in `outlook_token_cache.json`
- You can revoke access at any time in your Microsoft account settings

## What You Can Do

Once set up, you can ask Lea to:
- "Check my email" or "Check my inbox"
- "Read email [email_id]"
- "Send an email to [address] with subject [subject]"
- "Mark email [email_id] as read"

## Need Help?

If you encounter issues:
1. Check the error message in Lea's response
2. Check the console/logs for detailed error messages
3. Verify all steps in this guide were completed correctly
4. Make sure your Microsoft account has access to Outlook/Exchange

