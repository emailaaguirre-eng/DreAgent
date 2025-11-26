# Lea Update System

This system automatically checks for outdated packages and script updates to keep Lea running smoothly for years to come.

## Files Created

1. **`requirements.txt`** - Lists all Python package dependencies
2. **`lea_update_checker.py`** - Automatic update checking module
3. **Update button** - Added to Lea's toolbar (🔄 Check Updates)

## How It Works

### Automatic Checks
- **On Startup**: Lea automatically checks for updates when you launch it (every 7 days by default)
- **Check Interval**: Updates are checked every 7 days (configurable in `lea_update_checker.py`)
- **Non-Blocking**: Checks run in background and won't slow down startup

### Manual Checks
- Click the **🔄 Check Updates** button in Lea's toolbar anytime
- View detailed report of outdated packages and script updates

### What Gets Checked

1. **Python Packages**
   - Checks all packages listed in `requirements.txt`
   - Compares installed versions with latest available versions
   - Shows which packages need updating

2. **Lea Script Version** (Optional)
   - If you host Lea on GitHub, it can check for new releases
   - Configure in `lea_update_checker.py`:
     ```python
     GITHUB_REPO_OWNER = "yourusername"
     GITHUB_REPO_NAME = "LeaAssistant"
     ```

## Updating Packages

### Automatic Update (Recommended)
1. Click **🔄 Check Updates** button
2. Review the update report
3. Click **"Update Packages"** button
4. Confirm the update
5. Restart Lea if needed

### Manual Update (Command Line)
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update a specific package
pip install --upgrade package_name
```

## Configuration

### Change Check Interval
Edit `lea_update_checker.py`:
```python
checker = UpdateChecker(check_interval_days=7)  # Change 7 to your preferred days
```

### Disable Automatic Checks
Edit `Lea_Visual_Code_v2.5.1a_ TTS.py`:
- Comment out or remove the `self._check_updates_on_startup()` call

### Enable GitHub Version Checking
Edit `lea_update_checker.py`:
```python
GITHUB_REPO_OWNER = "yourusername"  # Your GitHub username
GITHUB_REPO_NAME = "LeaAssistant"   # Your repository name
```

## Files Generated

- **`last_update_check.json`** - Tracks when last check was performed
- **`update_check.log`** - Logs all update check activities

## Troubleshooting

### Update Checker Not Working
- Make sure `lea_update_checker.py` is in the same folder as Lea
- Check `update_check.log` for error messages
- Verify you have internet connection

### Package Update Fails
- Make sure you have admin/root permissions if needed
- Try updating packages one at a time
- Check pip is up to date: `pip install --upgrade pip`

### GitHub Version Check Not Working
- Make sure `GITHUB_REPO_OWNER` and `GITHUB_REPO_NAME` are set correctly
- Verify the repository exists and has releases
- Check your internet connection

## Best Practices

1. **Regular Updates**: Check for updates monthly or when you notice issues
2. **Backup First**: Before major updates, backup your Lea folder
3. **Test After Updates**: After updating packages, test Lea's features
4. **Version Control**: Keep track of Lea script versions if you modify it
5. **Documentation**: Update `requirements.txt` when adding new dependencies

## Long-Term Maintenance

To keep Lea running for years:

1. **Keep Dependencies Updated**: Run update checks regularly
2. **Monitor Logs**: Check `update_check.log` periodically
3. **Test Compatibility**: When Python updates, test Lea thoroughly
4. **Backup Configurations**: Save your `.env` and settings files
5. **Version Lea Script**: If you modify Lea, version your changes

## Support

If you encounter issues:
1. Check `update_check.log` for errors
2. Review the update report for details
3. Try manual package updates
4. Verify all dependencies are installed: `pip install -r requirements.txt`

