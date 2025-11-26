# Lea Agentic Task System Guide

## Overview

Lea now has an **agentic task execution system** that allows her to autonomously perform pre-configured tasks. This system is designed with safety in mind - Lea can only perform tasks that you have explicitly enabled and trained her to do.

## Key Features

✅ **Safe & Controlled**: Only pre-configured tasks can be executed  
✅ **Task Management UI**: Enable/disable tasks through the interface  
✅ **Confirmation System**: Dangerous tasks require explicit confirmation  
✅ **Audit Trail**: All task executions are logged  
✅ **Extensible**: Easy to add custom tasks  

## Built-in Tasks

### File Operations

1. **file_copy** - Copy files
   - Params: `source`, `destination`
   - Example: "Copy config.json to backup folder"

2. **file_move** - Move files (requires confirmation)
   - Params: `source`, `destination`
   - Example: "Move old files to archive"

3. **file_delete** - Delete files (requires confirmation)
   - Params: `path`
   - Safety: Blocks deletion of critical system paths

4. **file_read** - Read file contents
   - Params: `path`
   - Example: "Read the log file"

5. **file_write** - Write content to file
   - Params: `path`, `content`
   - Example: "Create a new config file"

### Directory Operations

6. **directory_create** - Create directories
   - Params: `path`
   - Example: "Create a Projects folder"

7. **directory_list** - List directory contents
   - Params: `path`
   - Example: "Show me what's in the Downloads folder"

### Text Processing

8. **text_replace** - Find and replace text in files (requires confirmation)
   - Params: `path`, `old_text`, `new_text`
   - Example: "Replace all instances of 'old' with 'new' in config.json"

### System Commands

9. **system_command** - Execute system commands (requires confirmation & whitelist)
   - Params: `command`
   - Safety: Only whitelisted commands allowed by default
   - Example: "Run git status"

## How It Works

### 1. Natural Language Interaction

Just tell Lea what you want done in natural language:

```
You: "Copy all .txt files from C:\Temp to C:\Backup"
Lea: [TASK: file_copy] [PARAMS: source=C:\Temp\*.txt, destination=C:\Backup]

You: "Read the config.json file"
Lea: [TASK: file_read] [PARAMS: path=config.json]

You: "Create a folder called Projects"
Lea: [TASK: directory_create] [PARAMS: path=Projects]
```

### 2. Task Execution Flow

1. **Detection**: Lea detects when you want a task performed
2. **Parsing**: The system parses the task name and parameters
3. **Validation**: Parameters are validated for safety
4. **Confirmation**: Tasks requiring confirmation will prompt you
5. **Execution**: Task is executed safely
6. **Reporting**: Results are reported back to you
7. **Logging**: All tasks are logged to history

### 3. Task Management

Click the **🤖 Tasks** button in Lea's interface to:

- View all available tasks
- Enable/disable tasks
- View task execution history
- Manage task permissions

## Adding Custom Tasks

### Example: Create a Custom Task

```python
# In lea_tasks.py or a new file

from lea_tasks import BaseTask, TaskResult

class EmailSendTask(BaseTask):
    """Send an email"""
    
    def __init__(self):
        super().__init__(
            name="email_send",
            description="Send an email",
            requires_confirmation=True
        )
    
    def get_required_params(self):
        return ["to", "subject", "body"]
    
    def validate_params(self, **kwargs):
        to = kwargs.get("to")
        if not to or "@" not in to:
            return False, "Invalid email address"
        return True, ""
    
    def execute(self, **kwargs):
        try:
            to = kwargs.get("to")
            subject = kwargs.get("subject", "")
            body = kwargs.get("body", "")
            
            # Your email sending logic here
            # send_email(to, subject, body)
            
            return TaskResult(
                True,
                f"Email sent successfully to {to}",
                data={"to": to, "subject": subject}
            )
        except Exception as e:
            return TaskResult(False, f"Failed to send email: {str(e)}", error=str(e))

# Register the task
from lea_tasks import get_task_registry
registry = get_task_registry()
registry.register_task(EmailSendTask())
```

### Extending Tasks with External Libraries

#### Web Automation (Selenium/Playwright)

```python
from selenium import webdriver
from lea_tasks import BaseTask, TaskResult

class WebScrapeTask(BaseTask):
    """Scrape a webpage"""
    
    def __init__(self):
        super().__init__(
            name="web_scrape",
            description="Scrape content from a webpage",
            requires_confirmation=False
        )
    
    def execute(self, **kwargs):
        url = kwargs.get("url")
        try:
            driver = webdriver.Chrome()
            driver.get(url)
            content = driver.page_source
            driver.quit()
            
            return TaskResult(True, f"Scraped {url}", data={"content": content})
        except Exception as e:
            return TaskResult(False, f"Scraping failed: {str(e)}", error=str(e))
```

#### API Integration

```python
import requests
from lea_tasks import BaseTask, TaskResult

class APICallTask(BaseTask):
    """Make an API call"""
    
    def __init__(self):
        super().__init__(
            name="api_call",
            description="Make an HTTP API call",
            requires_confirmation=False
        )
    
    def execute(self, **kwargs):
        url = kwargs.get("url")
        method = kwargs.get("method", "GET")
        
        try:
            response = requests.request(method, url)
            return TaskResult(
                True,
                f"API call successful: {response.status_code}",
                data={"status_code": response.status_code, "content": response.text}
            )
        except Exception as e:
            return TaskResult(False, f"API call failed: {str(e)}", error=str(e))
```

#### Database Operations

```python
import sqlite3
from lea_tasks import BaseTask, TaskResult

class DatabaseQueryTask(BaseTask):
    """Execute a database query"""
    
    def __init__(self):
        super().__init__(
            name="db_query",
            description="Execute a database query",
            requires_confirmation=True  # Database operations should be confirmed
        )
    
    def execute(self, **kwargs):
        db_path = kwargs.get("db_path")
        query = kwargs.get("query")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            return TaskResult(True, f"Query executed: {len(results)} rows", data={"results": results})
        except Exception as e:
            return TaskResult(False, f"Query failed: {str(e)}", error=str(e))
```

## Safety Features

### 1. Task Whitelist
- Only registered tasks can be executed
- Unknown tasks are rejected with an error message

### 2. Confirmation System
- Tasks marked with `requires_confirmation=True` must be explicitly approved
- Dangerous operations (delete, move, system commands) require confirmation

### 3. Parameter Validation
- All parameters are validated before execution
- Invalid parameters result in clear error messages

### 4. Critical Path Protection
- System paths (Windows, /etc, /usr) are protected from modification
- Deletion of critical directories is blocked

### 5. Command Whitelist
- System commands use a whitelist approach
- Only safe commands are allowed by default
- Dangerous commands are blocked

### 6. Audit Trail
- All task executions are logged
- History includes timestamp, task name, parameters, and results
- View history in the Tasks dialog

## Best Practices

### 1. Start Small
- Enable only the tasks you need
- Test tasks with safe operations first

### 2. Use Confirmation
- Enable confirmation for potentially dangerous tasks
- Review task results before approving

### 3. Monitor History
- Regularly check task execution history
- Look for unexpected task executions

### 4. Custom Tasks
- Validate all inputs thoroughly
- Add safety checks for destructive operations
- Test custom tasks before enabling them

### 5. Error Handling
- All tasks should return clear error messages
- Don't expose sensitive information in error messages

## Configuration File

Tasks are configured in `lea_tasks_config.json`:

```json
{
  "tasks": {
    "file_copy": {"allowed": true},
    "file_delete": {"allowed": false},
    "system_command": {"allowed": false}
  },
  "history": [...]
}
```

Edit this file directly or use the Tasks dialog in Lea's interface.

## Troubleshooting

### Task Not Executing

1. **Check if task is enabled**: Use the Tasks dialog to verify
2. **Check parameters**: Ensure all required parameters are provided
3. **Check confirmation**: Some tasks require explicit confirmation
4. **Check logs**: Review `lea_crash.log` for error details

### Task Execution Failing

1. **Validate parameters**: Check that file paths exist, etc.
2. **Check permissions**: Ensure Lea has permission to perform the operation
3. **Review error messages**: Task results include detailed error information

### Adding Custom Tasks

1. Create task class inheriting from `BaseTask`
2. Implement `execute()` method
3. Implement `validate_params()` if needed
4. Register task with `registry.register_task(YourTask())`
5. Restart Lea to load the new task

## Examples of Monotonous Tasks Lea Can Now Do

✅ **File Management**:
- "Organize all PDFs into the Documents folder"
- "Copy all .txt files from Downloads to Archive"
- "Delete temporary files older than 30 days"

✅ **Content Processing**:
- "Replace all instances of 'old_domain.com' with 'new_domain.com' in config files"
- "Read all log files and summarize errors"
- "Create a backup of all config files"

✅ **Directory Operations**:
- "Create a Projects folder structure"
- "List all files in the Downloads folder"
- "Organize files by extension"

✅ **Automation**:
- "Run git status on all project folders"
- "Check system disk space"
- "Generate a report from log files"

## Future Enhancements

Potential additions:
- Web automation (Selenium/Playwright)
- Email automation
- API integration
- Database operations
- Scheduled tasks
- Task chaining (execute multiple tasks in sequence)
- Conditional task execution
- Task templates

## Support

For issues or questions:
1. Check `lea_crash.log` for errors
2. Review task execution history
3. Verify task configuration in `lea_tasks_config.json`
4. Test with simple tasks first

---

**Remember**: Lea can only do what you train her to do. The task system is designed to be safe, controlled, and extensible. Start with simple tasks and gradually add more complex automation as needed.

