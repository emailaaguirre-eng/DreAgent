# How to Train Your Lea Agent

## Overview

Training Lea involves two main parts:
1. **Adding Tasks** - Teaching Lea new capabilities she can perform
2. **Configuring Behavior** - Setting permissions and training through system prompts

---

## Part 1: Adding Custom Tasks

### Step 1: Create a Custom Task Class

Create a new Python file or add to `lea_tasks.py`. Here's the template:

```python
from lea_tasks import BaseTask, TaskResult
import logging

class YourCustomTask(BaseTask):
    """Description of what your task does"""
    
    def __init__(self):
        super().__init__(
            name="task_name",  # Unique name Lea will use
            description="What this task does",  # Helpful description
            requires_confirmation=False  # Set True for dangerous operations
        )
    
    def get_required_params(self):
        """Return list of required parameter names"""
        return ["param1", "param2"]
    
    def validate_params(self, **kwargs):
        """Validate inputs before execution"""
        param1 = kwargs.get("param1")
        if not param1:
            return False, "param1 is required"
        return True, ""
    
    def execute(self, **kwargs):
        """The actual task execution"""
        try:
            param1 = kwargs.get("param1")
            param2 = kwargs.get("param2")
            
            # Validate
            valid, msg = self.validate_params(**kwargs)
            if not valid:
                return TaskResult(False, msg, error=msg)
            
            # Your task logic here
            # ... do something ...
            
            # Return success
            return TaskResult(
                True,
                f"Successfully completed: {param1}",
                data={"result": "your data here"}
            )
        except Exception as e:
            logging.error(f"YourCustomTask error: {e}")
            return TaskResult(False, f"Task failed: {str(e)}", error=str(e))
```

### Step 2: Register Your Task

In `lea_tasks.py`, at the end of the file, add:

```python
def create_task_registry() -> TaskRegistry:
    """Create and populate task registry"""
    registry = TaskRegistry()
    
    # ... existing tasks ...
    
    # Add your custom task
    from your_custom_tasks import YourCustomTask
    registry.register_task(YourCustomTask())
    
    return registry
```

Or register it directly:

```python
from lea_tasks import get_task_registry
from your_custom_tasks import YourCustomTask

registry = get_task_registry()
registry.register_task(YourCustomTask())
```

### Step 3: Update System Prompts (Optional)

In `Lea Visual Code v1.1.py`, you can add your task to the CORE_RULES:

```python
**Available tasks:**
- file_copy: Copy files (source, destination)
- your_task_name: Description of your task (param1, param2)
```

---

## Part 2: Example Custom Tasks

### Example 1: Send Email Task

```python
# custom_tasks/email_task.py
from lea_tasks import BaseTask, TaskResult
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

class EmailSendTask(BaseTask):
    """Send an email via SMTP"""
    
    def __init__(self):
        super().__init__(
            name="email_send",
            description="Send an email to a recipient",
            requires_confirmation=True
        )
    
    def get_required_params(self):
        return ["to", "subject", "body"]
    
    def validate_params(self, **kwargs):
        to = kwargs.get("to")
        if not to or "@" not in to:
            return False, "Valid email address required"
        if not kwargs.get("subject"):
            return False, "Subject is required"
        return True, ""
    
    def execute(self, **kwargs):
        try:
            to = kwargs.get("to")
            subject = kwargs.get("subject")
            body = kwargs.get("body", "")
            smtp_server = kwargs.get("smtp_server", "smtp.gmail.com")
            smtp_port = kwargs.get("smtp_port", 587)
            username = kwargs.get("username")  # From .env
            password = kwargs.get("password")  # From .env
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return TaskResult(True, f"Email sent successfully to {to}", data={"to": to})
        except Exception as e:
            return TaskResult(False, f"Failed to send email: {str(e)}", error=str(e))
```

### Example 2: Web Scraping Task

```python
# custom_tasks/web_scrape_task.py
from lea_tasks import BaseTask, TaskResult
import requests
from bs4 import BeautifulSoup
import logging

class WebScrapeTask(BaseTask):
    """Scrape content from a webpage"""
    
    def __init__(self):
        super().__init__(
            name="web_scrape",
            description="Extract text content from a webpage",
            requires_confirmation=False
        )
    
    def get_required_params(self):
        return ["url"]
    
    def execute(self, **kwargs):
        try:
            url = kwargs.get("url")
            if not url.startswith("http"):
                url = "https://" + url
            
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return TaskResult(
                True,
                f"Scraped {url} successfully",
                data={"url": url, "content": text[:5000]}  # Limit size
            )
        except Exception as e:
            return TaskResult(False, f"Failed to scrape: {str(e)}", error=str(e))
```

### Example 3: Database Query Task

```python
# custom_tasks/database_task.py
from lea_tasks import BaseTask, TaskResult
import sqlite3
import logging

class DatabaseQueryTask(BaseTask):
    """Execute a database query"""
    
    def __init__(self):
        super().__init__(
            name="db_query",
            description="Execute a SELECT query on a SQLite database",
            requires_confirmation=True  # DB operations should be confirmed
        )
    
    def get_required_params(self):
        return ["db_path", "query"]
    
    def validate_params(self, **kwargs):
        query = kwargs.get("query", "").upper()
        # Only allow SELECT queries for safety
        if not query.strip().startswith("SELECT"):
            return False, "Only SELECT queries are allowed"
        return True, ""
    
    def execute(self, **kwargs):
        try:
            db_path = kwargs.get("db_path")
            query = kwargs.get("query")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
            conn.close()
            
            # Format results
            data = {
                "columns": columns,
                "rows": [list(row) for row in results],
                "count": len(results)
            }
            
            return TaskResult(
                True,
                f"Query executed successfully: {len(results)} rows returned",
                data=data
            )
        except Exception as e:
            return TaskResult(False, f"Query failed: {str(e)}", error=str(e))
```

---

## Part 3: Configuring Task Permissions

### Method 1: Using the UI

1. Open Lea
2. Click the **🤖 Tasks** button
3. Select a task from the table
4. Click **Enable Selected** or **Disable Selected**
5. Changes are saved automatically

### Method 2: Edit Configuration File

Edit `lea_tasks_config.json`:

```json
{
  "tasks": {
    "file_copy": {"allowed": true},
    "file_delete": {"allowed": false},
    "email_send": {"allowed": true},
    "system_command": {"allowed": false}
  }
}
```

---

## Part 4: Training Through Conversation

### Teaching Lea When to Use Tasks

Lea learns from your system prompts. In `Lea Visual Code v1.1.py`, the CORE_RULES section teaches her:

- **When** to execute tasks (explicit requests, monotonous tasks)
- **How** to format task commands ([TASK: name] [PARAMS: ...])
- **What** tasks are available

### Example Training Prompt

You can enhance the system prompts with examples:

```python
**Task Execution Examples:**
- User: "Copy my documents to backup folder"
  You: I'll copy those files for you! [TASK: file_copy] [PARAMS: source=documents, destination=backup]

- User: "Organize my downloads folder"
  You: Let me help organize that! [TASK: directory_list] [PARAMS: path=downloads]
  Then analyze the results and suggest organization strategy.
```

---

## Part 5: Testing Your Tasks

### Test a Task Directly

```python
from lea_tasks import get_task_registry

registry = get_task_registry()

# Test your task
result = registry.execute_task(
    "your_task_name",
    {"param1": "value1", "param2": "value2"},
    confirmed=True  # If task requires confirmation
)

print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Data: {result.data}")
```

### Test Through Lea

1. Enable your task in the Tasks dialog
2. Ask Lea to perform the task naturally:
   - "Send an email to john@example.com about the meeting"
   - "Scrape the content from example.com"
   - "Query my database for all customers"

---

## Part 6: Best Practices

### 1. Safety First
- Always validate inputs
- Require confirmation for dangerous operations
- Never allow destructive operations without explicit permission

### 2. Clear Error Messages
- Provide helpful error messages
- Log errors for debugging
- Suggest alternatives when tasks fail

### 3. Task Design
- Keep tasks focused (one task = one purpose)
- Make tasks reusable
- Document required parameters clearly

### 4. Testing
- Test tasks with valid inputs
- Test with invalid inputs
- Test edge cases
- Test error conditions

### 5. Documentation
- Describe what each task does
- List required parameters
- Provide examples

---

## Part 7: Advanced: Task Chaining

You can create tasks that chain other tasks:

```python
class OrganizeFilesTask(BaseTask):
    """Organize files by extension"""
    
    def execute(self, **kwargs):
        source_dir = kwargs.get("source_dir")
        
        # Step 1: List directory
        list_result = registry.execute_task(
            "directory_list",
            {"path": source_dir},
            confirmed=False
        )
        
        if not list_result.success:
            return TaskResult(False, "Failed to list directory")
        
        # Step 2: Organize files
        for item in list_result.data['items']:
            if item['type'] == 'file':
                ext = item['name'].split('.')[-1]
                dest_dir = f"{source_dir}/{ext}_files"
                
                # Create directory if needed
                registry.execute_task(
                    "directory_create",
                    {"path": dest_dir},
                    confirmed=False
                )
                
                # Move file
                registry.execute_task(
                    "file_move",
                    {"source": item['path'], "destination": f"{dest_dir}/{item['name']}"},
                    confirmed=True
                )
        
        return TaskResult(True, "Files organized successfully")
```

---

## Part 8: Quick Start Checklist

- [ ] Decide what task you want to add
- [ ] Create task class inheriting from `BaseTask`
- [ ] Implement `execute()` method
- [ ] Add validation in `validate_params()`
- [ ] Register task with `registry.register_task()`
- [ ] Test task directly
- [ ] Enable task in Tasks dialog
- [ ] Test through Lea conversation
- [ ] Update system prompts if needed
- [ ] Document your task

---

## Need Help?

- Check `lea_tasks.py` for built-in task examples
- Review `AGENTIC_TASKS_GUIDE.md` for system overview
- Check `lea_crash.log` if tasks fail
- Review task execution history in Tasks dialog

---

**Remember**: Lea can only do what you train her to do. Start simple, test thoroughly, and gradually add more complex tasks!

