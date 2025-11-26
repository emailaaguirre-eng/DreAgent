"""
Lea Assistant Installer
Prompts user for agent name, user name, and personality description during installation
"""

import json
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                              QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class InstallerDialog(QDialog):
    """Installation dialog for configuring agent name, user name, and personality"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lea Assistant - Installation")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 11pt;
            }
            QLineEdit, QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                font-size: 10pt;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        self.agent_name = "Lea"
        self.user_name = "Dre"
        self.personality = ""
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Welcome to Lea Assistant!")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Please configure your assistant:")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 10pt;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(10)
        
        # Agent Name
        agent_label = QLabel("Agent Name:")
        agent_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(agent_label)
        
        self.agent_input = QLineEdit()
        self.agent_input.setPlaceholderText("e.g., Lea, Jack, Alex...")
        self.agent_input.setText("Lea")
        layout.addWidget(self.agent_input)
        
        # User Name
        user_label = QLabel("Your Name:")
        user_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(user_label)
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("e.g., Dre, Lu, Sarah...")
        self.user_input.setText("Dre")
        layout.addWidget(self.user_input)
        
        # Personality Description
        personality_label = QLabel("Personality Description:")
        personality_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(personality_label)
        
        help_text = QLabel("Describe how you want your assistant to behave (optional - defaults will be used if left empty)")
        help_text.setStyleSheet("color: #aaaaaa; font-size: 9pt; font-style: italic;")
        layout.addWidget(help_text)
        
        self.personality_input = QTextEdit()
        self.personality_input.setPlaceholderText(
            "Example:\n"
            "Warm and friendly, always enthusiastic about helping. "
            "Uses humor appropriately and keeps conversations engaging. "
            "Professional but approachable, like a trusted friend."
        )
        self.personality_input.setMaximumHeight(120)
        layout.addWidget(self.personality_input)
        
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #666;")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        install_btn = QPushButton("Install")
        install_btn.clicked.connect(self.accept_install)
        button_layout.addWidget(install_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def accept_install(self):
        """Validate and save configuration"""
        agent_name = self.agent_input.text().strip()
        user_name = self.user_input.text().strip()
        personality = self.personality_input.toPlainText().strip()
        
        if not agent_name:
            QMessageBox.warning(self, "Invalid Input", "Agent name cannot be empty.")
            return
        
        if not user_name:
            QMessageBox.warning(self, "Invalid Input", "Your name cannot be empty.")
            return
        
        self.agent_name = agent_name
        self.user_name = user_name
        self.personality = personality
        
        self.accept()


def run_installer():
    """Run the installer and return configuration"""
    app = QApplication(sys.argv)
    
    dialog = InstallerDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return {
            "agent_name": dialog.agent_name,
            "user_name": dialog.user_name,
            "personality": dialog.personality
        }
    return None


def save_config(config_data, config_path):
    """Save configuration to file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_config(config_path):
    """Load configuration from file"""
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
    
    # Return defaults
    return {
        "agent_name": "Lea",
        "user_name": "Dre",
        "personality": ""
    }


if __name__ == "__main__":
    # Get project directory
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "agent_config.json"
    
    # Check if already configured
    if config_path.exists():
        response = QMessageBox.question(
            None,
            "Already Configured",
            "Agent configuration already exists.\n\n"
            "Do you want to reconfigure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if response == QMessageBox.StandardButton.No:
            sys.exit(0)
    
    # Run installer
    config = run_installer()
    
    if config:
        if save_config(config, config_path):
            QMessageBox.information(
                None,
                "Installation Complete",
                f"Configuration saved successfully!\n\n"
                f"Agent Name: {config['agent_name']}\n"
                f"Your Name: {config['user_name']}\n\n"
                f"You can now run Lea Assistant."
            )
        else:
            QMessageBox.critical(
                None,
                "Installation Failed",
                "Failed to save configuration.\n\n"
                "Please check file permissions and try again."
            )
            sys.exit(1)
    else:
        sys.exit(0)

