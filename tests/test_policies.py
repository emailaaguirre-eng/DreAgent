"""Test safety policies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from policies import validate_path, validate_command

WORKSPACE_ROOT = Path("/workspace")


def test_blocks_outside_workspace():
    is_safe, _ = validate_path(Path("/etc/passwd"), WORKSPACE_ROOT)
    assert not is_safe


def test_blocks_ssh_directory():
    is_safe, _ = validate_path(WORKSPACE_ROOT / ".ssh" / "id_rsa", WORKSPACE_ROOT)
    assert not is_safe


def test_blocks_aws_directory():
    is_safe, _ = validate_path(WORKSPACE_ROOT / ".aws" / "credentials", WORKSPACE_ROOT)
    assert not is_safe


def test_allows_safe_path():
    is_safe, _ = validate_path(WORKSPACE_ROOT / "src" / "main.py", WORKSPACE_ROOT)
    assert is_safe


def test_allows_whitelisted_command():
    is_allowed, _ = validate_command("pytest -q")
    assert is_allowed


def test_blocks_non_whitelisted_command():
    is_allowed, _ = validate_command("rm -rf /")
    assert not is_allowed
