import sys
import pytest
from pathlib import Path

# Add repo root to sys.path for app.py import in tests
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture
def sample_txt_path():
    return Path("tests/data/sample.txt")

@pytest.fixture
def sample_docx_path():
    return Path("tests/data/sample.docx")
