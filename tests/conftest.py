import pytest
from pathlib import Path

@pytest.fixture
def sample_txt_path():
    return Path("tests/data/sample.txt")

@pytest.fixture
def sample_docx_path():
    return Path("tests/data/sample.docx")
