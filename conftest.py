"""Pytest configuration and shared fixtures"""
import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests with live pipeline (slow)"
    )
    parser.addoption(
        "--use-real-data",
        action="store_true",
        default=True,
        help="Use real enrichment data (default: True)"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return test data directory"""
    return project_root / "tests" / "fixtures" / "real_data"


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key
