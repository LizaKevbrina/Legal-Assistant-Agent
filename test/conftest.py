"""
Pytest configuration and fixtures for all tests.
Provides common fixtures, markers, and test configuration.
"""

import os
import sys
from pathlib import Path
from typing import Generator
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    # Set test environment
    os.environ["ENVIRONMENT"] = "development"
    os.environ["DEBUG"] = "true"
    
    # Mock API keys for tests
    os.environ["API__SECRET_KEY"] = "test-secret-key-must-be-at-least-32-chars-long-for-tests"
    os.environ["LLM__OPENAI_API_KEY"] = "sk-test-openai-key-for-testing"
    os.environ["LLAMAPARSE__LLAMAPARSE_API_KEY"] = "llx-test-llamaparse-key"
    os.environ["COHERE__COHERE_API_KEY"] = "test-cohere-key"
    
    # Disable external services for tests
    os.environ["MONITORING__ENABLE_LANGSMITH"] = "false"
    os.environ["MONITORING__ENABLE_PROMETHEUS"] = "false"
    os.environ["SECURITY__ENABLE_PII_DETECTION"] = "false"
    
    # Use test URLs
    os.environ["QDRANT__QDRANT_URL"] = "http://localhost:6333"
    os.environ["REDIS__REDIS_URL"] = "redis://localhost:6379/15"  # Test DB
    
    yield
    
    # Cleanup (optional)


@pytest.fixture
def test_data_dir() -> Path:
    """Get test data directory"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_pdf_path(test_data_dir) -> Path:
    """Get path to sample PDF file"""
    pdf_path = test_data_dir / "sample_contract.pdf"
    
    # Create dummy PDF if not exists
    if not pdf_path.exists():
        # Note: In real tests, you'd have actual test files
        pdf_path.write_text("Dummy PDF content")
    
    return pdf_path


@pytest.fixture
def sample_text() -> str:
    """Sample legal text for testing"""
    return """
    ДОГОВОР АРЕНДЫ НЕЖИЛОГО ПОМЕЩЕНИЯ
    
    г. Москва, 15 января 2025 г.
    
    ООО "Арендодатель", именуемое в дальнейшем "Арендодатель", 
    в лице Генерального директора Иванова И.И., действующего на 
    основании Устава, с одной стороны, и ООО "Арендатор", 
    именуемое в дальнейшем "Арендатор", в лице Генерального 
    директора Петрова П.П., действующего на основании Устава, 
    с другой стороны, заключили настоящий Договор о нижеследующем:
    
    1. ПРЕДМЕТ ДОГОВОРА
    1.1. Арендодатель обязуется предоставить Арендатору за плату 
    во временное владение и пользование нежилое помещение, 
    расположенное по адресу: г. Москва, ул. Примерная, д. 1, 
    общей площадью 100 кв.м.
    """


@pytest.fixture
def sample_query() -> str:
    """Sample legal query for testing"""
    return "Найти все договоры аренды, заключенные в 2025 году в Москве"


@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    """Mock embedding vectors"""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 614,  # 3072 dimensions (simplified)
    ]


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    from legal_assistant.core.config import get_settings
    
    yield
    
    # Clear settings cache
    get_settings.cache_clear()


@pytest.fixture
def caplog_json(caplog):
    """
    Fixture to capture JSON logs.
    Parses JSON log messages for easier assertion.
    """
    import json
    
    class JSONCapLog:
        def __init__(self, caplog):
            self.caplog = caplog
        
        @property
        def records(self):
            """Parse JSON records"""
            records = []
            for record in self.caplog.records:
                try:
                    # Try to parse as JSON
                    msg = json.loads(record.message)
                    records.append(msg)
                except (json.JSONDecodeError, AttributeError):
                    # Not JSON, use raw message
                    records.append({"message": record.message})
            return records
        
        def contains(self, key: str, value: any = None) -> bool:
            """Check if any record contains key/value"""
            for record in self.records:
                if key in record:
                    if value is None or record[key] == value:
                        return True
            return False
    
    return JSONCapLog(caplog)


# Markers for test organization
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_external: marks tests requiring external services"
    )


# Skip tests requiring external services if not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    skip_external = pytest.mark.skip(reason="External service not available")
    
    for item in items:
        if "requires_external" in item.keywords:
            # Check if external services available
            # For now, skip all external tests in CI
            if os.getenv("CI"):
                item.add_marker(skip_external)


# Async test configuration
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests"""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# Mock external service calls
@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock OpenAI client"""
    from unittest.mock import AsyncMock, MagicMock
    
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Mock response",
                        role="assistant"
                    )
                )
            ],
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        )
    )
    
    return mock_client


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    """Mock Qdrant client"""
    from unittest.mock import MagicMock, AsyncMock
    
    mock_client = MagicMock()
    mock_client.search = AsyncMock(return_value=[])
    mock_client.upsert = AsyncMock(return_value=None)
    
    return mock_client


@pytest.fixture
def mock_redis_client(monkeypatch):
    """Mock Redis client"""
    from unittest.mock import MagicMock, AsyncMock
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock(return_value=True)
    mock_client.incr = AsyncMock(return_value=1)
    
    return mock_client


# Temporary directory for test files
@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Create temporary directory for test files"""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir(exist_ok=True)
    return test_dir


# Helper to create test files
@pytest.fixture
def create_test_file(temp_dir):
    """Factory fixture to create test files"""
    def _create_file(filename: str, content: str = "test content") -> Path:
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    
    return _create_file
