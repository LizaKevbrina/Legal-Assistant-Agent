"""
Tests for configuration module.
Tests Pydantic Settings, validation, and environment loading.
"""

import os
from pathlib import Path
from typing import Generator
import pytest
from pydantic import ValidationError

from legal_assistant.core.config import (
    Settings,
    APISettings,
    LLMSettings,
    LlamaParseSettings,
    QdrantSettings,
    get_settings,
)


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before and after test"""
    # Store original env
    original_env = os.environ.copy()
    
    # Clear relevant env vars
    env_vars_to_clear = [
        k for k in os.environ.keys()
        if k.startswith(("API__", "LLM__", "QDRANT__", "REDIS__"))
    ]
    for var in env_vars_to_clear:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original env
    os.environ.clear()
    os.environ.update(original_env)
    
    # Clear settings cache
    get_settings.cache_clear()


@pytest.fixture
def valid_env_vars() -> dict[str, str]:
    """Valid environment variables for testing"""
    return {
        "ENVIRONMENT": "development",
        "DEBUG": "true",
        "API__SECRET_KEY": "test-secret-key-must-be-at-least-32-chars-long",
        "LLM__OPENAI_API_KEY": "sk-test-key",
        "LLAMAPARSE__LLAMAPARSE_API_KEY": "llx-test-key",
        "QDRANT__QDRANT_URL": "http://localhost:6333",
        "COHERE__COHERE_API_KEY": "test-cohere-key",
    }


class TestAPISettings:
    """Test APISettings configuration"""
    
    def test_default_values(self, clean_env):
        """Test default API settings"""
        settings = APISettings()
        
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.workers == 4
        assert settings.reload is False
        assert "http://localhost:3000" in settings.cors_origins
    
    def test_valid_port_range(self, clean_env):
        """Test port validation"""
        # Valid port
        settings = APISettings(
            port=8080,
            secret_key="a" * 32
        )
        assert settings.port == 8080
        
        # Invalid port (too low)
        with pytest.raises(ValidationError):
            APISettings(port=0, secret_key="a" * 32)
        
        # Invalid port (too high)
        with pytest.raises(ValidationError):
            APISettings(port=70000, secret_key="a" * 32)
    
    def test_secret_key_validation(self, clean_env):
        """Test secret key minimum length"""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            APISettings(secret_key="short")
        
        assert "at least 32 characters" in str(exc_info.value).lower()
        
        # Valid length
        settings = APISettings(secret_key="a" * 32)
        assert len(settings.secret_key.get_secret_value()) == 32
    
    def test_secret_key_masking(self, clean_env):
        """Test that secret key is masked in repr"""
        settings = APISettings(secret_key="a" * 32)
        
        # Secret should be masked
        assert "aaaa" not in repr(settings)
        assert "SecretStr" in repr(settings)
        
        # But can be accessed
        assert settings.secret_key.get_secret_value() == "a" * 32


class TestLLMSettings:
    """Test LLMSettings configuration"""
    
    def test_default_values(self, clean_env):
        """Test default LLM settings"""
        os.environ["LLM__OPENAI_API_KEY"] = "sk-test"
        settings = LLMSettings()
        
        assert settings.openai_model == "gpt-4-turbo-preview"
        assert settings.openai_temperature == 0.1
        assert settings.openai_max_tokens == 2048
        assert settings.embedding_model == "text-embedding-3-large"
        assert settings.embedding_dimensions == 3072
    
    def test_temperature_validation(self, clean_env):
        """Test temperature bounds"""
        os.environ["LLM__OPENAI_API_KEY"] = "sk-test"
        
        # Valid temperatures
        for temp in [0.0, 0.5, 1.0, 2.0]:
            settings = LLMSettings(openai_temperature=temp)
            assert settings.openai_temperature == temp
        
        # Invalid (too low)
        with pytest.raises(ValidationError):
            LLMSettings(
                openai_api_key="sk-test",
                openai_temperature=-0.1
            )
        
        # Invalid (too high)
        with pytest.raises(ValidationError):
            LLMSettings(
                openai_api_key="sk-test",
                openai_temperature=2.1
            )
    
    def test_max_tokens_validation(self, clean_env):
        """Test max tokens validation"""
        os.environ["LLM__OPENAI_API_KEY"] = "sk-test"
        
        # Valid
        settings = LLMSettings(openai_max_tokens=1000)
        assert settings.openai_max_tokens == 1000
        
        # Invalid (too low)
        with pytest.raises(ValidationError):
            LLMSettings(
                openai_api_key="sk-test",
                openai_max_tokens=0
            )
        
        # Invalid (too high)
        with pytest.raises(ValidationError):
            LLMSettings(
                openai_api_key="sk-test",
                openai_max_tokens=20000
            )
    
    def test_anthropic_optional(self, clean_env):
        """Test that Anthropic key is optional"""
        os.environ["LLM__OPENAI_API_KEY"] = "sk-test"
        settings = LLMSettings()
        
        assert settings.anthropic_api_key is None
        
        # Can be set
        settings = LLMSettings(
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-test"
        )
        assert settings.anthropic_api_key is not None


class TestLlamaParseSettings:
    """Test LlamaParseSettings configuration"""
    
    def test_default_values(self, clean_env):
        """Test default LlamaParse settings"""
        os.environ["LLAMAPARSE__LLAMAPARSE_API_KEY"] = "llx-test"
        settings = LlamaParseSettings()
        
        assert settings.llamaparse_timeout == 60
        assert settings.llamaparse_max_pages == 100
        assert settings.llamaparse_result_type == "markdown"
        assert settings.use_vision_fallback is True
        assert settings.use_ocr_fallback is True
    
    def test_timeout_validation(self, clean_env):
        """Test timeout bounds"""
        # Valid
        settings = LlamaParseSettings(
            llamaparse_api_key="llx-test",
            llamaparse_timeout=30
        )
        assert settings.llamaparse_timeout == 30
        
        # Invalid (too low)
        with pytest.raises(ValidationError):
            LlamaParseSettings(
                llamaparse_api_key="llx-test",
                llamaparse_timeout=5
            )
        
        # Invalid (too high)
        with pytest.raises(ValidationError):
            LlamaParseSettings(
                llamaparse_api_key="llx-test",
                llamaparse_timeout=400
            )
    
    def test_result_type_validation(self, clean_env):
        """Test result type literal validation"""
        # Valid
        for result_type in ["text", "markdown"]:
            settings = LlamaParseSettings(
                llamaparse_api_key="llx-test",
                llamaparse_result_type=result_type
            )
            assert settings.llamaparse_result_type == result_type
        
        # Invalid
        with pytest.raises(ValidationError):
            LlamaParseSettings(
                llamaparse_api_key="llx-test",
                llamaparse_result_type="json"
            )


class TestQdrantSettings:
    """Test QdrantSettings configuration"""
    
    def test_default_values(self, clean_env):
        """Test default Qdrant settings"""
        settings = QdrantSettings()
        
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.collection_name == "legal_documents"
        assert settings.image_collection_name == "legal_images"
        assert settings.vector_size == 3072
        assert settings.distance_metric == "cosine"
        assert settings.hnsw_m == 16
        assert settings.hnsw_ef_construct == 200
    
    def test_distance_metric_validation(self, clean_env):
        """Test distance metric literal"""
        # Valid
        for metric in ["cosine", "euclid", "dot"]:
            settings = QdrantSettings(distance_metric=metric)
            assert settings.distance_metric == metric
        
        # Invalid
        with pytest.raises(ValidationError):
            QdrantSettings(distance_metric="manhattan")
    
    def test_vector_size_validation(self, clean_env):
        """Test vector size bounds"""
        # Valid
        settings = QdrantSettings(vector_size=1536)
        assert settings.vector_size == 1536
        
        # Invalid (too small)
        with pytest.raises(ValidationError):
            QdrantSettings(vector_size=64)
    
    def test_hnsw_parameters(self, clean_env):
        """Test HNSW index parameters"""
        # Valid ranges
        settings = QdrantSettings(hnsw_m=32, hnsw_ef_construct=100)
        assert settings.hnsw_m == 32
        assert settings.hnsw_ef_construct == 100
        
        # Invalid m (too low)
        with pytest.raises(ValidationError):
            QdrantSettings(hnsw_m=2)
        
        # Invalid ef_construct (too low)
        with pytest.raises(ValidationError):
            QdrantSettings(hnsw_ef_construct=10)


class TestMainSettings:
    """Test main Settings class"""
    
    def test_nested_settings(self, clean_env, valid_env_vars):
        """Test that nested settings are properly initialized"""
        # Set environment
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        settings = Settings()
        
        # Check nested settings
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.llamaparse, LlamaParseSettings)
        assert isinstance(settings.qdrant, QdrantSettings)
    
    def test_environment_validation(self, clean_env, valid_env_vars):
        """Test environment literal validation"""
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        # Valid environments
        for env in ["development", "staging", "production"]:
            os.environ["ENVIRONMENT"] = env
            get_settings.cache_clear()
            settings = Settings()
            assert settings.environment == env
        
        # Invalid environment
        os.environ["ENVIRONMENT"] = "test"
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            Settings()
    
    def test_is_production_property(self, clean_env, valid_env_vars):
        """Test is_production helper property"""
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        # Development
        os.environ["ENVIRONMENT"] = "development"
        get_settings.cache_clear()
        settings = Settings()
        assert settings.is_production is False
        assert settings.is_development is True
        
        # Production
        os.environ["ENVIRONMENT"] = "production"
        get_settings.cache_clear()
        settings = Settings()
        assert settings.is_production is True
        assert settings.is_development is False
    
    def test_app_metadata(self, clean_env, valid_env_vars):
        """Test application metadata"""
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        settings = Settings()
        
        assert settings.app_name == "Legal Assistant"
        assert settings.app_version == "1.0.0"
        assert settings.debug is True  # From valid_env_vars
    
    def test_env_file_loading(self, clean_env, tmp_path):
        """Test loading from .env file"""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_content = """
ENVIRONMENT=staging
DEBUG=false
API__SECRET_KEY=test-secret-key-must-be-at-least-32-chars-long
API__PORT=9000
LLM__OPENAI_API_KEY=sk-test-from-file
LLAMAPARSE__LLAMAPARSE_API_KEY=llx-test-from-file
COHERE__COHERE_API_KEY=cohere-test-from-file
"""
        env_file.write_text(env_content)
        
        # Load settings with custom env file
        # Note: This requires modifying Settings to accept env_file parameter
        # For now, we test manual env var setting
        for line in env_content.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
        
        get_settings.cache_clear()
        settings = Settings()
        
        assert settings.environment == "staging"
        assert settings.debug is False
        assert settings.api.port == 9000


class TestGetSettings:
    """Test get_settings caching"""
    
    def test_singleton_pattern(self, clean_env, valid_env_vars):
        """Test that get_settings returns same instance"""
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Same instance
        assert settings1 is settings2
    
    def test_cache_clear(self, clean_env, valid_env_vars):
        """Test cache clearing"""
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        settings1 = get_settings()
        
        # Change environment
        os.environ["ENVIRONMENT"] = "production"
        
        # Without cache clear, same settings
        settings2 = get_settings()
        assert settings2.environment == "development"
        
        # With cache clear, new settings
        get_settings.cache_clear()
        settings3 = get_settings()
        assert settings3.environment == "production"
        
        # Different instances
        assert settings1 is not settings3


class TestSecuritySettings:
    """Test SecuritySettings"""
    
    def test_pii_entities_default(self, clean_env):
        """Test default PII entities"""
        from legal_assistant.core.config import SecuritySettings
        
        settings = SecuritySettings()
        
        expected_entities = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "LOCATION",
            "ORGANIZATION",
        ]
        
        for entity in expected_entities:
            assert entity in settings.pii_entities
    
    def test_file_size_validation(self, clean_env):
        """Test max file size validation"""
        from legal_assistant.core.config import SecuritySettings
        
        # Valid
        settings = SecuritySettings(max_file_size_mb=100)
        assert settings.max_file_size_mb == 100
        
        # Invalid (too small)
        with pytest.raises(ValidationError):
            SecuritySettings(max_file_size_mb=0)
        
        # Invalid (too large)
        with pytest.raises(ValidationError):
            SecuritySettings(max_file_size_mb=600)
    
    def test_allowed_file_types(self, clean_env):
        """Test allowed file types"""
        from legal_assistant.core.config import SecuritySettings
        
        settings = SecuritySettings()
        
        assert ".pdf" in settings.allowed_file_types
        assert ".png" in settings.allowed_file_types
        assert ".exe" not in settings.allowed_file_types


class TestIntegration:
    """Integration tests for config system"""
    
    def test_full_config_load(self, clean_env, valid_env_vars):
        """Test loading complete configuration"""
        for key, value in valid_env_vars.items():
            os.environ[key] = value
        
        get_settings.cache_clear()
        settings = get_settings()
        
        # Verify all nested settings loaded
        assert settings.api.secret_key is not None
        assert settings.llm.openai_api_key is not None
        assert settings.llamaparse.llamaparse_api_key is not None
        assert settings.qdrant.qdrant_url is not None
        assert settings.cohere.cohere_api_key is not None
        
        # Verify can access all properties
        assert settings.is_development is True
        assert settings.app_name == "Legal Assistant"
    
    def test_missing_required_fields(self, clean_env):
        """Test error when required fields missing"""
        # Missing API secret key
        os.environ["LLM__OPENAI_API_KEY"] = "sk-test"
        
        get_settings.cache_clear()
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Should mention missing secret_key
        assert "secret_key" in str(exc_info.value).lower()


# Run with: pytest tests/core/test_config.py -v
