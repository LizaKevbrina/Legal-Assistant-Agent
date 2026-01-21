"""
E2E test fixtures and configuration.

Provides:
- Test data setup
- Service mocking
- Cleanup utilities
- Performance tracking
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

from legal_assistant.core import initialize_core, get_settings
from legal_assistant.ingestion import DocumentParser
from legal_assistant.retrieval import QdrantVectorStore, EmbeddingGenerator
from legal_assistant.generation import ResponseSynthesizer
from legal_assistant.security import PIIRedactor, InputValidator, OutputGuardrails
from legal_assistant.evaluation import (
    DatasetManager,
    MetricsTracker,
    RAGASEvaluator,
)


# ==================== SETUP ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_e2e_environment():
    """Setup E2E test environment."""
    # Initialize core
    initialize_core()
    
    # Create test directories
    test_dirs = [
        Path("./data/test_documents"),
        Path("./data/test_evaluation"),
        Path("./logs/test"),
    ]
    
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup (optional - comment out to keep data for debugging)
    # for dir_path in test_dirs:
    #     if dir_path.exists():
    #         shutil.rmtree(dir_path)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    temp_path = Path(tempfile.mkdtemp(prefix="legal_ai_test_"))
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


# ==================== TEST DATA ====================

@pytest.fixture
def sample_legal_text() -> str:
    """Sample legal text for testing."""
    return """
ДОГОВОР КУПЛИ-ПРОДАЖИ № 123/2024

г. Москва                                               01 января 2024 г.

ООО "Продавец", именуемое в дальнейшем "Продавец", в лице генерального 
директора Иванова Ивана Ивановича, действующего на основании Устава, 
с одной стороны, и ООО "Покупатель", именуемое в дальнейшем "Покупатель", 
в лице генерального директора Петрова Петра Петровича, действующего 
на основании Устава, с другой стороны, заключили настоящий договор 
о нижеследующем:

1. ПРЕДМЕТ ДОГОВОРА

1.1. Продавец обязуется передать в собственность Покупателя товар 
(далее - "Товар"), а Покупатель обязуется принять Товар и уплатить 
за него цену, установленную настоящим договором.

1.2. Наименование, количество, качество и цена Товара указаны 
в Спецификации (Приложение № 1), являющейся неотъемлемой частью 
настоящего договора.

2. ЦЕНА И ПОРЯДОК РАСЧЕТОВ

2.1. Общая стоимость Товара по настоящему договору составляет 
1 000 000 (один миллион) рублей 00 копеек, в том числе НДС 20%.

2.2. Оплата производится Покупателем путем перечисления денежных 
средств на расчетный счет Продавца в течение 10 (десяти) банковских 
дней с момента подписания настоящего договора.

3. ОТВЕТСТВЕННОСТЬ СТОРОН

3.1. За неисполнение или ненадлежащее исполнение обязательств 
по настоящему договору Стороны несут ответственность в соответствии 
с действующим законодательством Российской Федерации.

3.2. В случае просрочки оплаты Покупатель уплачивает Продавцу пеню 
в размере 0.1% от неуплаченной суммы за каждый день просрочки.

Настоящий договор вступает в силу с момента его подписания и действует 
до полного исполнения Сторонами своих обязательств.
"""


@pytest.fixture
def sample_pdf_document(temp_dir: Path, sample_legal_text: str) -> Path:
    """Create sample PDF document for testing."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    
    pdf_path = temp_dir / "sample_contract.pdf"
    
    # Create simple PDF
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    for line in sample_legal_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
    
    doc.build(story)
    
    return pdf_path


@pytest.fixture
def sample_queries() -> list[str]:
    """Sample queries for testing."""
    return [
        "Что такое договор купли-продажи?",
        "Какая цена товара по договору?",
        "Какая ответственность за просрочку оплаты?",
        "В течение какого срока нужно оплатить?",
        "Какой НДС включен в стоимость?",
    ]


@pytest.fixture
def sample_ground_truths() -> list[str]:
    """Sample ground truth answers for evaluation."""
    return [
        "Договор купли-продажи — это соглашение, по которому продавец "
        "обязуется передать товар в собственность покупателя, а покупатель "
        "обязуется принять товар и уплатить за него цену.",
        
        "Общая стоимость товара составляет 1 000 000 (один миллион) рублей, "
        "в том числе НДС 20%.",
        
        "За просрочку оплаты покупатель уплачивает продавцу пеню в размере "
        "0.1% от неуплаченной суммы за каждый день просрочки.",
        
        "Оплата производится в течение 10 (десяти) банковских дней с момента "
        "подписания договора.",
        
        "В стоимость включен НДС 20%.",
    ]


# ==================== COMPONENT FIXTURES ====================

@pytest.fixture
async def document_parser():
    """Document parser instance."""
    parser = DocumentParser()
    yield parser


@pytest.fixture
async def vector_store():
    """Vector store instance with test collection."""
    collection_name = f"test_legal_docs_{asyncio.get_event_loop().time()}"
    
    async with QdrantVectorStore() as store:
        # Create test collection
        await store.create_collection(collection_name, vector_size=3072)
        yield store
        
        # Cleanup
        try:
            await store.delete_collection(collection_name)
        except Exception:
            pass


@pytest.fixture
async def embedding_generator():
    """Embedding generator instance."""
    async with EmbeddingGenerator() as embedder:
        yield embedder


@pytest.fixture
async def response_synthesizer():
    """Response synthesizer instance."""
    async with ResponseSynthesizer() as synthesizer:
        yield synthesizer


@pytest.fixture
def pii_redactor():
    """PII redactor instance."""
    return PIIRedactor()


@pytest.fixture
def input_validator():
    """Input validator instance."""
    return InputValidator()


@pytest.fixture
def output_guardrails():
    """Output guardrails instance."""
    return OutputGuardrails()


@pytest.fixture
def dataset_manager(temp_dir: Path):
    """Dataset manager instance."""
    return DatasetManager(data_dir=temp_dir / "datasets")


@pytest.fixture
def metrics_tracker(temp_dir: Path):
    """Metrics tracker instance."""
    return MetricsTracker(storage_path=temp_dir / "metrics")


@pytest.fixture
async def ragas_evaluator():
    """RAGAS evaluator instance."""
    return RAGASEvaluator()


# ==================== HELPER FUNCTIONS ====================

@pytest.fixture
def assert_metric_in_range():
    """Helper to assert metric is in expected range."""
    def _assert(metric_value: float, min_val: float, max_val: float, metric_name: str):
        assert min_val <= metric_value <= max_val, (
            f"{metric_name} ({metric_value:.3f}) not in range [{min_val}, {max_val}]"
        )
    return _assert


@pytest.fixture
def measure_performance():
    """Helper to measure operation performance."""
    import time
    
    class PerformanceMeasure:
        def __init__(self):
            self.start_time = None
            self.duration = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, *args):
            self.duration = time.time() - self.start_time
        
        def assert_faster_than(self, max_seconds: float, operation: str):
            assert self.duration < max_seconds, (
                f"{operation} took {self.duration:.2f}s (max: {max_seconds}s)"
            )
    
    return PerformanceMeasure


# ==================== MARKERS ====================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (> 10s)"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API"
    )
