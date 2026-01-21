"""
Q&A Prompts - Промпт-шаблоны для вопросно-ответной системы.

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

from typing import List, Dict, Any


def build_qa_prompt(
    query: str,
    context_documents: List[Dict[str, Any]],
    include_sources: bool = True,
) -> str:
    """
    Build Q&A prompt from query and context.
    
    Args:
        query: User question.
        context_documents: Retrieved documents with text and metadata.
        include_sources: Include source information.
        
    Returns:
        Formatted prompt for LLM.
    """
    # Format context
    context_parts = []
    for i, doc in enumerate(context_documents, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        source_info = ""
        if include_sources:
            doc_type = metadata.get("doc_type", "документ")
            doc_name = metadata.get("file_name", f"Документ {i}")
            source_info = f"\n[Источник: {doc_name}, тип: {doc_type}]"
        
        context_parts.append(
            f"--- Документ {i} ---\n{text}{source_info}\n"
        )
    
    context_text = "\n".join(context_parts)
    
    # Build full prompt
    prompt = f"""На основе предоставленных документов ответь на вопрос пользователя.

ДОКУМЕНТЫ:
{context_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ИНСТРУКЦИИ:
1. Используй ТОЛЬКО информацию из предоставленных документов
2. Цитируй источники для каждого утверждения в формате [Источник: название документа]
3. Если информации недостаточно, честно об этом скажи
4. Отвечай структурированно и по существу
5. Не выдумывай факты и не используй внешние знания

ОТВЕТ:"""
    
    return prompt


def build_followup_prompt(
    original_query: str,
    original_answer: str,
    followup_query: str,
    context_documents: List[Dict[str, Any]],
) -> str:
    """
    Build follow-up question prompt with conversation history.
    
    Args:
        original_query: Original user question.
        original_answer: Previous answer.
        followup_query: Follow-up question.
        context_documents: Retrieved documents.
        
    Returns:
        Formatted prompt.
    """
    context_parts = []
    for i, doc in enumerate(context_documents, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        doc_name = metadata.get("file_name", f"Документ {i}")
        
        context_parts.append(f"--- Документ {i} ({doc_name}) ---\n{text}\n")
    
    context_text = "\n".join(context_parts)
    
    prompt = f"""Ты продолжаешь диалог с пользователем о юридических документах.

КОНТЕКСТ ДИАЛОГА:
Пользователь спросил: "{original_query}"
Ты ответил: "{original_answer}"

ДОКУМЕНТЫ:
{context_text}

УТОЧНЯЮЩИЙ ВОПРОС:
{followup_query}

ИНСТРУКЦИИ:
1. Учитывай контекст предыдущего диалога
2. Используй информацию из документов
3. Цитируй источники
4. Будь последовательным с предыдущим ответом

ОТВЕТ:"""
    
    return prompt


def build_summarization_prompt(
    documents: List[Dict[str, Any]],
    focus: str = "general",
) -> str:
    """
    Build prompt for document summarization.
    
    Args:
        documents: Documents to summarize.
        focus: Focus area (general, financial, legal, risks).
        
    Returns:
        Formatted prompt.
    """
    doc_texts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        doc_name = doc.get("metadata", {}).get("file_name", f"Документ {i}")
        doc_texts.append(f"--- {doc_name} ---\n{text}\n")
    
    combined_text = "\n".join(doc_texts)
    
    focus_instructions = {
        "general": "Предоставь общий обзор содержания документа",
        "financial": "Сосредоточься на финансовых аспектах: цены, суммы, условия оплаты",
        "legal": "Выдели ключевые юридические аспекты: права, обязанности, ответственность",
        "risks": "Проанализируй потенциальные риски и проблемные моменты",
    }
    
    instruction = focus_instructions.get(focus, focus_instructions["general"])
    
    prompt = f"""Проанализируй и кратко изложи содержание следующих документов.

ДОКУМЕНТЫ:
{combined_text}

ЗАДАЧА: {instruction}

СТРУКТУРА ОТВЕТА:
1. Краткое резюме (2-3 предложения)
2. Ключевые моменты (список)
3. Важные детали (если применимо)

Будь кратким, но информативным. Используй маркированные списки где уместно."""
    
    return prompt


def build_extraction_prompt(
    document_text: str,
    fields_to_extract: List[str],
) -> str:
    """
    Build prompt for structured information extraction.
    
    Args:
        document_text: Document text.
        fields_to_extract: List of fields to extract (e.g., ["parties", "dates", "amounts"]).
        
    Returns:
        Formatted prompt.
    """
    fields_description = {
        "parties": "Стороны договора (полные наименования, ИНН, адреса)",
        "dates": "Все даты (подписания, действия, сроки)",
        "amounts": "Суммы (цены, штрафы, обеспечение)",
        "obligations": "Обязанности сторон",
        "terms": "Сроки исполнения обязательств",
        "termination": "Условия расторжения",
        "liability": "Ответственность сторон",
    }
    
    fields_list = "\n".join([
        f"- {field}: {fields_description.get(field, field)}"
        for field in fields_to_extract
    ])
    
    prompt = f"""Извлеки следующую структурированную информацию из документа:

ДОКУМЕНТ:
{document_text}

ПОЛЯ ДЛЯ ИЗВЛЕЧЕНИЯ:
{fields_list}

ФОРМАТ ОТВЕТА:
Для каждого поля:
**Название поля:**
- Найденная информация
- [Источник: пункт X.Y]

Если информация не найдена, укажи "Не указано в документе"."""
    
    return prompt


def build_comparison_prompt(
    document1: Dict[str, Any],
    document2: Dict[str, Any],
    comparison_aspects: List[str],
) -> str:
    """
    Build prompt for document comparison.
    
    Args:
        document1: First document with text and metadata.
        document2: Second document with text and metadata.
        comparison_aspects: Aspects to compare.
        
    Returns:
        Formatted prompt.
    """
    doc1_name = document1.get("metadata", {}).get("file_name", "Документ 1")
    doc2_name = document2.get("metadata", {}).get("file_name", "Документ 2")
    
    doc1_text = document1.get("text", "")
    doc2_text = document2.get("text", "")
    
    aspects_list = "\n".join([f"- {aspect}" for aspect in comparison_aspects])
    
    prompt = f"""Сравни два документа по указанным аспектам.

ДОКУМЕНТ 1 ({doc1_name}):
{doc1_text}

ДОКУМЕНТ 2 ({doc2_name}):
{doc2_text}

АСПЕКТЫ СРАВНЕНИЯ:
{aspects_list}

ФОРМАТ ОТВЕТА:
Для каждого аспекта:

**Аспект:**
- Документ 1: [краткое описание]
- Документ 2: [краткое описание]
- Различия: [что отличается]
- Значимость: [почему это важно]

Завершай кратким резюме основных различий."""
    
    return prompt


# Template registry
QA_TEMPLATES = {
    "qa": build_qa_prompt,
    "followup": build_followup_prompt,
    "summarization": build_summarization_prompt,
    "extraction": build_extraction_prompt,
    "comparison": build_comparison_prompt,
}


def get_qa_template(template_name: str):
    """
    Get Q&A template function.
    
    Args:
        template_name: Template name.
        
    Returns:
        Template function.
        
    Raises:
        ValueError: If template is unknown.
    """
    if template_name not in QA_TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available: {list(QA_TEMPLATES.keys())}"
        )
    
    return QA_TEMPLATES[template_name]
