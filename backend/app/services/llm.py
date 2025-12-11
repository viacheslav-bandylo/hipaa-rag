from anthropic import Anthropic

from app.config import settings
from app.database import Document


class LLMService:
    SYSTEM_PROMPT = """You are a HIPAA compliance assistant. Your role is to answer questions about HIPAA regulations accurately and helpfully.

IMPORTANT RULES:
1. Answer questions using ONLY the provided context from HIPAA documentation
2. For EVERY statement you make, cite the exact section reference (e.g., "According to §164.502(a)...")
3. If asked for the exact text of a regulation, quote it VERBATIM and provide the complete section reference
4. If the information is not in the provided context, clearly state: "This information is not available in the provided HIPAA documentation sections."
5. When listing related paragraphs or requirements, provide the complete section references for each
6. Be precise and avoid speculation - only state what is explicitly in the regulations

FORMAT:
- Use clear, professional language
- Include section references inline with your explanations
- When quoting, use quotation marks and cite the source immediately after"""

    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def generate_response(
        self,
        query: str,
        context_documents: list[Document]
    ) -> str:
        context = self._format_context(context_documents)

        user_message = f"""Based on the following HIPAA documentation excerpts, please answer the user's question.

CONTEXT FROM HIPAA DOCUMENTATION:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer with proper citations to the section references provided above."""

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        return response.content[0].text

    def _format_context(self, documents: list[Document]) -> str:
        formatted_sections = []

        for i, doc in enumerate(documents, 1):
            section = f"""
---
[Source {i}]
Section Reference: {doc.section_reference}
Context: {doc.parent_context}
Page: {doc.page_number}

Content:
{doc.content}
---"""
            formatted_sections.append(section)

        return "\n".join(formatted_sections)


llm_service = LLMService()
