import io
from typing import Optional

import logfire
from pypdf import PdfReader, PdfWriter
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from app.config import settings


MAX_PAGES_PER_REQUEST = 15

_client: Optional[documentai.DocumentProcessorServiceClient] = None


def get_client() -> documentai.DocumentProcessorServiceClient:
    """
    Builds the Document AI client lazily, pinned to the processor's region.

    The default endpoint only resolves processors in "us"; any other location
    (eu, asia, ...) needs a regional api_endpoint or every call 404s.
    """
    global _client
    if _client is None:
        _client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{settings.GCP_DOC_AI_LOCATION}-documentai.googleapis.com"
            )
        )
    return _client


def get_processor_name(client: documentai.DocumentProcessorServiceClient) -> str:
    """Builds the fully-qualified Document AI processor name."""
    if not settings.GCP_DOC_AI_PROCESSOR_ID:
        raise ValueError("GCP_DOC_AI_PROCESSOR_ID is not set")

    return client.processor_path(
        settings.PROJECT_ID,
        settings.GCP_DOC_AI_LOCATION,
        settings.GCP_DOC_AI_PROCESSOR_ID
    )


def parse_pdf(file_path: str) -> str:
    """
    Parse PDF using Google Cloud Document AI.
    Automatically splits large PDFs into 15-page chunks to bypass synchronous API limits.
    """

    with logfire.span("Document AI Parsing", filename=file_path):
        try:
            client = get_client()
            reader = PdfReader(file_path)

            # Many PDFs carry only an owner password and open with an empty one.
            if reader.is_encrypted:
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise ValueError(f"Could not decrypt encrypted PDF: {file_path}")

            total_pages = len(reader.pages)
            logfire.info(f"Total pages: {total_pages}")

            if total_pages == 0:
                logfire.warning(f"No pages found in {file_path}")
                return ""

            name = get_processor_name(client)

            parts = []

            # if small enough, process entirely
            if total_pages <= MAX_PAGES_PER_REQUEST:
                with open(file_path, "rb") as f:
                    image_content = f.read()
                parts.append(process_document_chunk(image_content, name, client))

            else:
                # Split into chunks of MAX_PAGES_PER_REQUEST
                logfire.info(f"PDF exceeds {MAX_PAGES_PER_REQUEST} pages. Splitting into chunks...")

                for i in range(0, total_pages, MAX_PAGES_PER_REQUEST):
                    writer = PdfWriter()
                    chunk_end = min(i + MAX_PAGES_PER_REQUEST, total_pages)

                    for page_num in range(i, chunk_end):
                        writer.add_page(reader.pages[page_num])

                    # Write chunk to bytes
                    bytes_stream = io.BytesIO()
                    writer.write(bytes_stream)
                    chunk_bytes = bytes_stream.getvalue()

                    with logfire.span("Processing pages", start=i + 1, end=chunk_end):
                        parts.append(process_document_chunk(chunk_bytes, name, client))

            full_text = "\n".join(part for part in parts if part)

            if not full_text.strip():
                logfire.warning(f"Document AI returned empty text for {file_path}")

            else:
                logfire.info(f"Document AI successfully parsed {len(full_text)} characters")

            return full_text

        except Exception as e:
            logfire.error(f"PDF parser failed for {file_path}: {e}")
            raise e


def process_document_chunk(
    image_content: bytes,
    name: str,
    client: Optional[documentai.DocumentProcessorServiceClient] = None,
) -> str:
    """Helper function to send a specific byte chunk to Document AI."""
    client = client or get_client()

    raw_document = documentai.RawDocument(
        content=image_content,
        mime_type="application/pdf",
    )

    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
    )

    result = client.process_document(request=request)
    if result.document is None:
        return ""

    return result.document.text
