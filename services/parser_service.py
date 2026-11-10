"""
services/parser_service.py
Multimodal Document & Image Parsing Service for Socratica Studio.

Integrates with `nvidia/nemotron-3.5-parse`:
- Renders `.pdf` pages into in-memory JPEG byte buffers using `pypdfium2` (or `PyMuPDF`/`fitz` fallback).
- Sends each rendered page (`image/jpeg` or `image/png`) as `multipart/form-data` to
  `https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse`.
- Concatenates `result["content"]` across pages.
- Directly extracts text from `.txt`, `.md`, and `.docx` files.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile

from models.schemas import ParseDocumentResponse

load_dotenv()
logger = logging.getLogger("socratica.parser_service")

NEMOTRON_PARSE_URL = (
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse"
)


class DocumentParserService:
    """Handles document ingestion, PDF page rasterization, and NVIDIA Nemotron-3.5-Parse calls."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.invoke_url = NEMOTRON_PARSE_URL

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("NVIDIA_API_KEY", "")
        return key.strip()

    # ------------------------------------------------------------------------
    # Core NVIDIA Nemotron-3.5-Parse Client Call
    # ------------------------------------------------------------------------

    def call_nemotron_parse_page(
        self,
        image_bytes: bytes,
        filename: str = "page_1.jpg",
        mime_type: str = "image/jpeg",
        max_output_tokens: int = 1024,
    ) -> str:
        """
        Send a single rendered page (`image/png` or `image/jpeg`) to
        `nvidia/nemotron-3.5-parse` via `multipart/form-data` and return `result["content"]`.
        """
        nvidia_api_key = self._get_api_key()
        if not nvidia_api_key or nvidia_api_key == "nvapi-your-nvidia-api-key-here":
            raise RuntimeError(
                "NVIDIA_API_KEY is not configured for nvidia/nemotron-3.5-parse."
            )

        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Accept": "application/json",
        }

        # One rendered page (image/png or image/jpeg) plus JSON options, sent as multipart/form-data
        files = {
            "document": (filename, image_bytes, mime_type),
            "options": (
                None,
                json.dumps({"decoding": {"max_output_tokens": max_output_tokens}}),
                "application/json",
            ),
        }

        response = requests.post(
            self.invoke_url,
            headers=headers,
            files=files,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, dict):
            if "content" in result and isinstance(result["content"], str):
                return result["content"].strip()
            # Handle potential nested response shapes gracefully
            if "choices" in result and result["choices"]:
                first = result["choices"][0]
                return (
                    first.get("message", {}).get("content", "")
                    or first.get("text", "")
                ).strip()

        raise ValueError(
            f"Unexpected payload structure from nemotron-3.5-parse: {result}"
        )

    # ------------------------------------------------------------------------
    # PDF Page Rendering (`pypdfium2` with `fitz` fallback)
    # ------------------------------------------------------------------------

    @staticmethod
    def render_pdf_to_jpeg_pages(
        pdf_bytes: bytes, scale: float = 2.0
    ) -> Tuple[List[bytes], List[str]]:
        """
        Convert each page of a PDF into in-memory JPEG bytes using `pypdfium2`
        (and also extract embedded text layer as a backup if offline).
        Returns `(jpeg_pages_bytes, fallback_page_texts)`.
        """
        jpeg_pages: List[bytes] = []
        fallback_texts: List[str] = []

        try:
            import pypdfium2 as pdfium  # type: ignore

            pdf_doc = pdfium.PdfDocument(pdf_bytes)
            page_Total = len(pdf_doc)

            for page_index in range(page_Total):
                page = pdf_doc[page_index]
                # Extract embedded text layer as fallback
                textpage = page.get_textpage()
                fallback_texts.append(textpage.get_text_range().strip())
                textpage.close()

                # Render page bitmap -> PIL Image -> in-memory JPEG bytes
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                buf = io.BytesIO()
                pil_image.save(buf, format="JPEG", quality=90)
                jpeg_pages.append(buf.getvalue())
                page.close()

            pdf_doc.close()
            return jpeg_pages, fallback_texts

        except ImportError:
            # Optional fallback to PyMuPDF (`fitz`) if installed instead of pypdfium2
            try:
                import fitz  # type: ignore

                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page in doc:
                    fallback_texts.append(page.get_text("text").strip())
                    pix = page.get_pixmap(dpi=150)
                    jpeg_pages.append(pix.tobytes("jpeg"))
                doc.close()
                return jpeg_pages, fallback_texts
            except ImportError as exc:
                raise RuntimeError(
                    "Neither `pypdfium2` nor `PyMuPDF (fitz)` is installed to render PDF pages."
                ) from exc

    # ------------------------------------------------------------------------
    # Direct Text & DOCX Extraction Helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def extract_docx_text(docx_bytes: bytes) -> Tuple[str, int]:
        """Extract paragraphs and table cells from a `.docx` file."""
        try:
            import docx  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "`python-docx` is required to parse .docx files."
            ) from exc

        document = docx.Document(io.BytesIO(docx_bytes))
        blocks: List[str] = []

        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                blocks.append(text)

        for table in document.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    blocks.append(" | ".join(row_cells))

        full_text = "\n\n".join(blocks)
        estimated_pages = max(1, math.ceil(len(full_text.split()) / 350))
        return full_text, estimated_pages

    @staticmethod
    def compute_reading_metrics(text: str) -> Tuple[int, int]:
        """Return `(word_count, estimated_reading_time_minutes)` at 200 WPM."""
        words = [w for w in text.split() if w.strip()]
        word_count = len(words)
        reading_time = max(1, math.ceil(word_count / 200)) if word_count > 0 else 1
        return word_count, reading_time

    # ------------------------------------------------------------------------
    # Unified UploadFile Entrypoint
    # ------------------------------------------------------------------------

    async def parse_uploaded_file(
        self, upload_file: UploadFile
    ) -> ParseDocumentResponse:
        """
        Parse an uploaded `.pdf`, `.png`, `.jpg`/`.jpeg`, `.txt`, `.md`, or `.docx` file
        and return a validated `ParseDocumentResponse`.
        """
        file_name = upload_file.filename or "uploaded_document"
        suffix = Path(file_name).suffix.lower()
        content_type = upload_file.content_type or self._infer_content_type(suffix)

        raw_bytes = await upload_file.read()
        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file '{file_name}' is empty.",
            )

        extracted_text = ""
        page_count = 1

        # Case 1: PDF Document -> Render each page to JPEG -> Call nvidia/nemotron-3.5-parse
        if suffix == ".pdf" or content_type == "application/pdf":
            content_type = "application/pdf"
            jpeg_pages, fallback_texts = self.render_pdf_to_jpeg_pages(raw_bytes)
            page_count = max(1, len(jpeg_pages))
            parsed_pages: List[str] = []

            for idx, page_jpeg in enumerate(jpeg_pages, start=1):
                try:
                    page_markdown = self.call_nemotron_parse_page(
                        image_bytes=page_jpeg,
                        filename=f"page_{idx}.jpg",
                        mime_type="image/jpeg",
                    )
                    parsed_pages.append(page_markdown)
                except Exception as exc:
                    logger.warning(
                        "nemotron-3.5-parse failed on page %d (%s); falling back to embedded PDF text.",
                        idx,
                        exc,
                    )
                    if idx - 1 < len(fallback_texts) and fallback_texts[idx - 1]:
                        parsed_pages.append(fallback_texts[idx - 1])
                    else:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Failed to parse PDF page {idx} via nvidia/nemotron-3.5-parse: {exc}",
                        ) from exc

            extracted_text = "\n\n---\n\n".join(p for p in parsed_pages if p)

        # Case 2: Single Image (.png, .jpg, .jpeg) -> Call nvidia/nemotron-3.5-parse directly
        elif suffix in {".png", ".jpg", ".jpeg"} or content_type in {
            "image/png",
            "image/jpeg",
        }:
            mime_type = "image/png" if suffix == ".png" or content_type == "image/png" else "image/jpeg"
            content_type = mime_type
            page_count = 1
            try:
                extracted_text = self.call_nemotron_parse_page(
                    image_bytes=raw_bytes,
                    filename=file_name,
                    mime_type=mime_type,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to parse image '{file_name}' via nvidia/nemotron-3.5-parse: {exc}",
                ) from exc

        # Case 3: Word Document (.docx)
        elif suffix == ".docx" or content_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }:
            content_type = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            extracted_text, page_count = self.extract_docx_text(raw_bytes)

        # Case 4: Plain Text or Markdown (.txt, .md)
        elif suffix in {".txt", ".md", ".markdown"} or content_type.startswith("text/"):
            content_type = "text/markdown" if suffix in {".md", ".markdown"} else "text/plain"
            extracted_text = raw_bytes.decode("utf-8", errors="replace").strip()
            word_cnt = len(extracted_text.split())
            page_count = max(1, math.ceil(word_cnt / 350))

        else:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file format '{suffix or content_type}'. "
                    "Supported formats: .pdf, .png, .jpg, .jpeg, .txt, .md, .docx."
                ),
            )

        word_count, reading_time = self.compute_reading_metrics(extracted_text)

        return ParseDocumentResponse(
            file_name=file_name,
            content_type=content_type,
            page_count=page_count,
            word_count=word_count,
            estimated_reading_time_minutes=reading_time,
            raw_text_extracted=extracted_text,
        )

    @staticmethod
    def _infer_content_type(suffix: str) -> str:
        mapping = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return mapping.get(suffix, "application/octet-stream")


parser_service = DocumentParserService()
