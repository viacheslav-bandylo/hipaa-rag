import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pdfplumber


@dataclass
class DocumentChunk:
    content: str
    filename: str
    part: Optional[str]  # e.g., "164"
    subpart: Optional[str]  # e.g., "Subpart C"
    section: Optional[str]  # e.g., "164.308"
    section_title: Optional[str]
    page_numbers: List[int]  # List because a chunk might span pages


class HIPAAParser:
    # Regex patterns refined for accuracy
    PART_PATTERN = re.compile(r"^PART\s+(\d+)", re.IGNORECASE | re.MULTILINE)
    SUBPART_PATTERN = re.compile(r"^SUBPART\s+([A-Z])", re.IGNORECASE | re.MULTILINE)
    # Catches "§ 164.308 Administrative safeguards."
    SECTION_PATTERN = re.compile(r"^§\s*(\d+\.\d+)\s+(.*)$", re.MULTILINE)

    # Header/Footer noise to ignore
    NOISE_PATTERNS = [
        re.compile(r"HIPAA Administrative Simplification Regulation Text", re.IGNORECASE),
        re.compile(r"March 2013", re.IGNORECASE),
        re.compile(r"^\d+$")  # Standalone page numbers
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Parses PDF as a continuous stream to handle cross-page sentences.
        """
        full_text_stream = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                # Pre-clean headers/footers BEFORE processing to avoid breaking sentences
                cleaned_lines = []
                for line in text.split('\n'):
                    if not self._is_noise(line):
                        cleaned_lines.append(line)

                # Store text with page tracking
                full_text_stream.append((page_num, "\n".join(cleaned_lines)))

        return self._chunk_continuous_stream(full_text_stream, pdf_path)

    def _is_noise(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        for pattern in self.NOISE_PATTERNS:
            if pattern.search(line):
                return True
        return False

    def _chunk_continuous_stream(self, text_stream: List[Tuple[int, str]], filename: str) -> List[DocumentChunk]:
        chunks = []

        # Context trackers
        current_part = "General"
        current_subpart = "General"
        current_section = None
        current_section_title = None

        # Buffers
        current_chunk_text = []
        current_chunk_len = 0
        current_chunk_pages = set()

        for page_num, text in text_stream:
            lines = text.split('\n')

            for line in lines:
                # 1. Update Context Hierarchy
                # Check for Part change
                part_match = self.PART_PATTERN.search(line)
                if part_match:
                    current_part = part_match.group(1)
                    # Reset lower hierarchy when top hierarchy changes
                    current_subpart = "General"
                    current_section = None
                    continue

                # Check for Subpart change
                subpart_match = self.SUBPART_PATTERN.search(line)
                if subpart_match:
                    current_subpart = f"Subpart {subpart_match.group(1)}"
                    continue

                # Check for Section change
                section_match = self.SECTION_PATTERN.search(line)
                if section_match:
                    # Save previous chunk if it exists (forcing a break at new sections is usually good practice)
                    if current_chunk_text:
                        chunks.append(self._finalize_chunk(
                            current_chunk_text, filename, current_part, current_subpart,
                            current_section, current_section_title, current_chunk_pages
                        ))
                        # Handle overlap logic here if needed, but strict section breaks are cleaner for legal
                        current_chunk_text = []
                        current_chunk_len = 0
                        current_chunk_pages = set()

                    current_section = section_match.group(1)
                    current_section_title = section_match.group(2).strip()

                # 2. Build Chunk
                # Skip TOC lines (simple heuristic: lines ending with lots of dots and a number)
                if re.search(r"\.{5,}\s*\d+$", line):
                    continue

                current_chunk_text.append(line)
                current_chunk_len += len(line)
                current_chunk_pages.add(page_num)

                # 3. Check Size Limits
                if current_chunk_len >= self.chunk_size:
                    chunks.append(self._finalize_chunk(
                        current_chunk_text, filename, current_part, current_subpart,
                        current_section, current_section_title, current_chunk_pages
                    ))

                    # Apply Overlap
                    # Keep the last N characters/lines for context in the next chunk
                    overlap_buffer = []
                    overlap_len = 0
                    for prev_line in reversed(current_chunk_text):
                        overlap_buffer.insert(0, prev_line)
                        overlap_len += len(prev_line)
                        if overlap_len >= self.chunk_overlap:
                            break

                    current_chunk_text = overlap_buffer
                    current_chunk_len = overlap_len
                    # Note: Page numbers for overlap might be imprecise, but acceptable

        # Final flush
        if current_chunk_text:
            chunks.append(self._finalize_chunk(
                current_chunk_text, filename, current_part, current_subpart,
                current_section, current_section_title, current_chunk_pages
            ))

        return chunks

    def _finalize_chunk(self, lines, filename, part, subpart, section, title, pages):
        content = "\n".join(lines).strip()
        # Enrich content with context string for better embedding retrieval
        # This prepends "Part 164 | Subpart C..." to the actual text content
        context_header = f"HIPAA Part {part} | {subpart} | Section {section} - {title}\n"
        full_content = context_header + content

        return DocumentChunk(
            content=full_content,
            filename=filename,
            part=part,
            subpart=subpart,
            section=section,
            section_title=title,
            page_numbers=sorted(list(pages))
        )