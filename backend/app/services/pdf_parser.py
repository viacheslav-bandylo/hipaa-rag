import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    content: str
    filename: str
    part: Optional[str]  # e.g., "164"
    subpart: Optional[str]  # e.g., "Subpart C"
    section: Optional[str]  # e.g., "164.308"
    section_title: Optional[str]
    page_numbers: List[int]  # List because a chunk might span pages
    paragraph_reference: Optional[str] = None  # e.g., "(a)(1)(i)"
    validation_warnings: List[str] = field(default_factory=list)


class HIPAAParser:
    # Regex patterns refined for accuracy
    PART_PATTERN = re.compile(r"^\s*PART\s+(\d+)", re.IGNORECASE | re.MULTILINE)
    SUBPART_PATTERN = re.compile(r"^\s*SUBPART\s+([A-Z])", re.IGNORECASE | re.MULTILINE)
    # Catches "§ 164.308 Administrative safeguards." - handles leading whitespace and Unicode
    # The § symbol can appear as U+00A7 or other Unicode variants
    SECTION_PATTERN = re.compile(r"^\s*[§\u00A7]\s*(\d+\.\d+)\s+(.*)$", re.MULTILINE)
    # Fallback pattern for inline section references (e.g., "Section 164.502")
    SECTION_FALLBACK_PATTERN = re.compile(r"^\s*Section\s+(\d+\.\d+)\s*[-–—]?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    # Pattern to detect section headers that might be split or have subsection markers
    SECTION_WITH_SUBSECTION_PATTERN = re.compile(r"^\s*[§\u00A7]\s*(\d+\.\d+)(?:\([a-z]\))?\s+(.*)$", re.MULTILINE)
    # Pattern to detect section references within text (for validation)
    INLINE_SECTION_REF_PATTERN = re.compile(r'[§\u00A7]\s*(\d+\.\d+)')

    # Header/Footer noise to ignore
    NOISE_PATTERNS = [
        re.compile(r"HIPAA Administrative Simplification Regulation Text", re.IGNORECASE),
        re.compile(r"March 2013", re.IGNORECASE),
        re.compile(r"^\d+$")  # Standalone page numbers
    ]

    # List detection patterns for HIPAA legal document structure
    # Matches "(a)", "(b)", etc. - lowercase letter markers
    LIST_LETTER_PATTERN = re.compile(r"^\s*\(([a-z])\)")
    # Matches "(1)", "(2)", etc. - numeric markers
    LIST_NUMBER_PATTERN = re.compile(r"^\s*\((\d+)\)")
    # Matches "(i)", "(ii)", "(iii)", etc. - roman numeral markers
    LIST_ROMAN_PATTERN = re.compile(r"^\s*\((i{1,3}|iv|v|vi{0,3}|ix|x)\)", re.IGNORECASE)
    # Matches "(A)", "(B)", etc. - uppercase letter markers (sub-sub items)
    LIST_UPPER_PATTERN = re.compile(r"^\s*\(([A-Z])\)")

    # Paragraph boundary detection
    PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n")  # Double newline
    SENTENCE_END_PATTERN = re.compile(r"[.!?]\s+(?=[A-Z])")  # Period followed by capital

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        respect_list_boundaries: bool = True,
        max_list_chunk_size: int = 2000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_list_boundaries = respect_list_boundaries
        self.max_list_chunk_size = max_list_chunk_size

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

        chunks = self._chunk_continuous_stream(full_text_stream, pdf_path)

        # Post-process to fix any remaining section misattributions
        chunks = self._post_process_chunks(chunks)

        return chunks

    def _is_noise(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        for pattern in self.NOISE_PATTERNS:
            if pattern.search(line):
                return True
        return False

    def _detect_list_marker(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Detect if a line starts with a list marker.
        Returns (marker_type, marker_value) or None.
        marker_type: 'letter', 'number', 'roman', 'upper'
        """
        letter_match = self.LIST_LETTER_PATTERN.match(line)
        if letter_match:
            return ('letter', letter_match.group(1))

        number_match = self.LIST_NUMBER_PATTERN.match(line)
        if number_match:
            return ('number', number_match.group(1))

        roman_match = self.LIST_ROMAN_PATTERN.match(line)
        if roman_match:
            return ('roman', roman_match.group(1).lower())

        upper_match = self.LIST_UPPER_PATTERN.match(line)
        if upper_match:
            return ('upper', upper_match.group(1))

        return None

    def _is_list_continuation(self, marker_info: Optional[Tuple[str, str]], prev_marker_info: Optional[Tuple[str, str]]) -> bool:
        """Check if the current marker continues a list from the previous marker."""
        if not marker_info or not prev_marker_info:
            return False

        curr_type, curr_val = marker_info
        prev_type, prev_val = prev_marker_info

        # Same type means we're likely in the same list
        if curr_type == prev_type:
            return True

        # Nested lists: number inside letter, roman inside number, upper inside roman
        nesting_order = ['letter', 'number', 'roman', 'upper']
        if curr_type in nesting_order and prev_type in nesting_order:
            return True

        return False

    def _detect_section_header(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Detect if a line is a section header using multiple patterns.
        Returns (section_number, section_title) or None.

        This uses a cascade of patterns to handle various PDF extraction quirks:
        1. Standard § symbol with number
        2. "Section" word prefix
        3. § with subsection markers like §164.308(a)
        """
        # Normalize the line - handle various Unicode whitespace and § variants
        normalized_line = line.strip()

        # Also check for common OCR errors: § might become S, s, or other characters
        # But be careful not to match false positives

        # Try primary pattern (§ 164.502 Title)
        match = self.SECTION_PATTERN.search(line)
        if match:
            section_num = match.group(1)
            title = match.group(2).strip()
            # Validate section number format (should be like 160.xxx, 162.xxx, or 164.xxx)
            if section_num.startswith(('160.', '162.', '164.')):
                return (section_num, title)

        # Try pattern with subsection marker (§ 164.308(a) Title)
        match = self.SECTION_WITH_SUBSECTION_PATTERN.search(line)
        if match:
            section_num = match.group(1)
            title = match.group(2).strip()
            if section_num.startswith(('160.', '162.', '164.')):
                return (section_num, title)

        # Try fallback pattern (Section 164.502 - Title)
        match = self.SECTION_FALLBACK_PATTERN.search(line)
        if match:
            section_num = match.group(1)
            title = match.group(2).strip()
            if section_num.startswith(('160.', '162.', '164.')):
                return (section_num, title)

        return None

    def _is_paragraph_boundary(self, line: str, prev_line: Optional[str]) -> bool:
        """Detect if we're at a paragraph boundary."""
        if not prev_line:
            return False

        # Double newline indicates paragraph break
        if not line.strip() and not prev_line.strip():
            return True

        # Significant indentation change
        curr_indent = len(line) - len(line.lstrip())
        prev_indent = len(prev_line) - len(prev_line.lstrip())

        # Going from indented to non-indented often indicates new paragraph
        if prev_indent > 4 and curr_indent == 0 and line.strip():
            return True

        return False

    def _find_best_break_point(self, lines: List[str], target_idx: int) -> int:
        """
        Find the best break point near target_idx.
        Priority:
        1. Paragraph boundaries (but not within a list)
        2. Sentence boundaries
        3. Target index as fallback
        """
        search_range = min(10, len(lines) // 4)  # Search within ~10 lines of target

        best_break = target_idx
        best_score = 0

        for i in range(max(0, target_idx - search_range), min(len(lines), target_idx + search_range)):
            if i == 0:
                continue

            score = 0
            line = lines[i]
            prev_line = lines[i - 1] if i > 0 else ""

            # Check for paragraph boundary
            if self._is_paragraph_boundary(line, prev_line):
                score += 10

            # Check for sentence end in previous line
            if prev_line.rstrip().endswith(('.', '!', '?')):
                score += 5

            # Prefer breaking before a new list item at the top level
            marker = self._detect_list_marker(line)
            if marker and marker[0] == 'letter':
                score += 3

            # Penalize breaking right after a list marker (mid-item)
            prev_marker = self._detect_list_marker(prev_line)
            if prev_marker and not marker:
                score -= 5

            # Distance penalty - prefer breaks closer to target
            distance_penalty = abs(i - target_idx) * 0.5
            score -= distance_penalty

            if score > best_score:
                best_score = score
                best_break = i

        return best_break

    def _extract_paragraph_reference(self, lines: List[str]) -> Optional[str]:
        """
        Extract the paragraph reference from chunk content.
        Returns format like "(a)(1)(i)" based on the first markers found.
        """
        refs = []
        seen_types = set()

        for line in lines:
            marker = self._detect_list_marker(line)
            if marker:
                marker_type, marker_val = marker

                # Only add if we haven't seen this type yet (first occurrence)
                if marker_type not in seen_types:
                    if marker_type == 'letter':
                        refs.append(f"({marker_val})")
                    elif marker_type == 'number':
                        refs.append(f"({marker_val})")
                    elif marker_type == 'roman':
                        refs.append(f"({marker_val})")
                    elif marker_type == 'upper':
                        refs.append(f"({marker_val})")
                    seen_types.add(marker_type)

        return "".join(refs) if refs else None

    def _is_in_active_list(self, lines: List[str], current_idx: int) -> bool:
        """
        Determine if we're currently inside an active list that shouldn't be broken.
        Looks at recent context to determine list state.
        """
        if not self.respect_list_boundaries:
            return False

        # Look at recent lines for list markers
        lookback = min(20, current_idx)
        recent_lines = lines[max(0, current_idx - lookback):current_idx + 1]

        list_depth = 0
        last_marker = None

        for line in recent_lines:
            marker = self._detect_list_marker(line)
            if marker:
                last_marker = marker
                list_depth = max(1, list_depth)
            elif line.strip() == "":
                # Empty line might end a list
                list_depth = max(0, list_depth - 1)

        # If we recently saw a list marker and haven't seen a clear list end
        return list_depth > 0 and last_marker is not None

    def _chunk_continuous_stream(self, text_stream: List[Tuple[int, str]], filename: str) -> List[DocumentChunk]:
        chunks = []

        # Context trackers
        current_part = "General"
        current_subpart = "General"
        current_section = None
        current_section_title = None

        # Buffers
        current_chunk_lines: List[str] = []
        current_chunk_len = 0
        current_chunk_pages = set()

        # List tracking
        in_list = False
        list_start_idx = 0
        last_list_marker: Optional[Tuple[str, str]] = None

        # Collect all lines with page info first
        all_lines: List[Tuple[int, str]] = []
        for page_num, text in text_stream:
            for line in text.split('\n'):
                all_lines.append((page_num, line))

        i = 0
        while i < len(all_lines):
            page_num, line = all_lines[i]

            # 1. Update Context Hierarchy
            # Check for Part change
            part_match = self.PART_PATTERN.search(line)
            if part_match:
                current_part = part_match.group(1)
                current_subpart = "General"
                current_section = None
                i += 1
                continue

            # Check for Subpart change
            subpart_match = self.SUBPART_PATTERN.search(line)
            if subpart_match:
                current_subpart = f"Subpart {subpart_match.group(1)}"
                i += 1
                continue

            # Check for Section change - ALWAYS break here (highest priority)
            # Try multiple patterns to catch different section header formats
            section_match = self._detect_section_header(line)
            if section_match:
                new_section, new_title = section_match

                # Only treat as section break if this is a DIFFERENT section
                if new_section != current_section:
                    logger.debug(f"Section transition detected: {current_section} -> {new_section} on page {page_num}")

                    # Save previous chunk if it exists
                    if current_chunk_lines:
                        chunk = self._finalize_chunk(
                            current_chunk_lines, filename, current_part, current_subpart,
                            current_section, current_section_title, current_chunk_pages
                        )
                        chunks.append(chunk)
                        current_chunk_lines = []
                        current_chunk_len = 0
                        current_chunk_pages = set()
                        in_list = False
                        last_list_marker = None

                    current_section = new_section
                    current_section_title = new_title
                    logger.info(f"Now processing section § {current_section}: {current_section_title}")

            # Skip TOC lines
            if re.search(r"\.{5,}\s*\d+$", line):
                i += 1
                continue

            # 2. Track list state
            marker = self._detect_list_marker(line)
            if marker:
                if not in_list:
                    in_list = True
                    list_start_idx = len(current_chunk_lines)
                last_list_marker = marker
            elif line.strip() == "" and in_list:
                # Empty line might signal list end
                # Check if next non-empty line is a list item
                next_marker = None
                for j in range(i + 1, min(i + 5, len(all_lines))):
                    next_line = all_lines[j][1]
                    if next_line.strip():
                        next_marker = self._detect_list_marker(next_line)
                        break
                if not next_marker:
                    in_list = False
                    last_list_marker = None

            # 3. Build Chunk
            current_chunk_lines.append(line)
            current_chunk_len += len(line) + 1  # +1 for newline
            current_chunk_pages.add(page_num)

            # 4. Check Size Limits with smart breaking
            effective_limit = self.max_list_chunk_size if in_list else self.chunk_size

            if current_chunk_len >= effective_limit:
                # Don't break if we're in a list and haven't exceeded max_list_chunk_size
                if in_list and current_chunk_len < self.max_list_chunk_size:
                    i += 1
                    continue

                # Find best break point
                if len(current_chunk_lines) > 1:
                    target_break = len(current_chunk_lines) - 1
                    if not in_list:
                        # Use smart break finding only if not forced by max_list_chunk_size
                        target_break = self._find_best_break_point(
                            current_chunk_lines,
                            len(current_chunk_lines) * self.chunk_size // current_chunk_len
                        )

                    # Split at break point
                    chunk_lines = current_chunk_lines[:target_break]
                    remaining_lines = current_chunk_lines[target_break:]

                    if chunk_lines:
                        chunk = self._finalize_chunk(
                            chunk_lines, filename, current_part, current_subpart,
                            current_section, current_section_title, current_chunk_pages
                        )
                        chunks.append(chunk)

                    # Apply overlap for context continuity
                    overlap_lines = []
                    overlap_len = 0
                    for prev_line in reversed(chunk_lines):
                        if overlap_len >= self.chunk_overlap:
                            break
                        overlap_lines.insert(0, prev_line)
                        overlap_len += len(prev_line) + 1

                    current_chunk_lines = overlap_lines + remaining_lines
                    current_chunk_len = sum(len(l) + 1 for l in current_chunk_lines)
                    # Keep pages, they're approximate anyway

                    # Reset list state if we broke in the middle
                    if in_list:
                        # Check if remaining content still has list items
                        has_list_marker = any(
                            self._detect_list_marker(l) for l in remaining_lines
                        )
                        if not has_list_marker:
                            in_list = False
                            last_list_marker = None

            i += 1

        # Final flush
        if current_chunk_lines:
            chunk = self._finalize_chunk(
                current_chunk_lines, filename, current_part, current_subpart,
                current_section, current_section_title, current_chunk_pages
            )
            chunks.append(chunk)

        return chunks

    def _validate_chunk(self, lines: List[str], assigned_section: Optional[str] = None) -> List[str]:
        """
        Validate chunk for potential issues.
        Returns list of warning messages.
        """
        warnings = []
        content = "\n".join(lines)

        # Check if chunk starts mid-sentence (starts with lowercase, not a list marker)
        first_line = lines[0].strip() if lines else ""
        if first_line and first_line[0].islower() and not self._detect_list_marker(first_line):
            warnings.append("Chunk may start mid-sentence")

        # Check for incomplete lists - looks for list markers without completion
        list_markers = []
        for line in lines:
            marker = self._detect_list_marker(line)
            if marker:
                list_markers.append(marker)

        # Check for orphan list items (e.g., (a) without (b))
        if list_markers:
            letter_markers = [m[1] for m in list_markers if m[0] == 'letter']
            if letter_markers:
                # If we have (a) but not (b), and this isn't a single-item list
                if 'a' in letter_markers and 'b' not in letter_markers:
                    # Check if content suggests more items should follow
                    if content.rstrip().endswith(':') or content.rstrip().endswith(';'):
                        warnings.append("List may be truncated (found (a) but no (b))")

        # Check for dangling list introductions
        if content.rstrip().endswith(':') and not list_markers:
            warnings.append("Chunk ends with colon but no list follows")

        # Check for very short chunks that might be fragments
        if len(content) < 100 and lines:
            warnings.append("Very short chunk, may be fragment")

        # CRITICAL: Check for section mismatch - detect if chunk content references
        # a different section than what it's being assigned to
        if assigned_section:
            # Look for section headers within the chunk that don't match assigned section
            for line in lines:
                header_match = self._detect_section_header(line)
                if header_match:
                    detected_section, detected_title = header_match
                    if detected_section != assigned_section:
                        warnings.append(
                            f"SECTION_MISMATCH: Content contains header for §{detected_section} "
                            f"but chunk is assigned to §{assigned_section}"
                        )
                        logger.error(
                            f"Section mismatch detected! Chunk assigned to {assigned_section} "
                            f"but contains header for {detected_section}: '{detected_title}'"
                        )

            # Also check for inline section references that might indicate misattribution
            inline_refs = self.INLINE_SECTION_REF_PATTERN.findall(content)
            # Filter to unique section numbers that appear prominently (not just cross-references)
            prominent_sections = set()
            for ref in inline_refs:
                # Only flag if this section appears at start of a line (likely a header we missed)
                for line in lines:
                    if line.strip().startswith(f"§{ref}") or line.strip().startswith(f"§ {ref}"):
                        prominent_sections.add(ref)

            for ref_section in prominent_sections:
                if ref_section != assigned_section and ref_section.startswith(('160.', '162.', '164.')):
                    # This is a different HIPAA section mentioned prominently
                    warnings.append(
                        f"POSSIBLE_MISATTRIBUTION: Chunk references §{ref_section} prominently "
                        f"but is assigned to §{assigned_section}"
                    )

        return warnings

    def _finalize_chunk(self, lines, filename, part, subpart, section, title, pages):
        content = "\n".join(lines).strip()
        # Enrich content with context string for better embedding retrieval
        # This prepends "Part 164 | Subpart C..." to the actual text content
        context_header = f"HIPAA Part {part} | {subpart} | Section {section} - {title}\n"
        full_content = context_header + content

        # Extract paragraph reference
        paragraph_ref = self._extract_paragraph_reference(lines)

        # Validate chunk - pass section for mismatch detection
        warnings = self._validate_chunk(lines, assigned_section=section)
        if warnings:
            for warning in warnings:
                logger.warning(f"Chunk validation [{section}]: {warning}")

        return DocumentChunk(
            content=full_content,
            filename=filename,
            part=part,
            subpart=subpart,
            section=section,
            section_title=title,
            page_numbers=sorted(list(pages)),
            paragraph_reference=paragraph_ref,
            validation_warnings=warnings
        )

    def _post_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Post-process chunks to fix section misattributions.
        If a chunk contains a section header for a DIFFERENT section,
        split it at that boundary.
        """
        corrected_chunks = []

        for chunk in chunks:
            # Parse the content to find section boundaries
            # Skip the context header line we prepend
            lines = chunk.content.split('\n')
            if lines and lines[0].startswith('HIPAA Part'):
                content_lines = lines[1:]  # Skip context header
            else:
                content_lines = lines

            # Find any section headers in the content
            section_indices = []
            for i, line in enumerate(content_lines):
                header_match = self._detect_section_header(line)
                if header_match:
                    detected_section, detected_title = header_match
                    if detected_section != chunk.section:
                        section_indices.append((i, detected_section, detected_title))
                        logger.warning(
                            f"Post-process: Found §{detected_section} in chunk assigned to §{chunk.section}"
                        )

            # If no mismatched sections, keep chunk as-is
            if not section_indices:
                corrected_chunks.append(chunk)
                continue

            # Split the chunk at section boundaries
            logger.info(f"Splitting chunk §{chunk.section} due to embedded section headers")

            prev_idx = 0
            current_section = chunk.section
            current_title = chunk.section_title

            for idx, new_section, new_title in section_indices:
                # Create chunk for content before this section header
                if idx > prev_idx:
                    sub_lines = content_lines[prev_idx:idx]
                    sub_content = '\n'.join(sub_lines).strip()
                    if sub_content:
                        context_header = f"HIPAA Part {chunk.part} | {chunk.subpart} | Section {current_section} - {current_title}\n"
                        corrected_chunks.append(DocumentChunk(
                            content=context_header + sub_content,
                            filename=chunk.filename,
                            part=chunk.part,
                            subpart=chunk.subpart,
                            section=current_section,
                            section_title=current_title,
                            page_numbers=chunk.page_numbers,
                            paragraph_reference=self._extract_paragraph_reference(sub_lines),
                            validation_warnings=[]
                        ))

                # Update current section for next segment
                current_section = new_section
                current_title = new_title
                prev_idx = idx

            # Create final chunk for remaining content
            if prev_idx < len(content_lines):
                sub_lines = content_lines[prev_idx:]
                sub_content = '\n'.join(sub_lines).strip()
                if sub_content:
                    context_header = f"HIPAA Part {chunk.part} | {chunk.subpart} | Section {current_section} - {current_title}\n"
                    corrected_chunks.append(DocumentChunk(
                        content=context_header + sub_content,
                        filename=chunk.filename,
                        part=chunk.part,
                        subpart=chunk.subpart,
                        section=current_section,
                        section_title=current_title,
                        page_numbers=chunk.page_numbers,
                        paragraph_reference=self._extract_paragraph_reference(sub_lines),
                        validation_warnings=[]
                    ))

        logger.info(f"Post-processing: {len(chunks)} chunks -> {len(corrected_chunks)} chunks")
        return corrected_chunks