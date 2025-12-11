"""Query analysis service for extracting citations and classifying queries."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""
    section_filters: list[str]  # e.g., ["164.502", "164.308"]
    part_filters: list[str]  # e.g., ["164", "160"]
    subpart_filter: Optional[str]  # e.g., "C" for Security Rule
    has_citation: bool


class QueryAnalyzer:
    """Analyzes queries to extract citation references and determine filtering strategy."""

    # Regex patterns for citation extraction
    SECTION_PATTERNS = [
        r"§\s*(\d{3}\.\d+)",  # §164.502 or § 164.502
        r"[Ss]ection\s+(\d{3}\.\d+)",  # Section 164.502
        r"(?<![.\d])(\d{3}\.\d{3})(?![.\d])",  # bare 164.502 (not part of longer number)
    ]

    # Part reference patterns
    PART_PATTERNS = [
        r"[Pp]art\s+(\d{3})",  # Part 164
    ]

    # Rule keyword mappings to parts/subparts
    RULE_MAPPINGS = {
        # Privacy Rule - Part 164 Subpart E
        "privacy rule": {"part": "164", "subpart": "E"},
        "privacy regulations": {"part": "164", "subpart": "E"},

        # Security Rule - Part 164 Subpart C
        "security rule": {"part": "164", "subpart": "C"},
        "security regulations": {"part": "164", "subpart": "C"},
        "security standards": {"part": "164", "subpart": "C"},

        # Breach Notification Rule - Part 164 Subpart D
        "breach notification": {"part": "164", "subpart": "D"},
        "breach rule": {"part": "164", "subpart": "D"},

        # Enforcement Rule - Part 160 Subpart D
        "enforcement rule": {"part": "160", "subpart": "D"},
        "enforcement regulations": {"part": "160", "subpart": "D"},

        # Transactions and Code Sets - Part 162
        "transactions": {"part": "162", "subpart": None},
        "code sets": {"part": "162", "subpart": None},
        "transaction standards": {"part": "162", "subpart": None},

        # General provisions - Part 160
        "general provisions": {"part": "160", "subpart": "A"},
        "administrative requirements": {"part": "160", "subpart": None},
    }

    def extract_citations(self, query: str) -> list[str]:
        """Extract section number citations from a query.

        Args:
            query: User query string

        Returns:
            List of section numbers (e.g., ["164.502", "164.308"])
        """
        citations = []

        for pattern in self.SECTION_PATTERNS:
            matches = re.findall(pattern, query)
            citations.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)

        return unique_citations

    def extract_part_references(self, query: str) -> list[str]:
        """Extract part number references from a query.

        Args:
            query: User query string

        Returns:
            List of part numbers (e.g., ["164", "160"])
        """
        parts = []

        for pattern in self.PART_PATTERNS:
            matches = re.findall(pattern, query)
            parts.extend(matches)

        # Deduplicate
        return list(dict.fromkeys(parts))

    def extract_rule_references(self, query: str) -> Optional[dict]:
        """Extract HIPAA rule references from query keywords.

        Args:
            query: User query string

        Returns:
            Dict with 'part' and 'subpart' keys if rule found, None otherwise
        """
        query_lower = query.lower()

        for keyword, mapping in self.RULE_MAPPINGS.items():
            if keyword in query_lower:
                return mapping

        return None

    def analyze(self, query: str) -> QueryAnalysis:
        """Perform full analysis of a query.

        Args:
            query: User query string

        Returns:
            QueryAnalysis with extracted filters and metadata
        """
        section_filters = self.extract_citations(query)
        part_filters = self.extract_part_references(query)
        subpart_filter = None

        # Check for rule keyword references
        rule_ref = self.extract_rule_references(query)
        if rule_ref:
            if rule_ref["part"] and rule_ref["part"] not in part_filters:
                part_filters.append(rule_ref["part"])
            if rule_ref["subpart"]:
                subpart_filter = rule_ref["subpart"]

        # If we have section filters, extract parts from them too
        for section in section_filters:
            part = section.split(".")[0]
            if part not in part_filters:
                part_filters.append(part)

        return QueryAnalysis(
            section_filters=section_filters,
            part_filters=part_filters,
            subpart_filter=subpart_filter,
            has_citation=len(section_filters) > 0
        )


# Singleton instance
query_analyzer = QueryAnalyzer()
