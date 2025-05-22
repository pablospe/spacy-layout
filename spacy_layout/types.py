from dataclasses import dataclass

from .model import BaseItem

# Use our internal model's BaseItem for document items
DoclingItem = BaseItem


@dataclass
class Attrs:
    """Custom atributes used to extend spaCy"""

    doc_layout: str
    doc_pages: str
    doc_tables: str
    doc_markdown: str
    span_layout: str
    span_data: str
    span_heading: str
    span_group: str


@dataclass
class PageLayout:
    page_no: int
    width: float
    height: float

    @classmethod
    def from_dict(cls, data: dict) -> "PageLayout":
        return cls(**data)


@dataclass
class DocLayout:
    """Document layout features added to Doc object"""

    pages: list[PageLayout]

    @classmethod
    def from_dict(cls, data: dict) -> "DocLayout":
        pages = [PageLayout.from_dict(page) for page in data.get("pages", [])]
        return cls(pages=pages)


@dataclass
class SpanLayout:
    """Text span layout features added to Span object"""

    x: float
    y: float
    width: float
    height: float
    page_no: int

    @classmethod
    def from_dict(cls, data: dict) -> "SpanLayout":
        return cls(**data)
