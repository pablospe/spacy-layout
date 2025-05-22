"""Pydantic models for document representation."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field


class CoordOrigin(str, Enum):
    """Coordinate system origin location."""

    TOPLEFT = "topleft"
    BOTTOMLEFT = "bottomleft"


class BoundingBox(BaseModel):
    """Represents a bounding box in document coordinates."""

    l: float  # left edge
    r: float  # right edge
    t: float  # top edge
    b: float  # bottom edge
    coord_origin: CoordOrigin = CoordOrigin.TOPLEFT

    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.r - self.l

    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.b - self.t


class Size(BaseModel):
    """Dimensions of a page or element."""

    width: float
    height: float


class ItemLabel(str, Enum):
    """Labels for document items."""

    TEXT = "text"
    SECTION_HEADER = "section_header"
    PAGE_HEADER = "page_header"
    TITLE = "title"
    TABLE = "table"
    LIST_ITEM = "list_item"
    DOCUMENT_INDEX = "document_index"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    CODE = "code"
    FIGURE = "figure"
    CHART = "chart"
    IMAGE = "image"
    LINK = "link"
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    FORM = "form"
    KEY_VALUE = "key_value"
    UNKNOWN = "unknown"


class ItemProvenance(BaseModel):
    """Provenance information for a document item."""

    bbox: Optional[BoundingBox] = None
    page_no: int = 1
    confidence: Optional[float] = None
    source: Optional[str] = None  # e.g., "azure", "docling"


class PageInfo(BaseModel):
    """Information about a document page."""

    page_no: int
    size: Optional[Size] = None
    image: Optional[bytes] = None
    rotation: int = 0  # Page rotation in degrees
    is_scanned: bool = False


class TableCellData(BaseModel):
    """Data for a single table cell."""

    row_idx: int
    col_idx: int
    text: Optional[str] = None
    rowspan: int = 1
    colspan: int = 1
    row_header: bool = False
    column_header: bool = False
    is_merged: bool = False


class TableData(BaseModel):
    """Represents tabular data in a document."""

    rows: int
    cols: int
    values: List[List[Optional[str]]]
    header_rows: int = 0
    header_cols: int = 0
    cells: List[TableCellData] = Field(default_factory=list)
    caption: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert table data to pandas DataFrame."""
        if self.rows == 0 or self.cols == 0 or not self.values:
            return pd.DataFrame()

        # Extract header if present
        headers = None
        data_start_row = 0

        if self.header_rows > 0 and len(self.values) > 0:
            headers = self.values[0]
            data_start_row = self.header_rows

        # Convert table data to DataFrame
        data = self.values[data_start_row:] if data_start_row < len(self.values) else []
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Apply headers if available and they match the number of columns
        if headers and len(headers) == len(df.columns):
            # Filter out None values in headers
            clean_headers = [
                str(h) if h is not None else f"Column {i+1}"
                for i, h in enumerate(headers)
            ]
            df.columns = clean_headers

        return df


class ChartData(BaseModel):
    """Base model for chart data."""

    chart_type: str
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    legend: Optional[List[str]] = None


class ImageData(BaseModel):
    """Image or figure data."""

    content_type: Optional[str] = None  # MIME type
    image_bytes: Optional[bytes] = None
    alt_text: Optional[str] = None
    description: Optional[str] = None


class FormulaData(BaseModel):
    """Mathematical formula data."""

    latex: Optional[str] = None
    mathml: Optional[str] = None
    text_repr: Optional[str] = None


class ListData(BaseModel):
    """List item data."""

    list_type: Literal["unordered", "ordered", "description"] = "unordered"
    level: int = 0  # Nesting level
    marker: Optional[str] = None  # The marker text (e.g., "•", "1.", "a.")


class CodeData(BaseModel):
    """Code block data."""

    language: Optional[str] = None
    line_numbers: bool = False


class BaseItem(BaseModel):
    """Base class for document items."""

    self_ref: str = Field(default_factory=lambda: f"item_{uuid4().hex[:8]}")
    label: ItemLabel
    prov: List[ItemProvenance] = Field(default_factory=list)
    parent_ref: Optional[str] = None
    children_refs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextItem(BaseItem):
    """Represents text content in a document."""

    text: str
    style: Dict[str, Any] = Field(default_factory=dict)  # Font, color, etc.


class SectionHeaderItem(TextItem):
    """Section header in document."""

    level: int = 1  # h1, h2, etc.


class TitleItem(TextItem):
    """Document title."""

    pass


class ListItem(TextItem):
    """List item in document."""

    list_data: ListData = Field(default_factory=ListData)


class FormulaItem(TextItem):
    """Mathematical formula."""

    formula_data: FormulaData = Field(default_factory=FormulaData)


class CodeItem(TextItem):
    """Code block."""

    code_data: CodeData = Field(default_factory=CodeData)


class TableItem(BaseItem):
    """Represents a table in a document."""

    table: TableData

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export table to pandas DataFrame."""
        return self.table.to_dataframe()


class FigureItem(BaseItem):
    """Represents a figure, image, or chart."""

    caption: Optional[str] = None
    image_data: Optional[ImageData] = None
    chart_data: Optional[ChartData] = None


class LinkItem(BaseItem):
    """Hyperlink or internal reference."""

    url: Optional[str] = None
    text: str
    target_ref: Optional[str] = None  # For internal document references


class FootnoteItem(TextItem):
    """Footnote or endnote."""

    marker: str  # Reference marker e.g., "1", "*"


class KeyValueItem(BaseItem):
    """Key-value pairs in a document."""

    key: str
    value: str


class FormItem(BaseItem):
    """Form field."""

    field_type: str  # e.g., "text", "checkbox", "radio"
    name: Optional[str] = None
    value: Optional[str] = None


class Document(BaseModel):
    """Complete document model."""

    # Document metadata
    name: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created: Optional[str] = None
    modified: Optional[str] = None
    source_type: Optional[str] = None  # e.g., "pdf", "docx"
    source_path: Optional[str] = None

    # Core content
    pages: Dict[int, PageInfo] = Field(default_factory=dict)

    # Items by type
    texts: List[TextItem] = Field(default_factory=list)
    headers: List[SectionHeaderItem] = Field(default_factory=list)
    tables: List[TableItem] = Field(default_factory=list)
    figures: List[FigureItem] = Field(default_factory=list)
    formulas: List[FormulaItem] = Field(default_factory=list)
    lists: List[ListItem] = Field(default_factory=list)
    links: List[LinkItem] = Field(default_factory=list)
    footnotes: List[FootnoteItem] = Field(default_factory=list)
    code_blocks: List[CodeItem] = Field(default_factory=list)
    key_values: List[KeyValueItem] = Field(default_factory=list)
    form_fields: List[FormItem] = Field(default_factory=list)

    # Raw document items for direct access
    items: Dict[str, BaseItem] = Field(default_factory=dict)

    def add_item(self, item: BaseItem) -> None:
        """Add an item to the document and appropriate type collection."""
        self.items[item.self_ref] = item

        # Add to type-specific collections
        if isinstance(item, TableItem):
            self.tables.append(item)
        elif isinstance(item, FigureItem):
            self.figures.append(item)
        elif isinstance(item, SectionHeaderItem):
            self.headers.append(item)
        elif isinstance(item, FormulaItem):
            self.formulas.append(item)
        elif isinstance(item, ListItem):
            self.lists.append(item)
        elif isinstance(item, LinkItem):
            self.links.append(item)
        elif isinstance(item, FootnoteItem):
            self.footnotes.append(item)
        elif isinstance(item, CodeItem):
            self.code_blocks.append(item)
        elif isinstance(item, KeyValueItem):
            self.key_values.append(item)
        elif isinstance(item, FormItem):
            self.form_fields.append(item)
        elif isinstance(item, TextItem):
            self.texts.append(item)

    def get_page_items(self, page_no: int) -> List[BaseItem]:
        """Get all items on a specific page."""
        page_items = []
        for item_id, item in self.items.items():
            for prov in item.prov:
                if prov.page_no == page_no:
                    page_items.append(item)
                    break
        return page_items

    def get_all_items(self) -> List[BaseItem]:
        """Get all document items in a single list.

        Returns items in a sensible reading order (by page and position)
        """
        all_items = []

        # First collect all items with provenance
        items_with_prov = []
        for item in self.items.values():
            if item.prov:
                items_with_prov.append(item)

        # Sort by page number, then y-coordinate, then x-coordinate
        # This approximates reading order
        def get_sort_key(item):
            if not item.prov:
                return (float("inf"), float("inf"), float("inf"))
            # Use first provenance entry for sorting
            prov = item.prov[0]
            if not prov.bbox:
                return (prov.page_no, float("inf"), float("inf"))
            return (prov.page_no, prov.bbox.t, prov.bbox.l)

        all_items = sorted(items_with_prov, key=get_sort_key)

        # Add any items without provenance at the end
        for item in self.items.values():
            if not item.prov and item not in all_items:
                all_items.append(item)

        return all_items

    def get_text_content(self) -> str:
        """Get the full text content of the document in reading order."""
        items = self.get_all_items()
        texts = []

        for item in items:
            if isinstance(item, TextItem):
                texts.append(item.text)
            elif isinstance(item, TableItem):
                # Add table caption if available
                if item.table.caption:
                    texts.append(item.table.caption)

        return "\n\n".join(texts)

    def to_markdown(self) -> str:
        """Convert document to markdown format."""
        md_parts = []

        # Add title if available
        if self.title:
            md_parts.append(f"# {self.title}\n")

        # Process items in reading order
        for item in self.get_all_items():
            if isinstance(item, SectionHeaderItem):
                # Add appropriate header level
                prefix = "#" * item.level
                md_parts.append(f"{prefix} {item.text}\n")

            elif isinstance(item, TextItem) and not isinstance(item, ListItem):
                md_parts.append(f"{item.text}\n")

            elif isinstance(item, ListItem):
                # Format list items with appropriate markers
                marker = "* " if item.list_data.list_type == "unordered" else "1. "
                indent = "  " * item.list_data.level
                md_parts.append(f"{indent}{marker}{item.text}")

            elif isinstance(item, TableItem):
                # Convert table to markdown
                df = item.export_to_dataframe()
                # Simple markdown table formatting
                if not df.empty:
                    md_parts.append(df.to_markdown())
                    md_parts.append("")  # Empty line after table

            elif isinstance(item, FormulaItem):
                # Format mathematical formula
                if item.formula_data.latex:
                    md_parts.append(f"$${item.formula_data.latex}$$\n")
                else:
                    md_parts.append(f"${item.text}$\n")

            elif isinstance(item, CodeItem):
                # Format code blocks
                lang = item.code_data.language or ""
                md_parts.append(f"```{lang}\n{item.text}\n```\n")

        return "\n".join(md_parts)
