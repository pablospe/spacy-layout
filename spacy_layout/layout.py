import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)

import srsly
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from pandas import DataFrame
from spacy.tokens import Doc, Span, SpanGroup

from .types import Attrs, DocLayout, PageLayout, SpanLayout
from .util import decode_df, decode_obj, encode_df, encode_obj

# Import python-dotenv for environment variable management (optional)
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

PDF_POINTS_PER_INCH = 72

if TYPE_CHECKING:
    from pandas import DataFrame
    from spacy.language import Language

# Type variable for contexts piped with documents
_AnyContext = TypeVar("_AnyContext")

TABLE_PLACEHOLDER = "TABLE"
TABLE_ITEM_LABELS = ["table", "document_index"]

# Register msgpack encoders and decoders for custom types
srsly.msgpack_encoders.register("spacy-layout.dataclass", func=encode_obj)
srsly.msgpack_decoders.register("spacy-layout.dataclass", func=decode_obj)
srsly.msgpack_encoders.register("spacy-layout.dataframe", func=encode_df)
srsly.msgpack_decoders.register("spacy-layout.dataframe", func=decode_df)


class spaCyLayoutAzure:
    def __init__(
        self,
        nlp: "Language",
        separator: str | None = "\n\n",
        attrs: dict[str, str] = {},
        headings: list[str] = [
            "section_header",
            "page_header",
            "title",
        ],
        display_table: Callable[["DataFrame"], str] | str = TABLE_PLACEHOLDER,
        azure_endpoint: str = None,
        azure_key: str = None,
        dotenv_path: str = None,
    ) -> None:
        """Initialize the layout parser and backend adapter."""
        self.nlp = nlp
        self.sep = separator
        self.attrs = Attrs(
            doc_layout=attrs.get("doc_layout", "layout"),
            doc_pages=attrs.get("doc_pages", "pages"),
            doc_tables=attrs.get("doc_tables", "tables"),
            doc_markdown=attrs.get("doc_markdown", "markdown"),
            span_layout=attrs.get("span_layout", "layout"),
            span_heading=attrs.get("span_heading", "heading"),
            span_data=attrs.get("span_data", "data"),
            span_group=attrs.get("span_group", "layout"),
        )
        self.headings = headings
        self.display_table = display_table

        # Initialize Azure Document Intelligence client
        self._init_azure_client(azure_endpoint, azure_key, dotenv_path)

        # Set spaCy extension attributes for custom data
        Doc.set_extension(self.attrs.doc_layout, default=None, force=True)
        Doc.set_extension(self.attrs.doc_pages, getter=self.get_pages, force=True)
        Doc.set_extension(self.attrs.doc_tables, getter=self.get_tables, force=True)
        Doc.set_extension(self.attrs.doc_markdown, default=None, force=True)
        Span.set_extension(self.attrs.span_layout, default=None, force=True)
        Span.set_extension(self.attrs.span_data, default=None, force=True)
        Span.set_extension(self.attrs.span_heading, getter=self.get_heading, force=True)

    def __call__(self, source: Union[str, Path, bytes]) -> Doc:
        """Call parser on a path to create a spaCy Doc object."""
        # Convert the document using Azure
        return self._convert_with_azure(source)

    @overload
    def pipe(
        self,
        sources: Iterable[str | Path | bytes],
        as_tuples: Literal[False] = ...,
    ) -> Iterator[Doc]: ...

    @overload
    def pipe(
        self,
        sources: Iterable[tuple[str | Path | bytes, _AnyContext]],
        as_tuples: Literal[True] = ...,
    ) -> Iterator[tuple[Doc, _AnyContext]]: ...

    def pipe(
        self,
        sources: (
            Iterable[str | Path | bytes]
            | Iterable[tuple[str | Path | bytes, _AnyContext]]
        ),
        as_tuples: bool = False,
    ) -> Iterator[Doc] | Iterator[tuple[Doc, _AnyContext]]:
        """Process multiple documents and create spaCy Doc objects."""
        if as_tuples:
            sources = cast(Iterable[tuple[str | Path | bytes, _AnyContext]], sources)
            for source, context in sources:
                doc = self._convert_with_azure(source)
                yield (doc, context)
        else:
            sources = cast(Iterable[str | Path | bytes], sources)
            for source in sources:
                doc = self._convert_with_azure(source)
                yield doc

    def get_pages(self, doc: Doc) -> list[tuple[PageLayout, list[Span]]]:
        """Get pages and their spans."""
        pages = []

        # Get doc layout if available
        layout = getattr(doc._, self.attrs.doc_layout, None)
        if not layout:
            return []

        # Get spans
        spans = doc.spans.get(self.attrs.span_group, [])

        # Group spans by page
        for page in layout.pages:
            page_spans = []
            for span in spans:
                span_layout = getattr(span._, self.attrs.span_layout, None)
                if span_layout and span_layout.page_no == page.page_no:
                    page_spans.append(span)
            pages.append((page, page_spans))

        return pages

    def get_tables(self, doc: Doc) -> list[Span]:
        """Get all table spans from the document."""
        tables = []

        # Get spans
        spans = doc.spans.get(self.attrs.span_group, [])

        # Filter for table spans
        for span in spans:
            if span.label_ in TABLE_ITEM_LABELS:
                tables.append(span)

        return tables

    def get_heading(self, span: Span) -> Span | None:
        """
        Get the closest heading to a span.

        This is a getter method for Span._.heading extension
        and attempts to find the closest section title or header
        that this span belongs to.
        """
        doc = span.doc

        # Get layout spans
        spans = doc.spans.get(self.attrs.span_group, [])
        if not spans:
            return None

        # Get heading spans
        heading_spans = [s for s in spans if s.label_ in self.headings]
        if not heading_spans:
            return None

        # Get layout for current span
        layout = getattr(span._, self.attrs.span_layout, None)
        if not layout:
            return None

        # Find closest preceding heading on the same page
        closest_heading = None
        for heading in heading_spans:
            heading_layout = getattr(heading._, self.attrs.span_layout, None)
            if not heading_layout:
                continue

            # Heading must be on same page and before current span
            if heading_layout.page_no == layout.page_no and (
                (heading_layout.y < layout.y)
                or (heading_layout.y == layout.y and heading_layout.x <= layout.x)
            ):
                if closest_heading is None:
                    closest_heading = heading
                else:
                    closest_layout = getattr(
                        closest_heading._, self.attrs.span_layout, None
                    )
                    if closest_layout and heading_layout.y > closest_layout.y:
                        closest_heading = heading

        return closest_heading

    def _init_azure_client(
        self, endpoint: str = None, key: str = None, dotenv_path: str = None
    ):
        """Initialize Azure Document Intelligence client."""
        # Load environment variables from .env file if available
        if DOTENV_AVAILABLE and dotenv_path is not None:
            load_dotenv(dotenv_path)
        elif DOTENV_AVAILABLE and endpoint is None and key is None:
            load_dotenv()  # Try to load from default locations only if no explicit values

        self.endpoint = (
            endpoint
            if endpoint is not None
            else os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        )
        self.key = (
            key
            if key is not None
            else os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        )

        if not self.endpoint or not self.key:
            raise ValueError(
                "Azure Document Intelligence endpoint and key must be provided or "
                "set as environment variables (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY) or in a .env file"
            )

        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )

    def _convert_with_azure(self, source: Union[str, Path, bytes]) -> Doc:
        """Convert a document using Azure Document Intelligence directly to spaCy Doc."""
        # Handle different source types
        if isinstance(source, (str, Path)):
            str(source)
            with open(source, "rb") as f:
                content = f.read()
        else:
            content = source

        # Process with Azure
        poller = self.client.begin_analyze_document("prebuilt-layout", content)
        result = poller.result()

        # Convert directly to spaCy Doc
        return self._azure_to_doc(result)

    def _polygon_to_span_layout(
        self,
        polygon: list[float],
        page_no: int,
        page_height: float,
        scale_factor: float = PDF_POINTS_PER_INCH,
    ) -> SpanLayout:
        """Create a SpanLayout from an Azure polygon."""
        x_values = polygon[::2]
        y_values = polygon[1::2]

        # Scale coordinates from inches to points
        x = min(x_values) * scale_factor
        y = min(y_values) * scale_factor
        width = (max(x_values) - min(x_values)) * scale_factor
        height = (max(y_values) - min(y_values)) * scale_factor

        return SpanLayout(
            x=x,
            y=y,
            width=width,
            height=height,
            page_no=page_no,
        )

    def _azure_to_doc(self, result) -> Doc:
        """Convert Azure Document Intelligence result directly to spaCy Doc."""
        # Collect page information
        pages = {}
        page_heights = {}
        for page in result.pages:
            page_no = page.page_number
            # Azure may return None for dimensions in some formats like DOCX
            width = page.width * PDF_POINTS_PER_INCH if page.width else 8.5 * PDF_POINTS_PER_INCH  # Default to letter size
            height = page.height * PDF_POINTS_PER_INCH if page.height else 11 * PDF_POINTS_PER_INCH
            pages[page_no] = PageLayout(
                page_no=page_no,
                width=width,
                height=height,
            )
            page_heights[page_no] = height

        # Process all content in reading order
        texts = []
        spans_info = []

        # Combine paragraphs and tables in the order they appear
        all_items = []

        # Add paragraphs
        for idx, paragraph in enumerate(result.paragraphs):
            all_items.append(("paragraph", idx, paragraph))

        # Add tables
        for idx, table in enumerate(result.tables):
            all_items.append(("table", idx, table))

        # Sort by position (using first bounding region if available)
        def get_position(item):
            _, _, content = item
            if hasattr(content, "bounding_regions") and content.bounding_regions:
                region = content.bounding_regions[0]
                return (region.page_number, region.polygon[1] if region.polygon else 0)
            return (1, 0)

        all_items.sort(key=get_position)

        # Process items in order
        for item_type, idx, content in all_items:
            if item_type == "paragraph":
                # Add text
                text = content.content
                if not text:
                    continue

                # Add separator if needed
                if texts and self.sep is not None:
                    texts.append(self.sep)

                texts.append(text)

                # Get layout information
                layout = None
                if content.bounding_regions:
                    region = content.bounding_regions[0]
                    page_no = region.page_number
                    if region.polygon and page_no in page_heights:
                        layout = self._polygon_to_span_layout(
                            region.polygon, page_no, page_heights[page_no]
                        )

                # Store span info
                spans_info.append(
                    {"text": text, "label": "text", "layout": layout, "data": None}
                )

            elif item_type == "table":
                # Convert table to DataFrame
                rows = content.row_count
                cols = content.column_count
                values = [[None for _ in range(cols)] for _ in range(rows)]

                # Fill cells with data
                for cell in content.cells:
                    values[cell.row_index][cell.column_index] = cell.content

                # Create DataFrame
                df = DataFrame(values)
                
                # Ensure columns are strings for display
                if not df.empty and len(df.columns) > 0:
                    df.columns = [str(col) for col in df.columns]

                # Generate table text
                if callable(self.display_table) and not df.empty:
                    text = self.display_table(df)
                elif isinstance(self.display_table, str):
                    text = self.display_table
                else:
                    text = TABLE_PLACEHOLDER

                # Add separator if needed
                if texts and self.sep is not None:
                    texts.append(self.sep)

                texts.append(text)

                # Get layout information
                layout = None
                if content.bounding_regions:
                    region = content.bounding_regions[0]
                    page_no = region.page_number
                    if region.polygon and page_no in page_heights:
                        layout = self._polygon_to_span_layout(
                            region.polygon, page_no, page_heights[page_no]
                        )

                # Store span info
                spans_info.append(
                    {
                        "text": text,
                        "label": "table",
                        "layout": layout,
                        "data": df if not df.empty else None,
                    }
                )

        # Create the spaCy Doc
        full_text = "".join(texts)
        doc = self.nlp.make_doc(full_text)

        # Create span group
        span_group = SpanGroup(doc, name=self.attrs.span_group)

        # Track character positions
        char_offset = 0
        span_idx = 0

        # Create spans
        for i, text in enumerate(texts):
            # Skip separators
            if self.sep is not None and text == self.sep:
                char_offset += len(text)
                continue

            # Get span info
            if span_idx < len(spans_info):
                info = spans_info[span_idx]
                span_idx += 1

                # Find token boundaries
                start_token = None
                end_token = None
                for token in doc:
                    if start_token is None and token.idx >= char_offset:
                        start_token = token.i
                    if token.idx + len(token.text) <= char_offset + len(text):
                        end_token = token.i + 1

                if start_token is not None and end_token is not None:
                    span = doc[start_token:end_token]
                    span.label = self.nlp.vocab.strings.add(info["label"])
                    span_group.append(span)

                    # Set layout
                    if info["layout"]:
                        span._.set(self.attrs.span_layout, info["layout"])

                    # Set data for tables
                    if info["data"] is not None:
                        span._.set(self.attrs.span_data, info["data"])

            char_offset += len(text)

        # Add spans to doc
        doc.spans[self.attrs.span_group] = span_group

        # Set document layout
        doc_pages = [pages[p] for p in sorted(pages.keys())]
        doc._.set(self.attrs.doc_layout, DocLayout(pages=doc_pages))

        return doc
