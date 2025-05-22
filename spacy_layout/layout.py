from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
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
from spacy.tokens import Doc, Span, SpanGroup

from .adapters.azure_adapter import AzureAdapter
from .adapters.docling_adapter import DoclingAdapter
from .model import BaseItem, Document, ItemLabel, TableItem, TextItem
from .types import Attrs, DocLayout, PageLayout, SpanLayout
from .util import decode_df, decode_obj, encode_df, encode_obj, get_bounding_box

if TYPE_CHECKING:
    from pandas import DataFrame
    from spacy.language import Language

# Type variable for contexts piped with documents
_AnyContext = TypeVar("_AnyContext")

TABLE_PLACEHOLDER = "TABLE"
TABLE_ITEM_LABELS = [ItemLabel.TABLE, ItemLabel.DOCUMENT_INDEX]

# Register msgpack encoders and decoders for custom types
srsly.msgpack_encoders.register("spacy-layout.dataclass", func=encode_obj)
srsly.msgpack_decoders.register("spacy-layout.dataclass", func=decode_obj)
srsly.msgpack_encoders.register("spacy-layout.dataframe", func=encode_df)
srsly.msgpack_decoders.register("spacy-layout.dataframe", func=decode_df)


class spaCyLayout:
    def __init__(
        self,
        nlp: "Language",
        separator: str | None = "\n\n",
        attrs: dict[str, str] = {},
        headings: list[str] = [
            ItemLabel.SECTION_HEADER,
            ItemLabel.PAGE_HEADER,
            ItemLabel.TITLE,
        ],
        display_table: Callable[["DataFrame"], str] | str = TABLE_PLACEHOLDER,
        backend: Literal["docling", "azure"] = "docling",
        backend_options: dict[str, Any] = None,
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

        # Initialize the backend adapter
        backend_options = backend_options or {}
        if backend == "docling":
            self.adapter = DoclingAdapter(
                format_options=backend_options.get("format_options")
            )
        elif backend == "azure":
            self.adapter = AzureAdapter(
                endpoint=backend_options.get("endpoint"),
                key=backend_options.get("key"),
                dotenv_path=backend_options.get("dotenv_path"),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Set spaCy extension attributes for custom data
        Doc.set_extension(self.attrs.doc_layout, default=None, force=True)
        Doc.set_extension(self.attrs.doc_pages, getter=self.get_pages, force=True)
        Doc.set_extension(self.attrs.doc_tables, getter=self.get_tables, force=True)
        Doc.set_extension(self.attrs.doc_markdown, default=None, force=True)
        Span.set_extension(self.attrs.span_layout, default=None, force=True)
        Span.set_extension(self.attrs.span_data, default=None, force=True)
        Span.set_extension(self.attrs.span_heading, getter=self.get_heading, force=True)

    def __call__(self, source: Union[str, Path, bytes, Document]) -> Doc:
        """Call parser on a path to create a spaCy Doc object."""
        if isinstance(source, Document):
            result = source
        else:
            # Use the adapter to convert the document
            result = self.adapter.convert(source)
        return self._result_to_doc(result)

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
            data = [source for source, _ in sources]
            contexts = [context for _, context in sources]
            results = self.adapter.convert_all(data)
            for result, context in zip(results, contexts):
                yield (self._result_to_doc(result.document), context)
        else:
            sources = cast(Iterable[str | Path | bytes], sources)
            results = self.adapter.convert_all(list(sources))
            for result in results:
                yield self._result_to_doc(result.document)

    def _result_to_doc(self, document: Document) -> Doc:
        """
        Convert internal Document model to spaCy Doc.

        Args:
            document: Internal Document model

        Returns:
            Doc: spaCy Doc with layout information
        """
        inputs = []
        pages = {
            (page.page_no): PageLayout(
                page_no=page.page_no,
                width=page.size.width if page.size else 0,
                height=page.size.height if page.size else 0,
            )
            for page in document.pages.values()
        }
        doc_pages = [pages[p] for p in sorted(pages.keys())]

        # Get all items in reading order
        items = document.get_all_items()

        # Print debug information
        print(f"Document has {len(items)} items")
        print(f"Document has {len(document.texts)} text items")
        print(f"Document has {len(document.tables)} table items")

        # Track headings for lookup
        heading_items = {}

        # Process each item and create span entries
        spans = []
        for idx, item in enumerate(items):
            # For tables, use a placeholder text
            text = ""
            if isinstance(item, TableItem):
                # Convert table to DataFrame or use placeholder
                if callable(self.display_table):
                    df = None
                    if hasattr(item, "export_to_dataframe"):
                        df = item.export_to_dataframe()
                    if df is not None and not df.empty:
                        text = self.display_table(df)
                    else:
                        text = TABLE_PLACEHOLDER
                else:
                    text = self.display_table
            elif isinstance(item, TextItem):
                text = item.text
            elif hasattr(item, "text"):
                text = item.text

            # Skip empty text
            if not text:
                continue

            # Add separator between inputs
            if (
                self.sep is not None
                and inputs
                and inputs[-1] != self.sep
                and text != self.sep
            ):
                inputs.append(self.sep)

            # Add input text
            inputs.append(text)

            # Get label as string
            if isinstance(item.label, ItemLabel):
                label = item.label.value
            elif hasattr(item.label, "__str__"):
                label = str(item.label)
            else:
                label = "text"

            # Track heading items for lookup
            if label in self.headings:
                heading_items[idx] = text

            # Store span information
            spans.append(
                {
                    "text": text,
                    "label": label,
                    "id": idx,
                    "item": item,
                }
            )

        # Create the Doc
        doc = self._texts_to_doc(inputs, spans, document)

        # Store layout information
        doc._.set(self.attrs.doc_layout, DocLayout(pages=doc_pages))

        # Store markdown representation if available
        doc._.set(
            self.attrs.doc_markdown,
            document.to_markdown() if hasattr(document, "to_markdown") else "",
        )

        return doc

    def _texts_to_doc(
        self, texts: list[str], spans: list[dict], document: Document
    ) -> Doc:
        """
        Create a Doc object from texts with spans.

        Args:
            texts: List of text segments
            spans: List of span information
            document: Internal Document model for accessing page info

        Returns:
            Doc: Created spaCy Doc with spans
        """
        # Create the doc with concatenated texts
        text = "".join(texts)
        doc = self.nlp.make_doc(text)

        # Create span group for layout spans
        span_group = SpanGroup(doc, name=self.attrs.span_group)

        # Track character offsets
        start_idx = 0
        start_char = 0
        in_span = False

        # Create spans
        span_idx = 0
        span_info = None
        layout_items = {}
        layout_spans = {}

        # Iterate through the texts and create spans
        for i, inp in enumerate(texts):
            end_idx = start_idx + len(self.nlp.make_doc(inp))
            end_char = start_char + len(inp)

            if i % 2 == 0 and self.sep is not None:
                # Start of a text span
                span_info = spans[span_idx]
                span_idx += 1
                in_span = True
            elif self.sep is not None:
                # End of a text span
                in_span = False

            if in_span and span_info:
                span = doc[start_idx:end_idx]
                span.label = self.nlp.vocab.strings.add(span_info["label"])
                span.id = span_info["id"]
                span_group.append(span)

                # Get bounding box for the span's layout
                layout = self._get_span_layout(span_info["item"], document)
                if layout:
                    span._.set(self.attrs.span_layout, layout)

                # For tables, also store the data
                if span_info["label"] in [label.value for label in TABLE_ITEM_LABELS]:
                    item = span_info["item"]
                    if isinstance(item, TableItem) and item.table:
                        try:
                            df = item.table.to_dataframe()
                            # Only set data if dataframe is valid
                            if df is not None and not df.empty and len(df.columns) > 0:
                                # Make sure columns are strings
                                if None in df.columns:
                                    df.columns = [
                                        f"Column {i + 1}"
                                        for i in range(len(df.columns))
                                    ]
                                span._.set(self.attrs.span_data, df)
                        except Exception as e:
                            print(f"Error converting table to DataFrame: {e}")

                # Store the span for heading lookup
                layout_spans[span_info["id"]] = span
                layout_items[span_info["id"]] = span_info

            # Update character indices
            start_idx = end_idx
            start_char = end_char

        # Add spans to document
        doc.spans[self.attrs.span_group] = span_group

        return doc

    def _get_span_layout(self, item: BaseItem, document: Document) -> SpanLayout | None:
        """Extract span layout from a document item."""
        if not hasattr(item, "prov") or not item.prov:
            return None

        # Find the first provenance with a bounding box
        for p in item.prov:
            if p.bbox:
                page_info = document.pages.get(p.page_no)
                (x, y, w, h) = get_bounding_box(p.bbox, page_info.size.height)
                return SpanLayout(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    page_no=p.page_no,
                )
        return None

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
            if span.label_ in [label.value for label in TABLE_ITEM_LABELS]:
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
