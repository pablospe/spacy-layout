from io import BytesIO
from pathlib import Path
from typing import Any, List, Union

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter, FormatOption
from docling_core.types.doc.document import DoclingDocument

from ..model import (
    BoundingBox,
    CoordOrigin,
    Document,
    ItemLabel,
    ItemProvenance,
    PageInfo,
    Size,
    TableData,
    TableItem,
    TextItem,
)
from .base import BackendAdapter


def convert_docling_provenance(prov_items) -> List[ItemProvenance]:
    """
    Convert Docling provenance items to internal ItemProvenance format.

    Args:
        prov_items: Docling provenance items

    Returns:
        List[ItemProvenance]: Converted provenance list
    """
    prov = []  # Always initialize as an empty list for pydantic validation
    if prov_items:
        for p in prov_items:
            bbox = None
            if hasattr(p, "bbox") and p.bbox:
                bbox = BoundingBox(
                    l=p.bbox.l,
                    r=p.bbox.r,
                    t=p.bbox.t,
                    b=p.bbox.b,
                    coord_origin=CoordOrigin.BOTTOMLEFT,
                )
            page_no = getattr(p, "page_no", 1)
            prov.append(ItemProvenance(bbox=bbox, page_no=page_no))
    return prov


class DoclingAdapter(BackendAdapter):
    """Adapter for Docling document processing."""

    def __init__(self, format_options: dict[str, FormatOption] = None):
        """
        Initialize the Docling adapter.

        Args:
            format_options: Options for Docling DocumentConverter
        """
        self.converter = DocumentConverter(format_options=format_options)

    def _get_source(
        self, source: Union[str, Path, bytes]
    ) -> Union[str, Path, DocumentStream]:
        """Convert source to a format Docling can process."""
        if isinstance(source, (str, Path)):
            return source
        return DocumentStream(name="source", stream=BytesIO(source))

    def _docling_to_document(self, docling_doc: DoclingDocument) -> Document:
        """
        Convert DoclingDocument to our internal Document model.

        Args:
            docling_doc: DoclingDocument instance

        Returns:
            Document: Document in internal format
        """
        doc = Document()

        # Process pages
        for page_no, page in docling_doc.pages.items():
            page_info = PageInfo(
                page_no=page_no,
                size=Size(
                    width=page.size.width if page.size else 0,
                    height=page.size.height if page.size else 0,
                ),
                image=None,
            )
            doc.pages[page_no] = page_info

        # Process text items
        for text_item in docling_doc.texts:
            # Convert label from Docling to our format
            label = ItemLabel.TEXT
            if hasattr(text_item, "label") and text_item.label:
                try:
                    label = ItemLabel(text_item.label)
                except ValueError:
                    # If label doesn't match our enum, default to TEXT
                    pass

            # Convert provenance if available
            prov = convert_docling_provenance(text_item.prov)

            # Create the text item
            new_text_item = TextItem(
                self_ref=text_item.self_ref, text=text_item.text, label=label, prov=prov
            )
            # Add to document (which adds to both items dict and texts list)
            doc.add_item(new_text_item)

        # Process table items
        for table_item in docling_doc.tables:
            # Convert label
            label = ItemLabel.TABLE
            if hasattr(table_item, "label") and table_item.label:
                try:
                    label = ItemLabel(table_item.label)
                except ValueError:
                    # If label doesn't match our enum, default to TABLE
                    pass

            # Convert provenance if available
            prov = convert_docling_provenance(table_item.prov)

            # Convert table data
            table_data = None
            try:
                # Access using property or attribute
                if hasattr(table_item, "data") and table_item.data:
                    # Create from Docling TableData
                    rows = getattr(table_item.data, "num_rows", 0)
                    cols = getattr(table_item.data, "num_cols", 0)

                    # Extract cells from table_cells
                    values = [[None for _ in range(cols)] for _ in range(rows)]
                    if (
                        hasattr(table_item.data, "table_cells")
                        and table_item.data.table_cells
                    ):
                        for cell in table_item.data.table_cells:
                            if (
                                hasattr(cell, "row_idx")
                                and hasattr(cell, "col_idx")
                                and hasattr(cell, "text")
                            ):
                                # Only set if valid indices
                                if (
                                    0 <= cell.row_idx < rows
                                    and 0 <= cell.col_idx < cols
                                ):
                                    values[cell.row_idx][cell.col_idx] = cell.text

                    # Create our table data structure
                    table_data = TableData(
                        rows=rows,
                        cols=cols,
                        values=values,
                        header_rows=1 if rows > 0 else 0,
                        header_cols=0,
                    )
            except Exception:
                # Fallback to empty table data
                table_data = TableData(
                    rows=0, cols=0, values=[], header_rows=0, header_cols=0
                )

            # Create the table item
            new_table_item = TableItem(
                self_ref=table_item.self_ref, label=label, prov=prov, table=table_data
            )
            # Add to document (which adds to both items dict and tables list)
            doc.add_item(new_table_item)

        return doc

    def convert(self, source: Union[str, Path, bytes]) -> Document:
        """Convert a document source to our internal Document model."""
        docling_doc = self.converter.convert(self._get_source(source)).document
        return self._docling_to_document(docling_doc)

    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """Convert multiple document sources to our internal Document model."""
        data = [self._get_source(source) for source in sources]
        results = []
        for result in self.converter.convert_all(data):
            # Convert to our internal Document model
            document = self._docling_to_document(result.document)
            # Wrap in the same format as the original for compatibility
            results.append(type("obj", (object,), {"document": document}))
        return results
