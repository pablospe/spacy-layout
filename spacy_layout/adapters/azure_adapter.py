import os
from pathlib import Path
from typing import Any, Union

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

# Import python-dotenv for environment variable management (optional)
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .base import BackendAdapter


class AzureAdapter(BackendAdapter):
    """Adapter for Azure AI Document Intelligence."""

    def __init__(self, endpoint: str = None, key: str = None, dotenv_path: str = None):
        """
        Initialize the Azure adapter.

        Args:
            endpoint: Azure Document Intelligence endpoint
            key: Azure Document Intelligence API key
            dotenv_path: Path to .env file containing Azure credentials
        """
        # Load environment variables from .env file if available
        if DOTENV_AVAILABLE and dotenv_path is not None:
            load_dotenv(dotenv_path)
        elif DOTENV_AVAILABLE:
            load_dotenv()  # Try to load from default locations

        self.endpoint = endpoint or os.environ.get(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
        )
        self.key = key or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not self.endpoint or not self.key:
            raise ValueError(
                "Azure Document Intelligence endpoint and key must be provided or "
                "set as environment variables (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY) or in a .env file"
            )

        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )

    def _azure_to_docling_document(self, result) -> DoclingDocument:
        """
        Convert Azure Document Intelligence result to DoclingDocument.

        Args:
            result: Azure Document Intelligence analysis result

        Returns:
            DoclingDocument: Document in Docling format
        """
        # Create an empty DoclingDocument
        doc = DoclingDocument()

        # Process pages
        for page_idx, page in enumerate(result.pages):
            # Create a page item (1-indexed in Docling)
            page_item = PageItem(
                page_no=page.page_number,
                size={"width": page.width, "height": page.height},
                image=None,
            )
            doc.pages[page_idx + 1] = page_item

        # Process content
        item_id = 0

        # Handle paragraphs and text
        for paragraph in result.paragraphs:
            page_number = (
                paragraph.bounding_regions[0].page_number
                if paragraph.bounding_regions
                else 1
            )

            # Create bounding box
            bbox = None
            if paragraph.bounding_regions:
                region = paragraph.bounding_regions[0]
                # Convert Azure polygon to Docling bounding box
                x_values = [p.x for p in region.polygon]
                y_values = [p.y for p in region.polygon]

                bbox = BoundingBox(
                    l=min(x_values),
                    r=max(x_values),
                    t=min(y_values),
                    b=max(y_values),
                    coord_origin=CoordOrigin.TOPLEFT,
                )

            # Create text item
            text_item = TextItem(
                self_ref=f"text_{item_id}",
                text=paragraph.content,
                label=DocItemLabel.TEXT,
                prov=[{"bbox": bbox, "page_no": page_number}] if bbox else None,
            )
            doc.texts.append(text_item)
            item_id += 1

        # Handle tables
        for table_idx, table in enumerate(result.tables):
            page_number = (
                table.bounding_regions[0].page_number if table.bounding_regions else 1
            )

            # Create bounding box
            bbox = None
            if table.bounding_regions:
                region = table.bounding_regions[0]
                # Convert Azure polygon to Docling bounding box
                x_values = [p.x for p in region.polygon]
                y_values = [p.y for p in region.polygon]

                bbox = BoundingBox(
                    l=min(x_values),
                    r=max(x_values),
                    t=min(y_values),
                    b=max(y_values),
                    coord_origin=CoordOrigin.TOPLEFT,
                )

            # Collect table data
            rows = table.row_count
            cols = table.column_count
            cells = [[None for _ in range(cols)] for _ in range(rows)]

            # Fill cells with data
            for cell in table.cells:
                row_idx = cell.row_index
                col_idx = cell.column_index
                cells[row_idx][col_idx] = cell.content

            # Create table item
            table_item = TableItem(
                self_ref=f"table_{table_idx}",
                label=DocItemLabel.TABLE,
                prov=[{"bbox": bbox, "page_no": page_number}] if bbox else None,
                table={
                    "rows": rows,
                    "cols": cols,
                    "values": cells,
                    "header_rows": 1 if rows > 0 else 0,
                    "header_cols": 0,
                },
            )
            doc.tables.append(table_item)

        return doc

    def convert(self, source: Union[str, Path, bytes]) -> DoclingDocument:
        """
        Convert a document source to a DoclingDocument using Azure AI.

        Args:
            source: Path to document, bytes, or other source format

        Returns:
            DoclingDocument: Document in Docling format
        """
        # Handle different source types
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                content = f.read()
        else:
            content = source

        # Process with Azure
        poller = self.client.begin_analyze_document("prebuilt-layout", content)
        result = poller.result()

        # Convert to DoclingDocument
        return self._azure_to_docling_document(result)

    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """
        Convert multiple document sources to DoclingDocuments.

        Args:
            sources: List of document sources

        Returns:
            List of DoclingDocuments
        """
        # Process each document individually
        # Azure doesn't have a batch processing API similar to Docling
        results = []
        for source in sources:
            result = self.convert(source)
            # Wrap to match Docling's output format
            results.append(type("obj", (object,), {"document": result}))
        return results
