import os
from pathlib import Path
from typing import Any, List, Union

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from ..model import (
    BoundingBox,
    CoordOrigin,
    Document,
    ItemLabel,
    ItemProvenance,
    PageInfo,
    Size,
    TableCellData,
    TableData,
    TableItem,
    TextItem,
)
from .base import BackendAdapter

PDF_POINTS_PER_INCH = 72


def create_bbox_from_polygon(
    polygon: List[float],
    scale_factor: float = PDF_POINTS_PER_INCH,
) -> BoundingBox:
    """
    Create a BoundingBox from an Azure polygon.

    Args:
        polygon: List of alternating x, y coordinates
        coord_origin: Coordinate system origin
        scale_factor: Factor to scale coordinates (e.g., 72 for inches to points)

    Returns:
        BoundingBox: Converted bounding box
    """
    x_values = polygon[::2]
    y_values = polygon[1::2]

    return BoundingBox(
        l=min(x_values) * scale_factor,
        r=max(x_values) * scale_factor,
        t=min(y_values) * scale_factor,
        b=max(y_values) * scale_factor,
        coord_origin=CoordOrigin.TOPLEFT,
    )


# Import python-dotenv for environment variable management (optional)
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


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

    def _azure_to_document(self, result) -> Document:
        """
        Convert Azure Document Intelligence result to internal Document model.

        Args:
            result: Azure Document Intelligence analysis result

        Returns:
            Document: Document in internal format
        """
        # Create a new document with source metadata
        doc = Document(
            source_type="azure_document_intelligence",
        )

        # Process pages
        for page_idx, page in enumerate(result.pages):
            # Create a page item (1-indexed in our model)
            # Convert page dimensions from inches to PDF points
            page_info = PageInfo(
                page_no=page.page_number,
                size=Size(
                    width=page.width * PDF_POINTS_PER_INCH,
                    height=page.height * PDF_POINTS_PER_INCH,
                ),
                image=None,
            )
            doc.pages[page_idx + 1] = page_info

        # Handle paragraphs and text
        for paragraph_idx, paragraph in enumerate(result.paragraphs):
            page_number = (
                paragraph.bounding_regions[0].page_number
                if paragraph.bounding_regions
                else 1
            )

            # Create bounding box
            bbox = None
            if paragraph.bounding_regions:
                region = paragraph.bounding_regions[0]
                bbox = create_bbox_from_polygon(region.polygon)

            # Create text item
            text_item = TextItem(
                self_ref=f"text_{paragraph_idx}",
                text=paragraph.content,
                label=ItemLabel.TEXT,
                prov=[
                    ItemProvenance(
                        bbox=bbox,
                        page_no=page_number,
                        source="azure",
                        confidence=paragraph.confidence
                        if hasattr(paragraph, "confidence")
                        else None,
                    )
                ]
                if bbox
                else [],
            )

            # Add to document
            doc.add_item(text_item)

        # Handle tables
        for table_idx, table in enumerate(result.tables):
            page_number = (
                table.bounding_regions[0].page_number if table.bounding_regions else 1
            )

            # Create bounding box
            bbox = None
            if table.bounding_regions:
                region = table.bounding_regions[0]
                bbox = create_bbox_from_polygon(region.polygon)

            # Collect table data
            rows = table.row_count
            cols = table.column_count
            values = [[None for _ in range(cols)] for _ in range(rows)]
            cells = []

            # Fill cells with data
            for cell in table.cells:
                row_idx = cell.row_index
                col_idx = cell.column_index
                values[row_idx][col_idx] = cell.content

                # Create cell data
                cell_data = TableCellData(
                    row_idx=row_idx,
                    col_idx=col_idx,
                    text=cell.content,
                    rowspan=cell.row_span
                    if hasattr(cell, "row_span") and cell.row_span is not None
                    else 1,
                    colspan=cell.column_span
                    if hasattr(cell, "column_span") and cell.column_span is not None
                    else 1,
                    row_header=False,  # Azure doesn't detect this directly
                    column_header=False,
                    is_merged=(
                        (getattr(cell, "row_span", 1) or 1) > 1
                        or (getattr(cell, "column_span", 1) or 1) > 1
                    ),
                )
                cells.append(cell_data)

            # Create table data
            table_data = TableData(
                rows=rows,
                cols=cols,
                values=values,
                header_rows=1 if rows > 0 else 0,  # Assume first row is header
                header_cols=0,
                cells=cells,
                caption=None,  # Azure doesn't extract captions
            )

            # Create table item
            table_item = TableItem(
                self_ref=f"table_{table_idx}",
                label=ItemLabel.TABLE,
                prov=[
                    ItemProvenance(
                        bbox=bbox,
                        page_no=page_number,
                        source="azure",
                        confidence=table.confidence
                        if hasattr(table, "confidence")
                        else None,
                    )
                ]
                if bbox
                else [],
                table=table_data,
            )

            # Add to document
            doc.add_item(table_item)

        return doc

    def convert(self, source: Union[str, Path, bytes]) -> Document:
        """
        Convert a document source to our internal Document model using Azure AI.

        Args:
            source: Path to document, bytes, or other source format

        Returns:
            Document: Document in internal format
        """
        # Handle different source types
        if isinstance(source, (str, Path)):
            source_path = str(source)
            with open(source, "rb") as f:
                content = f.read()
        else:
            source_path = None
            content = source

        # Process with Azure
        poller = self.client.begin_analyze_document("prebuilt-layout", content)
        result = poller.result()

        # Convert to internal Document model
        doc = self._azure_to_document(result)

        # Set source path if available
        if source_path:
            doc.source_path = source_path
            doc.name = Path(source_path).stem

        return doc

    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """
        Convert multiple document sources to internal Document model.

        Args:
            sources: List of document sources

        Returns:
            List of Document objects wrapped in a compatible format
        """
        # Process each document individually
        # Azure doesn't have a batch processing API similar to Docling
        results = []
        for source in sources:
            result = self.convert(source)
            # Wrap to match expected output format
            results.append(type("obj", (object,), {"document": result}))
        return results
