from pathlib import Path
from typing import Any, Union
from io import BytesIO

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter, FormatOption
from docling_core.types.doc.document import DoclingDocument

from .base import BackendAdapter


class DoclingAdapter(BackendAdapter):
    """Adapter for Docling document processing."""

    def __init__(self, format_options: dict[str, FormatOption] = None):
        """
        Initialize the Docling adapter.

        Args:
            format_options: Options for Docling DocumentConverter
        """
        self.converter = DocumentConverter(format_options=format_options)

    def _get_source(self, source: Union[str, Path, bytes]) -> Union[str, Path, DocumentStream]:
        """Convert source to a format Docling can process."""
        if isinstance(source, (str, Path)):
            return source
        return DocumentStream(name="source", stream=BytesIO(source))

    def convert(self, source: Union[str, Path, bytes]) -> DoclingDocument:
        """Convert a document source to a DoclingDocument."""
        return self.converter.convert(self._get_source(source)).document

    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """Convert multiple document sources to DoclingDocuments."""
        data = [self._get_source(source) for source in sources]
        return list(self.converter.convert_all(data))