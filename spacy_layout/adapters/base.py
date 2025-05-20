from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from docling_core.types.doc.document import DoclingDocument


class BackendAdapter(ABC):
    """Interface for document processing backend adapters."""

    @abstractmethod
    def convert(self, source: Union[str, Path, bytes]) -> DoclingDocument:
        """
        Convert a document source to a DoclingDocument.

        Args:
            source: Path to document, bytes, or other source format

        Returns:
            DoclingDocument: Document in Docling format for further processing
        """
        pass

    @abstractmethod
    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """
        Convert multiple document sources to DoclingDocuments.

        Args:
            sources: List of document sources

        Returns:
            List of document results
        """
        pass