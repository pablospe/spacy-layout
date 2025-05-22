from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from ..model import Document


class BackendAdapter(ABC):
    """Interface for document processing backend adapters."""

    @abstractmethod
    def convert(self, source: Union[str, Path, bytes]) -> Document:
        """
        Convert a document source to the internal Document model.

        Args:
            source: Path to document, bytes, or other source format

        Returns:
            Document: Document in internal format for further processing
        """
        pass

    @abstractmethod
    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """
        Convert multiple document sources.

        Args:
            sources: List of document sources

        Returns:
            List of document results
        """
        pass
