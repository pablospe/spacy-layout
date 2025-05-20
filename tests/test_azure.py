import pytest
import os
from pathlib import Path

import spacy
from spacy_layout.adapters.docling_adapter import DoclingAdapter
from spacy_layout.adapters.azure_adapter import AzureAdapter

TEST_DATA_DIR = Path(__file__).parent / "data"
PDF_SIMPLE = TEST_DATA_DIR / "simple.pdf"


def test_docling_adapter():
    """Test the Docling adapter directly."""
    adapter = DoclingAdapter()
    result = adapter.convert(PDF_SIMPLE)
    assert result is not None
    assert hasattr(result, "texts")
    assert hasattr(result, "tables")
    assert len(result.texts) > 0


def test_docling_adapter_convert_all():
    """Test batch processing with Docling adapter."""
    adapter = DoclingAdapter()
    results = adapter.convert_all([PDF_SIMPLE])
    assert len(results) == 1
    assert hasattr(results[0], "document")
    assert len(results[0].document.texts) > 0


# Skip Azure tests if credentials are not available
azure_credentials_available = (
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in os.environ
    and "AZURE_DOCUMENT_INTELLIGENCE_KEY" in os.environ
)

@pytest.mark.skipif(
    not azure_credentials_available,
    reason="Azure Document Intelligence credentials not available"
)
def test_azure_adapter():
    """Test the Azure adapter directly."""
    adapter = AzureAdapter()
    result = adapter.convert(PDF_SIMPLE)
    assert result is not None
    assert hasattr(result, "texts")
    assert hasattr(result, "tables")
    assert len(result.texts) > 0


@pytest.mark.skipif(
    not azure_credentials_available,
    reason="Azure Document Intelligence credentials not available"
)
def test_azure_adapter_convert_all():
    """Test batch processing with Azure adapter."""
    adapter = AzureAdapter()
    results = adapter.convert_all([PDF_SIMPLE])
    assert len(results) == 1
    assert hasattr(results[0], "document")
    assert len(results[0].document.texts) > 0