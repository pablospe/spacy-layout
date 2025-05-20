import pytest
from pathlib import Path

import spacy
from spacy_layout.adapters.docling_adapter import DoclingAdapter

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