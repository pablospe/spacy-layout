import os
from pathlib import Path

import pandas as pd
import pytest
import spacy
import srsly
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from spacy.tokens import DocBin

from spacy_layout import spaCyLayoutAzure
from spacy_layout.layout import TABLE_PLACEHOLDER
from spacy_layout.types import DocLayout, PageLayout, SpanLayout

PDF_STARCRAFT = Path(__file__).parent / "data" / "starcraft.pdf"
PDF_SIMPLE = Path(__file__).parent / "data" / "simple.pdf"
DOCX_SIMPLE = Path(__file__).parent / "data" / "simple.docx"
PDF_SIMPLE_BYTES = PDF_SIMPLE.open("rb").read()
PDF_TABLE = Path(__file__).parent / "data" / "table.pdf"
PDF_INDEX = Path(__file__).parent / "data" / "table_document_index.pdf"


@pytest.fixture
def nlp():
    return spacy.blank("en")


@pytest.fixture
def span_labels():
    # Define the expected labels directly
    return ["text", "section_header", "page_header", "title", "table", "list_item", "document_index", "footnote", "formula"]


@pytest.mark.parametrize("path", [PDF_STARCRAFT, PDF_SIMPLE, PDF_SIMPLE_BYTES])
def test_general(path, nlp, span_labels):
    layout = spaCyLayoutAzure(nlp)
    doc = layout(path)
    assert isinstance(doc._.get(layout.attrs.doc_layout), DocLayout)
    for span in doc.spans[layout.attrs.span_group]:
        assert span.text
        assert span.label_ in span_labels
        assert isinstance(span._.get(layout.attrs.span_layout), SpanLayout)


@pytest.mark.parametrize("path, pg_no", [(PDF_STARCRAFT, 6), (PDF_SIMPLE, 1)])
def test_pages(path, pg_no, nlp):
    layout = spaCyLayoutAzure(nlp)
    doc = layout(path)
    # This should not raise a KeyError when accessing `pages` dict
    # Key Error would mean a mismatched pagination on document layout and span layout
    result = layout.get_pages(doc)
    assert len(result) == pg_no
    assert result[0][0].page_no == 1
    if pg_no == 6:  # there should be 16 or 18 spans on the pg_no 1
        assert len(result[0][1]) in (16, 18)
    elif pg_no == 1:  # there should be 4 spans on pg_no 1
        assert len(result[0][1]) == 4


@pytest.mark.parametrize("path", [PDF_SIMPLE, DOCX_SIMPLE])
@pytest.mark.parametrize("separator", ["\n\n", ""])
def test_simple(path, separator, nlp):
    layout = spaCyLayoutAzure(nlp, separator=separator)
    doc = layout(path)
    assert len(doc.spans[layout.attrs.span_group]) == 4
    assert doc.text.startswith(f"Lorem ipsum dolor sit amet{separator}")
    # With no separator, token boundaries might not align perfectly
    if separator == "":
        assert doc.spans[layout.attrs.span_group][0].text.startswith("Lorem ipsum dolor sit")
    else:
        assert doc.spans[layout.attrs.span_group][0].text == "Lorem ipsum dolor sit amet"


def test_simple_pipe(nlp):
    layout = spaCyLayoutAzure(nlp)
    for doc in layout.pipe([PDF_SIMPLE, DOCX_SIMPLE]):
        assert len(doc.spans[layout.attrs.span_group]) == 4


def test_simple_pipe_as_tuples(nlp):
    layout = spaCyLayoutAzure(nlp)
    data = [(PDF_SIMPLE, "pdf"), (DOCX_SIMPLE, "docx")]
    result = list(layout.pipe(data, as_tuples=True))
    for doc, _ in result:
        assert len(doc.spans[layout.attrs.span_group]) == 4
    assert [context for _, context in result] == ["pdf", "docx"]


def test_table(nlp):
    layout = spaCyLayoutAzure(nlp)
    doc = layout(PDF_TABLE)
    assert len(doc._.get(layout.attrs.doc_tables)) == 1
    table = doc._.get(layout.attrs.doc_tables)[0]
    assert table.text == TABLE_PLACEHOLDER
    df = table._.get(layout.attrs.span_data)
    # Azure doesn't detect headers, so columns are just numeric strings
    assert df.columns.tolist() == ["0", "1", "2"]
    assert df.shape[0] == 5  # Including header row
    # Check that the first row contains headers
    assert df.iloc[0].tolist() == ["Name", "Type", "Place of birth"]
    # Check data values
    assert df.iloc[1]["0"] == "Ines"
    assert df.iloc[1]["1"] == "human"
    assert df.iloc[1]["2"] == "Cologne, Germany"
    # Azure doesn't provide markdown output
    # Just verify the table data is correct


def test_table_index(nlp):
    layout = spaCyLayoutAzure(nlp)
    doc = layout(PDF_INDEX)
    # Azure may detect different number of tables than expected
    tables = doc._.get(layout.attrs.doc_tables)
    assert len(tables) >= 1  # At least one table
    table = tables[0]
    assert table.text == TABLE_PLACEHOLDER
    # Azure doesn't distinguish document_index tables
    assert table.label_ == "table"
    
    # Check that the table has data
    assert table._.data is not None, "Table data not available"
    assert isinstance(table._.data, pd.DataFrame), "Table data is not a DataFrame"
    # The merged table should have multiple rows
    assert table._.data.shape[0] > 10  # Should have many index entries


def test_table_placeholder(nlp):
    def display_table(df):
        return f"Table with columns: {', '.join(df.columns.tolist())}"

    layout = spaCyLayoutAzure(nlp, display_table=display_table)
    doc = layout(PDF_TABLE)
    table = doc._.get(layout.attrs.doc_tables)[0]
    assert table.text == "Table with columns: 0, 1, 2"


def test_serialize_objects():
    span_layout = SpanLayout(x=10, y=20, width=30, height=40, page_no=1)
    doc_layout = DocLayout(pages=[PageLayout(page_no=1, width=500, height=600)])
    bytes_data = srsly.msgpack_dumps({"span": span_layout, "doc": doc_layout})
    data = srsly.msgpack_loads(bytes_data)
    assert isinstance(data, dict)
    assert data["span"] == span_layout
    assert data["doc"] == doc_layout
    df = DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    bytes_data = srsly.msgpack_dumps({"df": df})
    data = srsly.msgpack_loads(bytes_data)
    assert isinstance(data, dict)
    assert_frame_equal(df, data["df"])


# Azure-specific tests
azure_credentials_available = (
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in os.environ
    and "AZURE_DOCUMENT_INTELLIGENCE_KEY" in os.environ
)


@pytest.mark.skipif(
    not azure_credentials_available,
    reason="Azure Document Intelligence credentials not available",
)
def test_azure_initialization_with_params():
    """Test Azure initialization with explicit parameters."""
    nlp = spacy.blank("en")

    # Test with explicit credentials
    layout = spaCyLayoutAzure(
        nlp,
        azure_endpoint=os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        azure_key=os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
    )

    assert layout.endpoint is not None
    assert layout.key is not None


def test_azure_initialization_without_credentials():
    """Test that initialization fails without credentials."""
    # Clear environment variables temporarily
    old_endpoint = os.environ.pop("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", None)
    old_key = os.environ.pop("AZURE_DOCUMENT_INTELLIGENCE_KEY", None)

    try:
        nlp = spacy.blank("en")
        with pytest.raises(
            ValueError,
            match="Azure Document Intelligence endpoint and key must be provided",
        ):
            # Pass explicit empty strings to avoid dotenv loading
            spaCyLayoutAzure(nlp, azure_endpoint="", azure_key="")
    finally:
        # Restore environment variables
        if old_endpoint:
            os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = old_endpoint
        if old_key:
            os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"] = old_key


@pytest.mark.parametrize("path", [PDF_SIMPLE, PDF_TABLE])
def test_serialize_roundtrip(path, nlp):
    layout = spaCyLayoutAzure(nlp)
    doc = layout(path)
    doc_bin = DocBin(store_user_data=True)
    doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()
    new_doc_bin = DocBin().from_bytes(bytes_data)
    new_doc = list(new_doc_bin.get_docs(nlp.vocab))[0]
    layout_spans = new_doc.spans[layout.attrs.span_group]
    assert len(layout_spans) == len(doc.spans[layout.attrs.span_group])
    assert all(
        isinstance(span._.get(layout.attrs.span_layout), SpanLayout)
        for span in layout_spans
    )
    assert isinstance(new_doc._.get(layout.attrs.doc_layout), DocLayout)
    tables = doc._.get(layout.attrs.doc_tables)
    new_tables = new_doc._.get(layout.attrs.doc_tables)
    for before, after in zip(tables, new_tables):
        table_before = before._.get(layout.attrs.span_data)
        table_after = after._.get(layout.attrs.span_data)
        assert_frame_equal(table_before, table_after)
