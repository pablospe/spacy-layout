"""
Integration test for Azure backend.

This script demonstrates how to use spaCyLayoutAzure with Azure Document Intelligence.
Run this script manually to test the integration.

Prerequisites:
- Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY
  environment variables or create a .env file with these variables
"""

import os
import sys
from pathlib import Path

import spacy

from spacy_layout import spaCyLayoutAzure

# Check if Azure credentials are set
if not os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or not os.environ.get(
    "AZURE_DOCUMENT_INTELLIGENCE_KEY"
):
    print("Warning: Azure Document Intelligence credentials not set.")
    print(
        "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY"
    )
    print("or create a .env file with these values.")

# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "data"
PDF_SIMPLE = TEST_DATA_DIR / "simple.pdf"
PDF_TABLE = TEST_DATA_DIR / "table.pdf"


def main():
    # Initialize spaCy and layout processor
    print("Initializing spaCy and spaCyLayoutAzure...")
    nlp = spacy.blank("en")
    layout = spaCyLayoutAzure(nlp)

    print(f"\nProcessing {PDF_SIMPLE}...")
    try:
        doc_simple = layout(PDF_SIMPLE)

        print("\nDocument structure:")
        print(f"- {len(doc_simple.spans['layout'])} layout spans")
        print(f"- {len(doc_simple._.layout.pages)} pages")

        # Print text spans
        print("\nText spans:")
        for i, span in enumerate(doc_simple.spans["layout"]):
            if span._.layout:
                print(f"- Span {i}: {span.label_}, page {span._.layout.page_no}")
                print(
                    f'  "{span.text[:50]}..."'
                    if len(span.text) > 50
                    else f'  "{span.text}"'
                )

        print(f"\nProcessing {PDF_TABLE}...")
        doc_table = layout(PDF_TABLE)

        # Print tables
        tables = doc_table._.tables
        print(f"\nFound {len(tables)} tables:")
        for i, table in enumerate(tables):
            print(f"- Table {i}:")
            df = table._.data
            print(f"  {len(df)} rows × {len(df.columns)} columns")
            print(f"  Columns: {', '.join(df.columns)}")
            print("  First 2 rows:")
            print(df.head(2))

        print("\nIntegration test completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
