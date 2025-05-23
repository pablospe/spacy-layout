import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pypdfium2 as pdfium
import spacy
from matplotlib.patches import Rectangle
from spacy.tokens import DocBin

from spacy_layout import spaCyLayoutAzure

# Define paths
TEST_DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "cache"

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(document_path):
    """Generate a unique cache filename based on document path"""
    # Create a hash of the document path for a unique identifier
    path_str = str(document_path)
    hash_obj = hashlib.md5(path_str.encode())
    hash_str = hash_obj.hexdigest()
    return CACHE_DIR / f"{hash_str}.spacy"


def process_with_caching(layout, document_path):
    """Process a document with caching support"""
    cache_path = get_cache_path(document_path)

    # Check if cache exists
    if cache_path.exists():
        print(f"Loading cached document: {document_path.name}")
        doc_bin = DocBin(store_user_data=True).from_disk(cache_path)
        docs = list(doc_bin.get_docs(layout.nlp.vocab))
        return docs[0]  # Return the first (and only) doc

    # Process document
    print(f"Processing document (first time): {document_path.name}")
    doc = layout(document_path)

    # Save to cache
    doc_bin = DocBin(store_user_data=True)
    doc_bin.add(doc)
    doc_bin.to_disk(cache_path)

    return doc


def visualize_pdf_with_boxes(pdf_path, doc, layout, page_num=0):
    """
    Visualize a PDF page with bounding boxes for layout spans.

    Args:
        pdf_path: Path to the PDF file
        doc: spaCy Doc processed with spaCy-Layout
        page_num: Page number to visualize (0-indexed)
    """
    # Load PDF and render the page
    pdf = pdfium.PdfDocument(pdf_path)
    page_image = pdf[page_num].render(scale=1)
    numpy_array = page_image.to_numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 16))

    # Display the PDF image
    ax.imshow(numpy_array)

    # Add rectangles for each section's bounding box
    page_idx = (
        page_num  # Page number is 1-indexed in spaCy-Layout but 0-indexed in pypdfium2
    )

    # Get page layout and spans
    layout_instance = getattr(doc._, layout.attrs.doc_layout)
    if not layout_instance or not layout_instance.pages:
        print(f"No layout information found for page {page_num}")
        return None

    # Get pages from the layout processor
    pages = layout.get_pages(doc)
    if page_idx >= len(pages):
        print(f"Page {page_num} not found. Document has {len(pages)} pages.")
        return None

    page_layout, page_spans = pages[page_idx]

    # Create color map for different span types
    colors = {
        "text": "blue",
        "title": "red",
        "section_header": "green",
        "page_header": "orange",
        "table": "purple",
        "list_item": "cyan",
        "image": "magenta",
    }

    # Add bounding boxes
    for span in page_spans:
        # Get bounding box coordinates
        span_layout = getattr(span._, layout.attrs.span_layout)
        if not span_layout:
            continue

        x = span_layout.x
        y = span_layout.y
        width = span_layout.width
        height = span_layout.height

        # Use specific color based on span type or default to gray
        color = colors.get(span.label_, "gray")

        # Create rectangle patch
        rect = Rectangle(
            (x, y), width, height, fill=False, color=color, linewidth=1, alpha=0.8
        )
        ax.add_patch(rect)

        # Add text label at top of box
        ax.text(
            x,
            y + height + 5,
            span.label_,
            fontsize=8,
            color=color,
            verticalalignment="bottom",
        )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=2, label=label)
        for label, color in colors.items()
        if any(span.label_ == label for span in page_spans)
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Hide axes
    ax.axis("off")

    # Show plot
    plt.tight_layout()
    return fig


def main():
    """Visualize bounding boxes from PDF extraction"""
    # Initialize spaCy and layout processor
    nlp = spacy.blank("en")
    layout = spaCyLayoutAzure(nlp)

    # Choose a PDF file to visualize
    pdf_paths = {
        # "simple": TEST_DATA_DIR / "simple.pdf",
        # "table": TEST_DATA_DIR / "table.pdf",
        "starcraft": TEST_DATA_DIR / "starcraft.pdf",
    }

    # Create output directory for visualizations
    output_dir = Path(__file__).parent / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Process each PDF and visualize bounding boxes
    for name, pdf_path in pdf_paths.items():
        print(f"Processing {name}.pdf...")

        # Load document (from cache if available)
        doc = process_with_caching(layout, pdf_path)

        # Get number of pages
        layout_instance = getattr(doc._, layout.attrs.doc_layout)
        if not layout_instance or not layout_instance.pages:
            print(f"No layout information found for {name}")
            continue

        num_pages = len(layout_instance.pages)
        print(f"Document has {num_pages} pages")

        # Visualize first page (or more if desired)
        for page_idx in range(num_pages):  # Visualize up to 2 pages
            print(f"Visualizing page {page_idx + 1}...")

            # Create visualization
            fig = visualize_pdf_with_boxes(pdf_path, doc, layout, page_idx)

            if fig is None:
                print(f"Skipping page {page_idx + 1} - no visualization created")
                continue

            # Save the figure
            output_path = output_dir / f"{name}_page{page_idx + 1}.png"
            fig.savefig(output_path, dpi=150)
            print(f"Saved visualization to {output_path}")

            # Close figure to free memory
            plt.close(fig)

    print(f"\nVisualizations saved to {output_dir}")
    print("To view these visualizations, you can use an image viewer or file browser.")


if __name__ == "__main__":
    main()
