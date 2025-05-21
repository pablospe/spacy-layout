# Azure AI Document Intelligence Integration Summary

## Overview

This implementation extends spaCy-Layout to support Azure AI Document Intelligence as an alternative backend for PDF/document processing. The goal was to maintain the existing spaCy-Layout API while allowing users to choose between Docling and Azure for document processing.

## Architecture

We implemented an adapter-based architecture that:

1. Keeps `spaCyLayout` as the main public interface
2. Introduces backend adapters to handle specific document processors
3. Normalizes different outputs into a consistent format for spaCy integration

```
+----------------+     +----------------------+
|                |     |                      |
|  spaCyLayout   +---->+  BackendAdapter      |
|                |     |  (Interface)         |
+----------------+     +----------+-----------+
                                  |
                +-----------------+------------------+
                |                                    |
    +-----------v-----------+         +-------------v-----------+
    |                       |         |                         |
    |  DoclingAdapter       |         |  AzureAdapter           |
    |                       |         |                         |
    +-----------+-----------+         +-------------+-----------+
                |                                   |
    +-----------v-----------+         +-------------v-----------+
    |                       |         |                         |
    |  Docling API          |         |  Azure AI Document      |
    |                       |         |  Intelligence API        |
    +-----------------------+         +-------------------------+
```

## Implementation Details

### 1. Adapter Interface

Created a `BackendAdapter` abstract base class that defines the contract for document processing backends:

```python
class BackendAdapter(ABC):
    @abstractmethod
    def convert(self, source: Union[str, Path, bytes]) -> DoclingDocument:
        """Convert a document source to a DoclingDocument."""
        pass

    @abstractmethod
    def convert_all(self, sources: list[Union[str, Path, bytes]]) -> list[Any]:
        """Convert multiple document sources to DoclingDocuments."""
        pass
```

### 2. Backend Adapters

- **DoclingAdapter**: Wraps the existing Docling functionality in the new interface
- **AzureAdapter**: Implements the adapter interface for Azure AI Document Intelligence:
  - Connects to Azure Document Intelligence API
  - Processes documents with Azure
  - Converts Azure output to DoclingDocument format
  - Maps Azure layout elements, paragraphs, and tables to match Docling structure

### 3. spaCyLayout Integration

Modified the `spaCyLayout` class to:
- Add backend selection parameter (`backend="docling"` or `backend="azure"`)
- Update initialization to use the appropriate adapter
- Support backend-specific options through `backend_options` parameter
- Maintain backward compatibility for existing code

### 4. Testing

- Created unit tests for both adapters
- Added integration tests for Azure backend
- Implemented automatic skipping of Azure tests when credentials are not available
- Verified compatibility with existing functionality

## Usage Instructions

### Installation

```bash
# Install with Docling support only (default)
pip install spacy-layout

# Install with Azure AI Document Intelligence support
pip install "spacy-layout[azure]"
```

### Basic Usage

```python
import spacy
from spacy_layout import spaCyLayout

# Use Docling backend (default)
nlp = spacy.blank("en")
layout = spaCyLayout(nlp, backend="docling")

# Use Azure backend
layout = spaCyLayout(
    nlp,
    backend="azure",
    backend_options={
        "endpoint": "https://your-resource.cognitiveservices.azure.com/",
        "key": "your-api-key"
    }
)

# Process documents with either backend using the same API
doc = layout("./document.pdf")
```

### Setting Azure Credentials

There are three ways to provide Azure credentials:

1. **Environment variables**:
   ```python
   # Set in your shell or script
   os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = "your-endpoint"
   os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"] = "your-key"
   ```

2. **.env file** (requires python-dotenv):
   ```python
   # Create a .env file with:
   # AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your-endpoint
   # AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

   # Then initialize with:
   layout = spaCyLayout(nlp, backend="azure")  # Auto-loads from .env

   # Or specify a custom path:
   layout = spaCyLayout(
       nlp,
       backend="azure",
       backend_options={"dotenv_path": "/path/to/.env"}
   )
   ```

3. **Direct parameter passing**:
   ```python
   layout = spaCyLayout(
       nlp,
       backend="azure",
       backend_options={
           "endpoint": "https://your-resource.cognitiveservices.azure.com/",
           "key": "your-api-key"
       }
   )
   ```

## Future Improvements

1. **Enhanced Azure Feature Support**:
   - Add support for more Azure Document Intelligence models
   - Support for form recognition and custom models
   - Include confidence scores from Azure in metadata

2. **Performance Optimizations**:
   - Implement batch processing for Azure
   - Add caching for processed documents
   - Stream processing for large documents

3. **Cross-Backend Calibration**:
   - Tools to normalize and calibrate results between backends
   - Benchmark suite for comparing accuracy
   - Methods to combine results from multiple backends
