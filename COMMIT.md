# Add Azure AI Document Intelligence backend support

This commit adds support for Azure AI Document Intelligence as an alternative backend for document processing in spaCy-Layout, while maintaining the existing API and compatibility with Docling.

## Features

- **Multiple backend support**: Users can now choose between Docling and Azure for document processing
- **Adapter-based architecture**: Clean separation between backends with a common interface
- **Azure AI Document Intelligence integration**:
  - Extract document structure, text, and tables from PDFs
  - Convert Azure output to DoclingDocument format
  - Map Azure's layout information to spaCy Doc objects
- **Flexible credential management**:
  - Environment variables
  - .env files via python-dotenv
  - Direct parameter passing
- **Comprehensive documentation**:
  - Updated README with Azure examples
  - API documentation for new parameters
  - Integration test script for Azure backend

## Technical details

- **Adapter interface**: Created a `BackendAdapter` abstract class that defines a common interface for document processing backends
- **DoclingAdapter**: Wraps existing Docling functionality in the adapter interface
- **AzureAdapter**: Connects to Azure AI Document Intelligence API and converts output to Docling format
- **Modified spaCyLayout**: Added backend selection and configuration parameters
- **Test coverage**:
  - Unit tests for new adapter implementations
  - Integration tests for both backends
  - Tests automatically skip Azure tests if credentials are not available
- **Dependencies**:
  - Added Azure dependencies as optional extras in setup.py
  - Added python-dotenv for environment variable management