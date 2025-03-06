# Markdown to ChromaDB Indexer

This tool indexes markdown files into ChromaDB for efficient semantic search capabilities, with support for both default embeddings and OpenAI's text-embedding-3-small model for enhanced search quality.

## Features

- Flexible embedding options:
  - Default ChromaDB embeddings (no API key required)
  - Optional OpenAI text-embedding-3-small model for enhanced quality
- Recursively processes markdown files in a directory
- Intelligent text chunking with configurable size and overlap
- Sentence-aware splitting to maintain context
- Extracts and preserves frontmatter metadata
- Converts markdown to searchable text
- Stores documents with their metadata in ChromaDB
- Supports semantic search queries
- Batch processing for large datasets

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set up OpenAI embeddings:
   - Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

To index your markdown files:

```bash
python index_markdown.py /path/to/your/markdown/directory
```

Optional arguments:
- `--db-path`: Specify a custom path for ChromaDB persistence (default: "chroma_db")
- `--chunk-size`: Maximum number of characters per chunk (default: 500)
- `--chunk-overlap`: Number of characters to overlap between chunks (default: 50)
- `--use-openai`: Use OpenAI embeddings instead of default embeddings (requires API key)

### Examples

Index with custom settings:
```bash
# Using default embeddings
python index_markdown.py /path/to/markdown --chunk-size 1000 --chunk-overlap 100

# Using OpenAI embeddings (requires API key)
python index_markdown.py /path/to/markdown --use-openai
```

## Python API Usage

```python
from index_markdown import MarkdownIndexer

# Initialize the indexer with custom settings
indexer = MarkdownIndexer(
    persist_dir="chroma_db",
    chunk_size=500,  # characters per chunk
    chunk_overlap=50,  # overlap between chunks
    use_openai=True  # set to True to use OpenAI embeddings
)

# Index a directory of markdown files
indexer.index_directory("/path/to/markdown/files")

# Query the indexed documents
results = indexer.query_documents("your search query", n_results=5)
```

## Text Chunking

The indexer uses an intelligent chunking strategy:

1. **Sentence-Aware Splitting**: Text is split at sentence boundaries to maintain context
2. **Configurable Chunk Size**: Control the size of each chunk (default: 500 characters)
3. **Overlap Between Chunks**: Maintains context between chunks (default: 50 characters)
4. **Metadata Preservation**: Each chunk maintains:
   - Original document metadata
   - Chunk index
   - Total chunks in document
   - Source file path

## Batch Processing

Documents are processed in batches (100 chunks per batch) to efficiently handle large datasets and manage memory usage.

## Notes

- Processes all files with `.md` or `.markdown` extensions
- Each chunk is stored with complete metadata for traceability
- Uses BeautifulSoup for robust HTML parsing
- ChromaDB persistence directory is created if it doesn't exist
- Unique IDs are generated for each chunk (format: `filename_chunk_N`)

## Query Results

When querying, results include:
- Chunk content
- Original document metadata
- Chunk position information
- Relevance scores

Results are ordered by semantic similarity to the query.
