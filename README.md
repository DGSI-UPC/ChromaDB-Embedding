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
python index_markdown.py index /path/to/your/markdown/directory
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
python index_markdown.py index /path/to/markdown --chunk-size 1000 --chunk-overlap 100

# Using OpenAI embeddings (requires API key)
python index_markdown.py index /path/to/markdown --use-openai
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

## Querying the Index

### Using Command Line

The script provides a simple command-line interface for searching:

```bash
# Basic search
python index_markdown.py query "your search query"

# Search with more results
python index_markdown.py query "your search query" --n-results 10

# Search using OpenAI embeddings
python index_markdown.py query "your search query" --use-openai

# Filter results by file path
python index_markdown.py query "your search query" --path-filter "docs/"

# Specify different database path
python index_markdown.py query "your search query" --db-path "custom_db"
```

### Using Python

You can query the indexed documents in two ways:

1. Using the existing indexer:
```python
from index_markdown import MarkdownIndexer

# Initialize with the same settings used for indexing
indexer = MarkdownIndexer(
    persist_dir="chroma_db",  # use the same directory as indexing
    use_openai=True  # set to True if you used OpenAI embeddings for indexing
)

# Simple query
results = indexer.query_documents(
    query_text="your search query here",
    n_results=5  # number of results to return
)

# Process results
for i, (document, metadata, score) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
)):
    print(f"\nResult {i+1} (similarity: {1-score:.3f})")
    print(f"Source: {metadata['source_path']}")
    print(f"Chunk: {metadata['chunk_index']+1}/{metadata['total_chunks']}")
    print("Content:", document)
```

2. Using ChromaDB directly:
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize the client
client = chromadb.PersistentClient(path="chroma_db")

# Get the collection
collection = client.get_collection(
    name="markdown_docs",
    # Use the same embedding function as during indexing
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
    # Or for OpenAI:
    # embedding_function=embedding_functions.OpenAIEmbeddingFunction(
    #     api_key="your_key",
    #     model_name="text-embedding-3-small"
    # )
)

# Query with filters
results = collection.query(
    query_texts=["your search query"],
    n_results=5,
    # Optional: filter by metadata
    where={"source_path": {"$contains": "specific/path"}},
    # Optional: include relevance score
    include=["metadatas", "documents", "distances"]
)
```

### Advanced Query Features

1. **Metadata Filtering**: Filter results based on metadata fields:
```python
# Filter by specific file path
results = collection.query(
    query_texts=["query"],
    where={"source_path": {"$contains": "docs/"}}
)

# Filter by chunk index
results = collection.query(
    query_texts=["query"],
    where={"chunk_index": {"$lt": 3}}  # only first 3 chunks
)
```

2. **Batch Queries**: Search multiple queries at once:
```python
results = collection.query(
    query_texts=["query1", "query2", "query3"],
    n_results=3
)
```

### Query Results

Results include:
- **Document Content**: The text chunk that matches your query
- **Metadata**:
  * `source_path`: Original markdown file path
  * `chunk_index`: Position of the chunk in the document
  * `total_chunks`: Total number of chunks in the document
  * Any frontmatter metadata from the original markdown
- **Distance Score**: Lower scores indicate better matches (using cosine similarity)

Results are ordered by semantic similarity to the query, with the most relevant chunks appearing first.
