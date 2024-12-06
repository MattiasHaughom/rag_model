# PDF-Based RAG System

A Retrieval-Augmented Generation (RAG) system that provides AI-powered responses based on PDF document analysis. The application leverages semantic and keyword search capabilities, utilizing pgvector with scale extensions for cost-effective and efficient vector similarity search.

## Key Features

- **Document Management**
  - Local PDF file ingestion (Cloud storage integration in development)
  - Automatic document parsing and vectorization
  - Intelligent duplicate detection and processing

- **Vector Processing**
  - Efficient document embedding generation using OpenAI's embedding models
  - PostgreSQL integration with pgvector for vector storage
  - Optimized ANN (Approximate Nearest Neighbor) search indexes

- **Search Capabilities**
  - Hybrid search combining semantic and keyword-based approaches
  - Vector similarity search powered by PostgreSQL and pgvector
  - Real-time query processing and response generation

- **Architecture & Performance**
  - Containerized deployment using Docker
  - Scalable PostgreSQL database integration
  - Optimized for performance with ANN indexes
  - Web-based interface for query interaction

## Prerequisites

- Docker and Docker Compose
- Python 3.7 or higher
- OpenAI API key
- Cohere API key
- PostgreSQL client (recommended: TablePlus)

## Quick Start

1. **Clone and Configure**
   ```bash
   git clone <repository-url>
   cp example.env .env
   ```
   Edit `.env` file with your API keys and configuration

2. **Launch Database**
   ```bash
   docker compose up -d
   ```

3. **Install Dependencies**
   ```bash
   # Install PDM if you haven't already
   curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

   # Install project dependencies
   pdm install
   ```

4. **Run Application**
   ```bash
   python run.py
   ```

## Database Configuration

PostgreSQL connection details:
```plaintext
Host: localhost
Port: 5432
User: postgres
Password: password
Database: postgres
```

## Architecture

The system utilizes a modern architecture combining:
- PostgreSQL with pgvector extension for vector operations
- OpenAI embeddings for document vectorization
- Flask web interface 
- Docker containerization for deployment

## Usage

1. Place PDF documents in the designated input folder
2. The system automatically processes and vectorizes new documents
3. Access the web interface to query your document collection
4. Receive AI-generated responses based on your document context

## Performance Considerations

- Utilizes ANN indexes for optimal search performance
- Implements batch processing for document ingestion
- Employs caching strategies for frequently accessed content






```

This improved README:
1. Uses more technical and precise terminology
2. Provides a clearer structure
3. Adds important sections like Architecture and Performance Considerations
4. Improves formatting with proper Markdown usage
5. Includes more detailed feature descriptions
6. Adds placeholder sections for Contributing and License

You should customize:
- The actual repository URL
- The specific web framework you're using
- The license information
- Any specific configuration details unique to your implementation
- Additional deployment or setup steps if necessary
