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
- PostgreSQL client (im using pgadmin4)

## Quick Start

1. **Clone and Configure**
   ```bash
   git clone https://github.com/MattiasHaughom/rag_model.git
   cp .env
   ```
   Edit `.env` file with your API keys and configuration

2. **Launch Database**
   ```bash
   docker compose up -d
   ```
   You need to be in the docker directory to run this command.

3. **Install Dependencies**
   ```bash
   # Install PDM if you haven't already
   curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

   # Install project dependencies
   pdm install
   ```

4. **Run Application**
   ```bash
   pdm run python run.py
   ```


5. **Access Interface**
   Open your browser and navigate to `http://localhost:5000` to interact with the application.
   Only in development mode.


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


# Future work
- Search capabilities
  - Improve the keyword search to select the most relevant keywords
  - Include search history
- Cloud storage integration
  - Google docs integration
  - Perhaps azure postgres integration eventually?
- Web interface for query interaction
  - Need to improve the interface esthetics
- Improved user interface and experience
  - Structure of the output needs to be improved
- Improved error handling and user feedback
