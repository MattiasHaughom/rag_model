from datetime import datetime
import os
import pandas as pd
import psycopg
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Initialize VectorStore
vec = VectorStore()

# Check if the database exists
if not vec.settings.database.exists():
    vec.create_tables()
    vec.create_index()  # DiskAnnIndex
    vec.create_keyword_search_index()  # GIN Index


def is_document_exists(doc_id, chunk_id):
    """
    Check if the document or chunk already exists in the database using metadata filtering.
    
    Args:
        doc_id (str): The document identifier
        chunk_id (int): The chunk identifier
    
    Returns:
        bool: True if the document exists, False otherwise
    
    Raises:
        psycopg.Error: If there's a database connection or query error
    """
    check_sql = """
    SELECT EXISTS (
        SELECT 1
        FROM {table}
        WHERE metadata->>'doc_id' = %s 
        AND metadata->>'chunk_id' = %s::text
    )
    """.format(table=vec.vector_settings.table_name)
    
    try:
        with psycopg.connect(vec.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(check_sql, (doc_id, chunk_id))
                return cur.fetchone()[0]
    except psycopg.Error as e:
        print(f"Database error: {e}")
        raise


# Function to load and process PDFs
def load_and_process_pdfs(pdf_folder: str, chunk_size: int = 500, chunk_overlap: int = 50) -> pd.DataFrame:
    """
    Load PDF files, split their contents into chunks, and create a DataFrame.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.
        chunk_size (int, optional): Size of each text chunk. Defaults to 500.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 50.

    Returns:
        pd.DataFrame: A dataframe with chunked contents and metadata.

    Raises:
        FileNotFoundError: If the pdf_folder doesn't exist
        ValueError: If no PDF files are found in the folder
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate folder
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Folder not found: {pdf_folder}")

    # Get PDF files
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_folder}")

    documents = []
    # Use tqdm for progress tracking
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Successfully loaded {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {str(e)}")
            continue

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")

    # Convert to DataFrame
    data: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunked_docs):
        source_file = Path(chunk.metadata.get("source", "unknown"))
        data.append({
            "chunk_id": i,
            "doc_id": f"{source_file.stem}_{source_file.stat().st_mtime_ns}",  # More unique ID
            "content": chunk.page_content,
            "page": chunk.metadata.get("page", 0),  # Additional metadata
            "total_pages": len(documents)  # Additional metadata
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} rows")
    return df

# Replace this path with the folder where your PDFs are stored
pdf_folder_path = os.getenv('PDF_FOLDER_PATH', os.path.join(os.path.dirname(__file__), '..', 'data'))

# Allow customization of chunk parameters
df = load_and_process_pdfs(
    pdf_folder_path,
    chunk_size=int(os.getenv('CHUNK_SIZE', '500')),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '50'))
)

def prepare_record(row):
    """
    Prepare a record for insertion into the vector store.
    
    Args:
        row (pd.Series): A row from the DataFrame containing document information
        
    Returns:
        dict: Prepared record with metadata and embedding, or None if document exists
    """
    doc_id = row["doc_id"]
    chunk_id = row["chunk_id"]
    
    try:
        # Check if the document or chunk already exists
        if is_document_exists(doc_id, chunk_id):
            logging.info(f"Skipping existing document: {doc_id}, chunk: {chunk_id}")
            return None

        content = row["content"]
        embedding = vec.get_embedding(content)
        
        return {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "created_at": datetime.now().isoformat(),
                "page": row.get("page", 0),  # Include additional metadata
                "total_pages": row.get("total_pages", 0),
            },
            "contents": content,
            "embedding": embedding,
        }
    except Exception as e:
        logging.error(f"Error preparing record for doc_id: {doc_id}, chunk: {chunk_id}: {str(e)}")
        return None

# Apply the preparation to each row
records_df = df.apply(prepare_record, axis=1)

# Drop any failed records (None values) and convert to DataFrame
records_df = pd.DataFrame([r for r in records_df if r is not None])
vec.upsert(records_df)