from datetime import datetime
import os
import pandas as pd
import psycopg
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

def is_document_exists(doc_id, chunk_id):
    """Check if the document or chunk already exists in the database using metadata filtering."""
    
    # SQL query to check for the document based on doc_id and chunk_id
    check_sql = f"""
    SELECT 1
    FROM {vec.vector_settings.table_name}
    WHERE metadata->>'doc_id' = %s AND metadata->>'chunk_id' = %s::text
    LIMIT 1
    """
    
    # Perform the check directly in the database
    with psycopg.connect(vec.settings.database.service_url) as conn:
        with conn.cursor() as cur:
            cur.execute(check_sql, (doc_id, chunk_id))
            result = cur.fetchone()
    
    # If result is not None, the document or chunk already exists
    return result is not None


# Function to load and process PDFs
def load_and_process_pdfs(pdf_folder):
    """
    Load PDF files, split their contents into chunks, and create a DataFrame.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.

    Returns:
        pd.DataFrame: A dataframe with chunked contents and metadata.
    """
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()
            documents.extend(docs)

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(documents)

    # Convert to DataFrame
    data = []
    for i, chunk in enumerate(chunked_docs):
        data.append({
            "chunk_id": i,
            "doc_id": chunk.metadata.get("source", filename),
            "content": chunk.page_content
        })
    return pd.DataFrame(data)

# Replace this path with the folder where your PDFs are stored
pdf_folder_path = "/Users/mattiashaughom/Documents/Repositories/AI repositories/rag_model/data"
df = load_and_process_pdfs(pdf_folder_path)


def prepare_record(row):
    """Prepare a record for insertion into the vector store."""
    doc_id = row["doc_id"]
    chunk_id = row["chunk_id"]

    # Check if the document or chunk already exists
    if is_document_exists(doc_id, chunk_id):
        print(f"Skipping existing document: {doc_id}, chunk: {chunk_id}")
        return None  # Skip this record if it already exists

    content = row["content"]
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )

# Apply the preparation to each row
records_df = df.apply(prepare_record, axis=1).dropna()

# Create tables and insert data into the vector store
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.create_keyword_search_index()  # GIN Index
vec.upsert(records_df)