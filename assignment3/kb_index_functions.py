# index_kb.py

import os
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- 1. Load Knowledge Base ---
def load_kb_data(path: str) -> list[Document]:
    """Loads and formats the KB data from JSON."""
    print(f"Loading data from {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return []

    documents = []
    for i, entry in enumerate(data):
        # Assuming the JSON structure has 'text' and 'source' or similar fields.
        # We combine text and a KB ID for content and use source for metadata.
        content = entry.get('content', entry.get('text', ''))
        source = entry.get('source', f"KB{i:03d}") # Use KBxxx as ID/Source
       
        # Create a document object
        documents.append(
            Document(
                page_content=content,
                metadata={"source": source, "id": f"KB{i:03d}"}
            )
        )
    print(f"Loaded {len(documents)} documents.")
    return documents

# --- 2. Initialize Embeddings and Vector DB ---
def setup_pinecone_and_embeddings():
    """Initializes Azure Embeddings and Pinecone Client."""
    # 2.1 Azure Embeddings
    embeddings_model = AzureOpenAIEmbeddings(
                model = os.getenv("EMBEDDING_MODEL_NAME"),
                deployment = os.getenv("EMBEDDING_DEPLOYMENT"), 
                azure_endpoint = os.getenv("EMBEDDING_ENDPOINT"),
                api_key = os.getenv("EMBEDDING_API_KEY"), 
                api_version = os.getenv("EMBEDDING_API_VERSION")
            )

    # 2.2 Pinecone Client Setup
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        # text-embedding-3-small has 1536 dimensions
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust cloud/region as needed
        )
        print("Index created. Waiting for initialization...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Index is ready.")
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)
    return embeddings_model, index

# --- 3. Indexing Function ---
def index_documents(documents: list[Document], embeddings_model, index):
    """Generates embeddings and upserts to Pinecone."""
    print("Generating embeddings and upserting vectors...")
    vectors_to_upsert = []
   
    for doc in documents:
        # Generate embedding
        vector = embeddings_model.embed_query(doc.page_content)
       
        # Prepare for upsert
        vectors_to_upsert.append(
            (
                doc.metadata['id'],
                vector,
                doc.metadata # Store KB ID and content in metadata for retrieval
            )
        )

    # Upsert in batches (or all at once for small datasets)
    index.upsert(vectors=vectors_to_upsert)
    print(f"Successfully indexed {len(vectors_to_upsert)} documents.")
    print(f"Index status: {index.describe_index_stats()}")