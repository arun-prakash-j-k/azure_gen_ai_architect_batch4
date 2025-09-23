from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tabulate import tabulate
import chromadb
import numpy as np
import pandas as pd

class CourseRecommendationEngine():

    def __init__(self):
        # load environment variables
        load_dotenv(".env")

        self.embeddings = self._initialize_embedding_model()
        self.collection = self._intialize_chromadb_collection()

        self.df_courses = self._initialize_course_catalog()
        self._describe_course_catalog()

        self.docs, self.doc_ids, self.metadatas = self._split_text()
        self._generate_embedding_and_add_to_collection()


    def _initialize_embedding_model(self):
        try:
            # initialize the azure open ai embedding model
            embedding_model_endpoint = os.getenv("OPENAI_ENDPOINT")
            embedding_model_name = os.getenv("OPENAI_MODEL_NAME")
            embedding_deployment_name = os.getenv("OPENAI_DEPLOYMENT")
            embedding_model_api_version = os.getenv("OPENAI_API_VERSION")

            embeddings = AzureOpenAIEmbeddings(
                model = embedding_model_name,
                deployment = embedding_deployment_name, 
                azure_endpoint = embedding_model_endpoint,
                api_key = os.getenv("OPENAI_API_KEY"), 
                api_version = embedding_model_api_version
            )
            print("embedding model initialization completed")
            return embeddings
        except Exception as e:
            print(f"error in initializing the embedding model - {e}")
            raise

    def _initialize_course_catalog(self):
        try:
            df_courses = pd.read_csv(os.getenv("COURSE_CATALOG_URI"))
            print("course catalog initialization completed")
            return df_courses
        except Exception as e:
            print(f"error in initializing the course catalog - {e}")
            raise

    def _describe_course_catalog(self):
        print("--- statistical description of course catalog data:")
        self.df_courses['desc_length'] = self.df_courses['description'].apply(lambda x: len(x))
        print(tabulate(self.df_courses.describe(include='all').T, headers='keys', tablefmt="psql"))
        pass

    def _intialize_chromadb_collection(self):
        try:
            # Initialize ChromaDB client (in-memory)
            chroma_client = chromadb.Client()
            # Create collection
            try:
                # Delete existing collection if it exists to avoid conflicts
                chroma_client.delete_collection(name="course_catalog")
            except:
                pass
            collection = chroma_client.create_collection(name="course_catalog", metadata={"use_type": "COURSE_RECOMMENDATION"})
            print("chroma-db collection initialization completed")
            return collection
        except Exception as e:
            print("error in initializing the chromadb collection - {e}")
            raise

    def _split_text(self):
        try:
            max_size = int(np.max(self.df_courses['desc_length']))
            min_size = int(np.min(self.df_courses['desc_length']))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=int(min_size/2))
            docs = []
            doc_ids = []
            metadatas = []
            for i, (title, desc) in enumerate(zip(self.df_courses['title'], self.df_courses['description'])):
                desc = str(desc)  # Ensure description is a string
                if len(desc.strip()) > 100:  # Skip short or empty descriptions
                    split_docs = text_splitter.split_text(desc)
                    for j, split in enumerate(split_docs):
                        docs.append(split)
                        doc_ids.append(f"{i}_{j}")
                        metadatas.append({"title": str(title)})
            print(f"splitting the text completed - splitted into {len(docs)} documents")
            print(f"-----Sample ID: {doc_ids[0]}")
            print(f"-----Sample Metadata: {metadatas[0]}")
            print(f"-----Sample Content: {docs[0][:200]}...")
            return docs, doc_ids, metadatas
        except Exception as e:
            print("error in splitting the text - {e}")
            raise

    def _generate_embedding_and_add_to_collection(self):
        # Generate embeddings for documents
        try:
            self.embeddings_list = self.embeddings.embed_documents(self.docs)
            print("embedding generation completed")
        except Exception as e:
            print(f"error in generating embeddings: {e}")
            raise
        # Add documents to the collection
        try:
            print(f"--no of docs before update in the db: {self.collection.count()}")
            self.collection.add(
                documents = self.docs,
                ids = self.doc_ids,
                metadatas = self.metadatas,
                embeddings = self.embeddings_list
            )
            print("adding the embedding to the collection completed")
            print(f"--no of docs after update in the db: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            raise

    def document_retriever_from_chroma(self, query, n_retrieved=10):
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings = [query_embedding],
                n_results = n_retrieved,
                include = ["documents", "metadatas", "distances"]
            )# documents that are retrieved are already sorted by distance in ascending order
            # return results
            return [Document(page_content=doc, metadata=meta) for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
        except Exception as e:
            print(f"Error in retriever: {e}")



engine = CourseRecommendationEngine()