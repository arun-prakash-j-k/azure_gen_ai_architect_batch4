from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tabulate import tabulate
import chromadb
import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class CourseRecommendationEngine():

    def __init__(self):
        # load environment variables
        load_dotenv(".env")

        # Initialize the embedding and chat model
        self.embeddings = self._initialize_embedding_model()
        self.llm = self._initialize_llm_model()
        # Initialize vector db & collection
        self.collection = self._intialize_chromadb_collection()
        # Get the course catalog from source
        self.df_courses = self._initialize_course_catalog()
        self._describe_course_catalog()
        # Generate Embeddings and store it to vector DB
        self.docs, self.doc_ids, self.metadatas = self._split_text()
        self._generate_embedding_and_add_to_collection()

        # Define Prompt Templates
        self.query_transform_prompt = self._initialize_query_transform_prompt()
        self.rag_prompt = self._initialize_rag_prompt()
       
        # Define the RAG Chain
        self.rag_chain = self._initialize_rag_chain()


    def _initialize_embedding_model(self):
        try:
            # initialize the azure open ai embedding model
            embeddings = AzureOpenAIEmbeddings(
                model = os.getenv("EMBEDDING_MODEL_NAME"),
                deployment = os.getenv("EMBEDDING_DEPLOYMENT"), 
                azure_endpoint = os.getenv("EMBEDDING_ENDPOINT"),
                api_key = os.getenv("EMBEDDING_API_KEY"), 
                api_version = os.getenv("EMBEDDING_API_VERSION")
            )
            print("embedding model initialization completed")
            return embeddings
        except Exception as e:
            print(f"error in initializing the embedding model - {e}")
            raise

    def _initialize_llm_model(self):
        try:
            # initialize the azure open ai chat model (GPT-4o for analysis/generation)
            llm = AzureChatOpenAI(
                openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                temperature = 0.0
            )
            print("LLM (GPT-4o) model initialization completed")
            return llm
        except Exception as e:
            print(f"error in initializing the LLM model - {e}")
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

    def _document_retriever_from_chroma(self, query, n_retrieved=10):
        try:
            query_embedding = self.embeddings.embed_query(query)
            # Retrieve documents
            results = self.collection.query(
                query_embeddings = [query_embedding],
                n_results = n_retrieved,
                include = ["documents", "metadatas", "distances"]
            )
            # Convert results to list of Document objects
            retrieved_docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
           
            # Filter for top N unique course titles (to avoid multiple chunks from the same course dominating the context)
            unique_titles = set()
            final_docs = []
            top_n = 5 # We aim for top 5 courses as context for final answer
            for doc in retrieved_docs:
                 title = doc.metadata.get('title')
                 if title not in unique_titles and len(unique_titles) < top_n:
                      unique_titles.add(title)
                      final_docs.append(doc)
           
            print(f"Retriever found {len(retrieved_docs)} chunks, reduced to {len(final_docs)} unique course documents for RAG.")
            return final_docs
        except Exception as e:
            print(f"Error in retriever: {e}")
            raise

    def _initialize_query_transform_prompt(self):
        # Prompt to rephrase the user's natural language query into an optimal search query
        template = """
        You are an intelligent course recommendation assistant. Your task is to analyze the user's query and suggest the 'next best recommendation' by rephrasing it into a highly specific and effective search query for a course catalog vector database.
       
        The goal is to generate a query that will retrieve the most relevant course descriptions.
       
        User Query: {user_query}
       
        Generate the single, most relevant search query (e.g., 'Advanced Python development with Django and machine learning fundamentals'). Do not include any explanation or extra text.
        """
        return ChatPromptTemplate.from_template(template)
    
    def _initialize_rag_prompt(self):
        # Prompt for the final answer generation using retrieved context
        template = """
        You are a helpful and expert course recommendation engine. Based *only* on the following retrieved course information, provide the user with the top 5 course recommendations.
       
        For each course, list the 'title' and provide a brief, compelling summary of its content, focusing on why it might be relevant to the user's initial interest.
       
        The courses are provided in the context below. You *must* select the top 5 unique courses (by title) and format the output clearly.
       
        Context:
        {context}
       
        User's Initial Query: {user_query}
       
        Top 5 Course Recommendations:
        """
        return ChatPromptTemplate.from_template(template)
    
    def _initialize_rag_chain(self):
        # Step 1: Query Transformation Chain
        query_transformer = (
            self.query_transform_prompt | self.llm | StrOutputParser()
        ).with_config(run_name="QueryTransformer")
       
        # Step 2: Retriever (Using the custom method as a runnable)
        retriever_runnable = RunnableLambda(
            lambda x: self._document_retriever_from_chroma(x['transformed_query'], n_retrieved=10)
        ).with_config(run_name="CustomRetriever")

        # Step 3: Final Generation Chain
        final_generator = (
            RunnablePassthrough.assign(context=lambda x: "\n\n---\n\n".join([doc.page_content + f" (Title: {doc.metadata.get('title', 'Unknown Title')})" for doc in x['retrieved_docs']]))
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        ).with_config(run_name="FinalGenerator")
       
        # Step 4: Full RAG Flow (LangChain Expression Language)
        rag_chain = (
            RunnablePassthrough.assign(
                transformed_query=query_transformer
            )
            | RunnablePassthrough.assign(
                retrieved_docs=retriever_runnable
            )
            | final_generator
        ).with_config(run_name="CourseRecommendationRAG")

        return rag_chain

    def recommend_courses(self, user_query: str):
        """
        Executes the two-step RAG process:
        1. LLM generates the best search query.
        2. Vector DB is searched using the generated query.
        3. LLM synthesizes the top 5 retrieved results into a final recommendation.
        """
        print(f"\n--- Starting Course Recommendation for: '{user_query}' ---")
       
        # The chain handles all steps now: transform -> retrieve -> generate
        response = self.rag_chain.invoke({"user_query": user_query})
       
        print("\n--- Recommendation Complete ---")
        return response
    
# course_recommender = CourseRecommendationEngine()