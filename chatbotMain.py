import os
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

openai_api = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pinecone_client = Pinecone(pinecone_api)

index = pinecone_client.Index(pinecone_index_name)

embeddings = OpenAIEmbeddings()







