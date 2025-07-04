import time
import os
import requests
import json
from google.cloud import storage
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from uuid import uuid4
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

load_dotenv("api.env")

def env_access():
    openai_api = os.getenv("OPENAI_API_KEY")
    pinecone_api = os.getenv("PINECONE_API_KEY")                    # Init .env objects]
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    user_agent = os.getenv("USER_AGENT")
  
    return openai_api, pinecone_api, pinecone_index_name, user_agent

def bucket_access(bucket_name):
    bucket_name = "aicanbucket"
    client = storage.Client()                                   # Init bucket objects 
    bucket = client.bucket(bucket_name)
  
    return bucket


def init_pinecone(pinecone_api, pinecone_index_name):
    pinecone_client = Pinecone(pinecone_api)  # Init client object https://docs.pinecone.io/reference/api/authentication?utm_source=chatgpt.com
    
    if pinecone_index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=pinecone_index_name,           # Checks for a pinmecone index, if not makes one
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    index = pinecone_client.Index(pinecone_index_name) # Use for later vector store and upload https://docs.pinecone.io/reference/api/2024-07/data-plane/upsert?utm_source=chatgpt.com
    
    index_stats=index.describe_index_stats()
  
    return index, index_stats

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def doc_to_dict_(docs):
    dicts = []
    for doc in docs:
        add_dicts = {                                   # For changing document objects to dictionary format for json upload
                "page_content": doc.page_content,
                "metadata": doc.metadata 
            }
        dicts.append(add_dicts)
    return dicts
   
def dict_to_doc(dicts):
    docs = []                               # For changing dictionary format back to document objects
    for d in dicts:
        doc = Document(page_content=d.get("page_content",""), metadata=d.get("metadata", {}))
        docs.append(doc)
    return docs

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def save_load_to_bucket(bucket, blob_name, c_function, on_load=None, on_save=None):
    blob = bucket.blob(blob_name)
    
    if blob.exists():
        print(f"File {blob_name} loaded from {bucket}")
        blob_data = blob.download_as_text()                         # Checks if file with json already exists and extracts the data as document objects
        json_data = json.loads(blob_data)
        if on_load:
            json_data = on_load(json_data)
        return json_data
  
    print(f"File {blob_name} not found in {bucket}. Creating new file.")
    create_data = c_function()
    if on_save:                                                                                 
        create_data = on_save(create_data)
        blob.upload_from_string(json.dumps(create_data), content_type="application/json")       # Converts document objects to json and uploads it 
        print(f"saved {blob_name} to {bucket}")
        create_data = on_load(create_data)
    else:
        blob.upload_from_string(json.dumps(create_data), content_type="application/json")               # If fild not found , creates file and uploads the data as json
        print(f"saved {blob_name} to {bucket}")     
    return create_data 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def url_scrape(url):
    response = requests.get(url)  # GET request 
    soup = BeautifulSoup(response.content, "lxml")  # Lxml recommended: faster parser and better for larger and possibly non-perfect formats https://www.crummy.com/software/BeautifulSoup/bs4/doc/#differences-between-parsers 
    doc_links = []

    for div in soup.find_all('div', class_ = 'region region-content'):
        for a_tag in div.find_all("a", href=True):                              # Searches for 
            href = a_tag.get("href")
            doc_links.append(href)
          
    return doc_links


def doc_loader(doc_links, user_agent):
  headers = {"User-Agent": user_agent}
  loader = WebBaseLoader(web_paths=(doc_links),header_template=headers)     # Uses webbased loader from langchain community, loads multiple webpages into document object
  docs = loader.load()
  
  for doc in docs:      
      doc.page_content = ' '.join(doc.page_content.split())  # Replace multiple whitespace chars with a single space and strip leading/trailing spaces

  for doc in docs:
      doc.metadata = {"source": doc.metadata.get("source", "")}  # Add source url to metadata for each document, removes the rest of metadata to keep size smaller
  
  return docs


def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)                            # Uses recursive character text splitter to split large documents into smaller chunks
  
    return chunks

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def upsert_to_index(vector_store, chunks):
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        doc_id = [str(uuid4()) for _ in chunk_batch]                    # Uses add_documents from langchain to add documents to pinecone vector store
        try:                                                                    # USed batch to help improve performance and reduce API calls
            vector_store.add_documents(documents=chunk_batch, ids=doc_id)
        except Exception as e:
            print(f"Error on batch {i//batch_size}: {e}")
        time.sleep(10.0)
            

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
  
    openai_api, pinecone_api, pinecone_index_name, user_agent = env_access()
    bucket = bucket_access("aicanbucket")
    pinecone_index, index_stats = init_pinecone(pinecone_api, pinecone_index_name)
    embeddings = OpenAIEmbeddings(api_key=openai_api)  # Init open ai embeddings, no arg = default embeddings from langchain and auto pull api key
    vector_store = PineconeVectorStore(pinecone_index, embeddings)  # Init vector store using pinecone index and embeddings
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    url = "https://www.irs.gov/irm/part2"
  
    doc_links = save_load_to_bucket(bucket, "links.json", lambda: url_scrape(url))                  # Clean up and data rwapping fucnctions
  
    docs = save_load_to_bucket(bucket, "docs.json", lambda: doc_loader(doc_links, user_agent), on_load=dict_to_doc, on_save=doc_to_dict_)
  
    chunks = save_load_to_bucket(bucket, "chunks.json", lambda: split_docs(docs), on_load=dict_to_doc, on_save=doc_to_dict_)
  
    print (index_stats)
    print (len(chunks))             
    
    
    if index_stats['total_vector_count'] == 0:  
        upsert_to_index(vector_store, chunks)               # Checks to see if vector store is empty and adds chunks to it
        
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8})        # Sets langchain retriever with mmr (maximal marginal relevance)
                                                                                                                # Using 5 chunks and lamba_multi 0.8 to get decent breath of diversity in chunks
    custom_prompt = PromptTemplate(
            input_variables=["summaries", "question"],
            template = """
                    You are an IRS support assistant specialized in Information Technology (IT) policy and documentation. 
                    Use the following document excerpts to answer the user’s question. 
                    Provide accurate, helpful, and concise answers in a clear and professional tone. 
                    Only use the information from the excerpts below. If the answer is not found, say so.           # Custom prompt template for LLM to answer questions using document excerpts and context from sources
                    Cite each source explicitly using (Source: [source]).

                    Document Excerpts:
                    {summaries}

                    User Question:
                    {question}

                    Answer:
                    """

            )  
                                                                # QAchain setup                       
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(             
            llm=llm,
            retriever=retriever,                
            chain_type="map_reduce",                             # Use "stuff", "map_reduce", or "refine" , more direct answer -> more indirect answer
            chain_type_kwargs={"combine_prompt": custom_prompt}, # Use "prompt" for "stuff", "combine prompt" for "map_reduce", and "question_prompt" or "refine_prompt" for "refine"
    )
  
    print("Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:       # While loop for runtime chatbot interaction
            print("Goodbye!")
            break
        
        result = qa_chain({"question": query})

        print("Answer:", result["answer"])
        print("Sources:", result["sources"])  # source
    

    
if __name__=="__main__":
   main()           


#vector_store._index.delete(delete_all=True)
#print (index_stats)
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
