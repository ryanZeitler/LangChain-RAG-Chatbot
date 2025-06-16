import time
import os
import getpass
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
#if not os.getenv("OPENAI_API_KEY"):
  #os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

def env_access():
    openai_api = os.getenv("OPENAI_API_KEY")
    pinecone_api = os.getenv("PINECONE_API_KEY")                    #init .env objects]
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    user_agent = os.getenv("USER_AGENT")
  
    return openai_api, pinecone_api, pinecone_index_name, user_agent

def bucket_access(bucket_name):
    bucket_name = "aicanbucket"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
  
    return bucket


def init_pinecone(pinecone_api, pinecone_index_name):
    pinecone_client = Pinecone(pinecone_api)  #init client object https://docs.pinecone.io/reference/api/authentication?utm_source=chatgpt.com
    
    if pinecone_index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=pinecone_index_name,           #checks for a pinmecone index, if not makes one
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    index = pinecone_client.Index(pinecone_index_name) #use for later upsert https://docs.pinecone.io/reference/api/2024-07/data-plane/upsert?utm_source=chatgpt.com
    
    index_stats=index.describe_index_stats()
  
    return index, index_stats

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def doc_to_dict_(docs):
    dicts = []
    for doc in docs:
        add_dicts = {
                "page_content": doc.page_content,
                "metadata": doc.metadata 
            }
        dicts.append(add_dicts)
    return dicts
   
def dict_to_doc(dicts):
    docs = []
    for d in dicts:
        doc = Document(page_content=d.get("page_content",""), metadata=d.get("metadata", {}))
        docs.append(doc)
    return docs

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def save_load_to_bucket(bucket, blob_name, c_function, on_load=None, on_save=None):
    blob = bucket.blob(blob_name)
    
    if blob.exists():
        print(f"File {blob_name} loaded from {bucket}")
        blob_data = blob.download_as_text()
        json_data = json.loads(blob_data)
        if on_load:
            json_data = on_load(json_data)
        return json_data
  
    print(f"File {blob_name} not found in {bucket}. Creating new file.")
    create_data = c_function()
    if on_save:
        create_data = on_save(create_data)
        blob.upload_from_string(json.dumps(create_data), content_type="application/json")
        print(f"saved {blob_name} to {bucket}")
        create_data = on_load(create_data)
    else:
        blob.upload_from_string(json.dumps(create_data), content_type="application/json")
        print(f"saved {blob_name} to {bucket}")
    return create_data 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def url_scrape(url):
    response = requests.get(url)  #GET request 
    soup = BeautifulSoup(response.content, "lxml")  #lxml recommended: faster parser and better for larger and possibly non-perfect formats https://www.crummy.com/software/BeautifulSoup/bs4/doc/#differences-between-parsers 
    doc_links = []

    for div in soup.find_all('div', class_ = 'region region-content'):
        for a_tag in div.find_all("a", href=True):
            href = a_tag.get("href")
            doc_links.append(href)
          
    return doc_links


def doc_loader(doc_links, user_agent):
  headers = {"User-Agent": user_agent}
  loader = WebBaseLoader(web_paths=(doc_links),header_template=headers)
  docs = loader.load()
    
  for doc in docs:
      # Replace multiple whitespace chars with a single space and strip leading/trailing spaces
      doc.page_content = ' '.join(doc.page_content.split())
        
  for doc in docs:
      doc.metadata = {"source": doc.metadata.get("source", "")}  #add source url to metadata for each document, removes the rest of metadata to keep size smaller
  
  return docs


def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
  
    return chunks

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def upsert_to_index(vector_store, chunks):
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        doc_id = [str(uuid4()) for _ in chunk_batch]
        try:
            vector_store.add_documents(documents=chunk_batch, ids=doc_id)
        except Exception as e:
            print(f"Error on batch {i//batch_size}: {e}")
        time.sleep(10.0)
            

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
  
    openai_api, pinecone_api, pinecone_index_name, user_agent = env_access()
    bucket = bucket_access("aicanbucket")
    pinecone_index, index_stats = init_pinecone(pinecone_api, pinecone_index_name)
    embeddings = OpenAIEmbeddings()  #init open ai embeddings, no arg = default embeddings from langchain and auto pull api key
                                 #OpenAIEmbeddings(model: str = "text-embedding-3-small",organization: Optional[str] = None,dimensions: Optional[int] = None,request_timeout: Optional[float] = None,
                                 #                 chunk_size: int = 1000,max_retries: int = 6,headers: Optional[Dict[str, str]] = None,show_progress_bar: bool = False)
    vector_store = PineconeVectorStore(pinecone_index, embeddings)  #init vector store using pinecone index and embeddings
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    url = "https://www.irs.gov/irm/part2"
  
    doc_links = save_load_to_bucket(bucket, "links.json", lambda: url_scrape(url))
  
    docs = save_load_to_bucket(bucket, "docs.json", lambda: doc_loader(doc_links, user_agent), on_load=dict_to_doc, on_save=doc_to_dict_)
  
    chunks = save_load_to_bucket(bucket, "chunks.json", lambda: split_docs(docs), on_load=dict_to_doc, on_save=doc_to_dict_)
  
    print (index_stats)
    print (len(chunks))
    
    
    if index_stats['total_vector_count'] == 0:  
        upsert_to_index(vector_store, chunks)
        
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8})
    
    custom_prompt = PromptTemplate(
            input_variables=["summaries", "question"],
            template = """
                    You are an IRS support assistant specialized in Information Technology (IT) policy and documentation. 
                    Use the following document excerpts to answer the user’s question. 
                    Provide accurate, helpful, and concise answers in a clear and professional tone. 
                    Only use the information from the excerpts below. If the answer is not found, say so.
                    Cite each source explicitly using (Source: [source]).

                    Document Excerpts:
                    {summaries}

                    User Question:
                    {question}

                    Answer:
                    """
            
            )
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="map_reduce",# Use "stuff", "map_reduce", or "refine" depending on needs
            chain_type_kwargs={"combine_prompt": custom_prompt},# Use "prompt" for "stuff", "combine prompt" for "map_reduce", and "question_prompt" or "refine_prompt" for "refine"
    )
  
    print("Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        result = qa_chain({"question": query})

        print("Answer:", result["answer"])
        print("Sources:", result["sources"])  # source
    
    
if __name__=="__main__":
   main()           


#vector_store._index.delete(delete_all=True)
#print (index_stats)
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
