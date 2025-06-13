import os
import getpass
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
#from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()
os.environ["USER_AGENT"]
#if not os.getenv("OPENAI_API_KEY"):
  #os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


openai_api = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API_KEY")                    #init .env objects]
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
user_agent = os.getenv("USER_AGENT")

pinecone_client = Pinecone(pinecone_api)  #init client object https://docs.pinecone.io/reference/api/authentication?utm_source=chatgpt.com

if pinecone_index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
index = pinecone_client.Index(pinecone_index_name) #use for later upsert https://docs.pinecone.io/reference/api/2024-07/data-plane/upsert?utm_source=chatgpt.com

embeddings = OpenAIEmbeddings()  #init open ai embeddings, no arg = default embeddings from langchain and auto pull api key
                                 #OpenAIEmbeddings(model: str = "text-embedding-3-small",organization: Optional[str] = None,dimensions: Optional[int] = None,request_timeout: Optional[float] = None,
                                 #                 chunk_size: int = 1000,max_retries: int = 6,headers: Optional[Dict[str, str]] = None,show_progress_bar: bool = False)


url = "https://www.irs.gov/irm/part2"
response = requests.get(url)  #GET request 
soup = BeautifulSoup(response.content, "lxml")  #lxml recommended: faster parser and better for larger and possibly non-perfect formats https://www.crummy.com/software/BeautifulSoup/bs4/doc/#differences-between-parsers 
headers = {"User-Agent": user_agent}

doc_links = []

for div in soup.find_all('div', class_ = 'region region-content'):
    for a_tag in div.find_all("a"):
        href = a_tag.get("href")
        doc_links.append(href)
    
loader= WebBaseLoader(web_paths=(doc_links),header_template=headers)
docs= loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print(chunks)







#if __name__=="__main__":
  #  main()