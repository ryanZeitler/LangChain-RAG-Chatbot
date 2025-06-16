# LangChain-RAG-Chatbot

- Description - 

A Data Wrangler + Retrieval-Augmented Generation (RAG) Chatbot that scims links from an IRS Internal Revenue Manuels on Information Technology Tabel of Contents

- Features - 

Vector search with Pinecone
Embeds documents
Summarize and answer user queries
Cite source documents
Interactive terminal chat
MMR and similarity search

- Dependencies - 

Located within ImportantRequirements.txt
    pip install -r ImportRequirements.txt

- ENV Variables -
Store in .env file 

[OPENAI_API_KEY= , PINECONE_API_KEY= , PINECONE_INDEX_NAME= USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36]

- Structure - 

chatbotMain.py
bucket/
    -chunks.json
    -docs.jaon
    links.json
.env
ImportRequirements.txt
README.md










