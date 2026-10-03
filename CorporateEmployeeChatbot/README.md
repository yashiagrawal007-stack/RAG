## Corporate Employee Handbook Q&A Chatbot (RAG Pipeline)
This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and query corporate PDF documentation (specifically tailored in this instance for a corporate employee handbook). Built using the LangChain ecosystem, the application uses Hugging Face for vector embeddings, Pinecone for high-performance vector storage, and Groq to power accurate, context-aware LLM answers.
------------------------------
## Features

* PDF Document Loading: Automatically reads and parses large PDF handbooks using PyPDFLoader.
* Intelligent Text Chunking: Breaks data down with RecursiveCharacterTextSplitter using a sliding window strategy (500 chunk size, 20 chunk overlap) to optimize text context stability.
* Open-Source Embeddings: Computes highly accurate text embeddings locally using the Hugging Face sentence-transformers/all-MiniLM-L6-v2 model.
* Cloud Vector Search: Uploads and queries dense representations inside a Serverless Pinecone index (corporateemployee-chatbot) with Cosine distance metric configuration.
* Blazing Fast Inference: Interfaces with a Groq Hosted LLM to answer employee queries instantly based strictly on retrieved contextual blocks.

------------------------------
## Architecture Overview

[ PDF Document ] ──> [ PyPDFLoader ] ──> [ Recursive Character Splitting ]
                                                         │
                                                         ▼
[ Pinecone Vector Store ] <── [ Hugging Face Embeddings ] ◄─┘
         │
         ├── (Similarity Search via Retriever)
         ▼
[ Context Blocks ] ──┐
                     ▼
[ User Inquiry ] ──> [ ChatPromptTemplate ] ──> [ Groq LLM Inference ] ──> [ Final Answer ]

------------------------------
## Prerequisites & Installation## 1. Clone the Workspace
Ensure your environment contains the file setup. If using an interactive development environment like Google Colab, you can upload the notebook file natively.
## 2. Install Project Dependencies
Run the installation block to set up the runtime framework:

pip install -q \
    langchain \
    langchain-community \
    langchain-core \
    langchain-text-splitters \
    langchain-huggingface \
    langchain-pinecone \
    langchain-groq \
    pinecone \
    sentence-transformers \
    python-dotenv \
    pypdf \
    langgraph \
    langgraph-prebuilt \
    requests==2.32.4

------------------------------
## Configuration & Environment Setup
Before running the workflow pipelines, you must obtain and map out api credentials for cloud service providers. Set up your API workspace variables as follows:

import os

os.environ["GROQ_API_KEY"] = "your_groq_api_credential_here"
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_credential_here"

------------------------------
## Detailed Execution Steps## 1. Ingesting and Processing Documents
The system verifies the local existence of the source PDF, parses pages, and chops strings up cleanly:

from langchain_community.document_loaders import PyPDFLoaderfrom langchain_text_splitters import RecursiveCharacterTextSplitter
# Loadloader = PyPDFLoader("DKT-Employee-Handbook-12.23.pdf")extracted_data = loader.load()
# Splittext_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)text_chunks = text_splitter.split_documents(extracted_data)

## 2. Setting Up Vector Space
Downloads embedding weights from Hugging Face and connects directly with Pinecone to prepare a serverless cloud search index:

from pinecone import Pinecone, ServerlessSpecfrom langchain_pinecone import PineconeVectorStorefrom langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])index_name = "corporateemployee-chatbot"
# Creates database vector index space if it doesn't already existif index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

## 3. Setting Up the RAG Chain & Inference
Constructs a deterministic template prompt, attaches similarity metrics to a context retriever (k=3), and passes information downstream into the generative Large Language Model layer:

from langchain_groq import ChatGroqfrom langchain_core.prompts import ChatPromptTemplatefrom langchain_core.runnables import RunnablePassthroughfrom langchain_core.output_parsers import StrOutputParser
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Example Queryresponse = rag_chain.invoke("What are the benefits provided to employee")
print(response)

------------------------------
## License
This software utility code is provided "as-is" for enterprise exploration and prototyping frameworks. Feel free to modify chunk dimensions or upgrade base context LLM parameters to support custom workflows.
------------------------------


