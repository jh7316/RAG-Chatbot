import re
from io import BytesIO
from typing import Tuple, List
import pickle

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import faiss


def get_vectorstore(pdf_file, pdf_name, openai_api_key):
    #parse pdf
    pdf = PdfReader(BytesIO(pdf_file),pdf_name)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)

    #create document chunks
    page_docs = [Document(page_content=page) for page in output]
    doc_chunks = []
    for p,doc in enumerate(page_docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": p+1, "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = pdf_name
            doc_chunks.append(doc)
    
    
    #create index
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index = FAISS.from_documents(doc_chunks, embeddings)
    return index


def augment_prompt(query,vectorstore):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.
    
    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt
