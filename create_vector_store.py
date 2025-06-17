import os

os.environ['HF_HOME'] = "hf_cache"

from datasets import load_dataset

lectures = load_dataset("enesxgrahovac/the-feynman-lectures-on-physics")

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_text_chunks_langchain(example):
    text = example['section_text']
    chapter_title = example['chapter_title']
    section_title = example['section_title']
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for chunk in text_splitter.split_text(text):
        new_doc = Document(
            page_content=chunk,
            metadata={
                'chapter_title': chapter_title,
                'section_title': section_title
            }
        )
        docs.append(new_doc)
    return docs

def get_chunks_for_dataset(dataset):
    chunks = []
    for example in tqdm(dataset, desc="Processing examples", total=len(dataset)):
        chunks.extend(get_text_chunks_langchain(example))
    return chunks

docs = get_chunks_for_dataset(lectures['train'])


model_name = "intfloat/multilingual-e5-large"

hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

chroma_data_path = "chroma_data"

print("Starting")
vectorstore = Chroma.from_documents(
    docs , hf_embeddings, persist_directory= chroma_data_path
)
print("Done")

