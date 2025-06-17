import os

os.environ['HF_HOME'] = "hf_cache"

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


model_name = "intfloat/multilingual-e5-large"

hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)


chroma_data_path = "chroma_data"


vector_database_db = Chroma(
    persist_directory=chroma_data_path,
    embedding_function=hf_embeddings
)

retreiver = vector_database_db.as_retriever(k=5)

