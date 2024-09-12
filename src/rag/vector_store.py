from langchain_community.vectorstores import Chroma
from ..model_config import get_embeddings

def create_vectorstore(documents):
    embeddings = get_embeddings()
    return Chroma.from_documents(documents, embeddings)