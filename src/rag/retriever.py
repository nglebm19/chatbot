from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from ..model_config import get_llm

def get_retriever(vectorstore):
    base_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = load_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)