from langgraph.graph import StateGraph, END
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from ..model_config import get_llm

def create_rag_graph(retriever):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following pieces of retrieved context to answer the user's question. If you don't know the answer, just say that you don't know."),
        ("human", "Context: {context}\n\nQuestion: {query}"),
    ])

    def retrieve(state):
        state["context"] = retriever.get_relevant_documents(state["query"])
        return state

    def generate_answer(state):
        response = prompt | llm | StrOutputParser()
        context = "\n".join([doc.page_content for doc in state["context"]])
        answer = response.invoke({"context": context, "query": state["query"]})
        state["answer"] = answer
        return state

    workflow = StateGraph()

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    rag_graph = workflow.compile()
    return rag_graph