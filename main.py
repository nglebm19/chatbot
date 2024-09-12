from flask import Flask, render_template, request, jsonify
from src import load_documents, create_vectorstore, get_retriever, create_rag_graph

app = Flask(__name__)

# Initialize RAG components
documents = load_documents("data/documents")
vectorstore = create_vectorstore(documents)
retriever = get_retriever(vectorstore)
rag_graph = create_rag_graph(retriever)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = rag_graph.invoke({"query": user_message})
    return jsonify({"response": response})

def create_app():
    return app

if __name__ == "__main__":
    app.run(debug=True)