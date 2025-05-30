import dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_doc(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_doc(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def create_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    from langchain_core.vectorstores import InMemoryVectorStore
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

def add_docs(vector_store, all_splits):
    ids = vector_store.add_documents(documents=all_splits)


def get_response(vector_store, query, k = 3):
    query = "what is a Recommendation System?"
    results = vector_store.similarity_search_with_score(query, k=k)
    for i in range(k):
        doc, score = results[i]
        print("-" * 20)
        print(f"Score: {score}\n")
        print(doc.page_content)


file_path = "basics-of-data-science-kpk.pdf"
def run_rag(query):
    docs = load_doc(file_path)
    all_splits = split_doc(docs)
    vector_store = create_vector_store()
    add_docs(vector_store, all_splits)
    get_response(vector_store, query)

if __name__ == "__main__":
    run_rag(query="what is a Recommendation System?")