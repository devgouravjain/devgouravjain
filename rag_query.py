from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# -------- Load & index docs --------
def load_vector_store():
    loaders = [
        TextLoader("docs/product_catalog_en.txt.txt", encoding="utf-8"),
        TextLoader("docs/product_catalog_ar.txt.txt", encoding="utf-8"),
        TextLoader("docs/quotation_policy.txt.txt", encoding="utf-8"),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


# -------- RAG Query --------
def answer_question(db, question):
    docs = db.similarity_search(question, k=3)

    if not docs:
        return "Not available in the provided documents.", []

    answer = []
    sources = set()

    for d in docs:
        answer.append(d.page_content)
        sources.add(d.metadata.get("source", "unknown"))

    final_answer = "\n\n".join(answer[:2])

    return final_answer, list(sources)


# -------- CLI --------
if __name__ == "__main__":
    db = load_vector_store()

    while True:
        question = input("\nAsk your question (AR/EN): ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        answer, sources = answer_question(db, question)

        print("\nAnswer (based on documents):\n")
        print(answer)

        if sources:
            print("\n--- Sources ---")
            for s in sources:
                print(f"- {s}")