from dotenv import load_dotenv
load_dotenv()

from embed_retriever import build_or_load_index
from rag_prompt import make_rag_chain, answer_with_rag

def main():
    vectorstore = build_or_load_index(recreate=False)
    chain = make_rag_chain(vectorstore)

    print("Clinical RAG (LangChain + FAISS + OpenAI). Type 'exit' to quit.")
    while True:
        q = input("\nAsk: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("Searching...")
        try:
            ans = answer_with_rag(chain, q)
            print("\nAnswer:", ans)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
