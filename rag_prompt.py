import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 400))

PROMPT_TEMPLATE = """You are a concise clinical assistant. Use ONLY the CONTEXT below to answer the user's question.
If the answer is not present in the CONTEXT, reply exactly: "I do not have this information."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

chat = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

def make_rag_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": int(os.getenv("TOP_K", 4))}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

def answer_with_rag(chain, question: str) -> str:
    res = chain.invoke({"query": question})
    return res.get("result", "").strip()
