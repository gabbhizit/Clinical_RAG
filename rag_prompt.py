import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 400))

PROMPT_TEMPLATE = """You are a helpful clinical booking agent. Use ONLY the CONTEXT below to answer questions.
Your goal is to help users understand our services and guide them to book appointments.

When users ask about services, hours, or policies, answer helpfully.
When appropriate, suggest booking an appointment and provide booking instructions.
Be friendly, conversational, and proactive about scheduling.

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

class AppointmentTracker:
    def __init__(self):
        self.user_intent = None
        self.service_type = None
        self.booking_started = False

tracker = AppointmentTracker()

def make_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": int(os.getenv("TOP_K", 4))})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )
    return chain

def answer_with_rag(chain, question: str) -> str:
    res = chain.invoke(question)
    answer = res.strip()
    
    keywords = ["book", "appointment", "schedule", "when", "available", "time"]
    if any(kw in question.lower() for kw in keywords):
        tracker.booking_started = True
    
    if tracker.booking_started and "book" not in answer.lower():
        answer += "\n\n📅 Ready to book? Visit our website or call +1-555-987-0000 to schedule your appointment!"
    
    return answer
