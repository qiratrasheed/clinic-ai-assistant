from agents.base_agent import BaseAgent
from agents.language_utils import enforce_english
from tools.rag_tool import retrieve_context

class RAGAgent(BaseAgent):
    def __init__(self, llm, name: str, vectorstore):
        super().__init__(llm, name)
        self.vectorstore = vectorstore

    def run(self, state: dict) -> dict:
        query = state["user_query"]

        print("[RAG Agent] Searching documents for context...")

        # Step 1 - Retrieve from documents
        context = retrieve_context(self.vectorstore, query)
        state["retrieved_context"] = context

        # Step 2 - Generate answer
        answer_prompt = f"""You are a helpful medical clinic assistant for Al-Shifa Clinic, Sahiwal.

The patient query has already been translated to English.
Answer the patient's question using ONLY the information provided below.
- Be polite and clear
- Always answer in English only
- If the patient asks about services, prices, or fees, list the relevant service names with prices
- If the answer is not found in the context, say: "Please contact our clinic for this information: 0300-1234567"

Clinic Documents:
{context}

Patient Question: {query}

Answer:"""

        response = self.llm.invoke(answer_prompt)
        state["agent_response"] = enforce_english(self.llm, response.content)
        print("[RAG Agent] Answer ready!")
        return state
