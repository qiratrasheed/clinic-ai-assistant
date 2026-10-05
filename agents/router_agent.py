from agents.base_agent import BaseAgent

class RouterAgent(BaseAgent):
    def run(self, state: dict) -> dict:
        query = state["user_query"]

        prompt = f"""You are a router for Al-Shifa Medical Clinic.
Classify the query into exactly one word.

- 'rag'     → doctor info, clinic timing, fees, services, 
               treatment info, policies, FAQs
- 'task'    → appointment booking, symptoms, which doctor to see,
               emergency, triage
- 'general' → greetings, thank you, unrelated chat

Query: {query}
Answer (one word only - rag or task or general):"""

        response = self.llm.invoke(prompt)
        query_type = response.content.strip().lower()

        if query_type not in ["rag", "task", "general"]:
            query_type = "general"

        state["query_type"] = query_type
        print(f"[Router] → {query_type.upper()}")
        return state