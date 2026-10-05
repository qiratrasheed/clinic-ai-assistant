from agents.base_agent import BaseAgent
from agents.language_utils import enforce_english

class GeneralAgent(BaseAgent):
    def run(self, state: dict) -> dict:
        query = state["user_query"]
        
        prompt = f"""You are a helpful AI assistant.
Answer this question clearly and politely in English only.
Do not use Urdu or Roman Urdu words.
Question: {query}"""

        response = self.llm.invoke(prompt)
        state["agent_response"] = enforce_english(self.llm, response.content)
        return state
