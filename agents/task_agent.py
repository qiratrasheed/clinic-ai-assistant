from agents.base_agent import BaseAgent
from agents.language_utils import enforce_english

class TaskAgent(BaseAgent):
    def run(self, state: dict) -> dict:
        query = state["user_query"]

        prompt = f"""You are an appointment and triage assistant 
for Al-Shifa Medical Clinic, Sahiwal.

You handle:
1. Appointment scheduling -> Tell patient to call 0300-1234567 
   or visit clinic 9AM-5PM
2. Symptom triage -> Based on symptoms, suggest which doctor to see:
   - Fever/Flu/General -> Dr. Ahmed Khan
   - Heart/BP issues  -> Dr. Bilal Hussain  
   - Women health     -> Dr. Ayesha Malik
   - Children issues  -> Dr. Sara Ahmed
3. Urgent cases -> Tell to come immediately or call emergency

The patient query has already been translated to English.
Be polite. Always answer in English only.

Patient Query: {query}

Response:"""

        response = self.llm.invoke(prompt)
        state["agent_response"] = enforce_english(self.llm, response.content)
        print("[Task Agent] Task handled!")
        return state
