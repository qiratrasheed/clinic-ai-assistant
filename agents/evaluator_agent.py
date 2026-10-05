from agents.base_agent import BaseAgent

class EvaluatorAgent(BaseAgent):
    def run(self, state: dict) -> dict:
        query = state["user_query"]
        response = state["agent_response"]

        prompt = f"""You are a quality checker for a medical clinic assistant.

Check if this response is:
1. Relevant to the patient question
2. Polite and professional
3. Not giving wrong medical advice
4. Complete answer

Patient Question: {query}
Assistant Response: {response}

Reply with ONLY one word:
- 'pass' if response is good
- 'fail' if response is poor"""

        result = self.llm.invoke(prompt)
        evaluation = result.content.strip().lower()

        if "pass" in evaluation:
            state["evaluation_result"] = "pass"
            state["final_response"] = response
        else:
            state["evaluation_result"] = "fail"
            state["final_response"] = (
                "Sorry, I was unable to answer your question properly. "
                "Please contact our clinic: 0300-1234567"
            )

        print(f"[Evaluator] Result: {state['evaluation_result']}")
        return state