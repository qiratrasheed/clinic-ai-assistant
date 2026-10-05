class BaseAgent:
    def __init__(self, llm, name: str):
        self.llm = llm
        self.name = name

    def run(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name} must implement run()")