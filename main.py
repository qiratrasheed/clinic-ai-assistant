import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.router_agent import RouterAgent
from agents.general_agent import GeneralAgent
from agents.rag_agent import RAGAgent
from agents.task_agent import TaskAgent
from agents.evaluator_agent import EvaluatorAgent
from tools.rag_tool import build_vector_store
from graph import create_graph

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

if groq_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
    print("🤖 LLM: Groq loaded!")
elif gemini_key:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_key)
    print("🤖 LLM: Gemini loaded!")
else:
    raise RuntimeError("Koi API key nahi mili!")

print("📚 Knowledge base load ho rahi hai...")
vectorstore = build_vector_store("knowledge_base")

router    = RouterAgent(llm, "Router")
rag       = RAGAgent(llm, "RAG", vectorstore)
task      = TaskAgent(llm, "Task")
general   = GeneralAgent(llm, "General")
evaluator = EvaluatorAgent(llm, "Evaluator")

app = create_graph(router, rag, task, general, evaluator)
print("\n✅ Al-Shifa Medical Clinic Assistant Ready!")
print("=" * 50)

try:
    print("\n📊 Graph Structure:")
    print(app.get_graph().draw_mermaid())
except:
    pass

while True:
    query = input("\n🏥 Patient ka sawal: ")
    if query.lower() in ["exit", "quit"]:
        print("Khuda Hafiz!")
        break

    state = {
        "user_query": query,
        "query_type": "",
        "retrieved_context": "",
        "agent_response": "",
        "evaluation_result": "",
        "final_response": "",
        "messages": []
    }

    result = app.invoke(state)

    print(f"\n💬 Assistant: {result['final_response']}")
    print("-" * 50)