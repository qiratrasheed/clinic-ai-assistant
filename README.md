# 🏥 Al-Shifa Medical Clinic — Multi-Agent AI Assistant

## Project Overview
A Multi-Agent AI Assistant built with LangGraph and Agentic RAG
for Al-Shifa Medical Clinic, Sahiwal, Pakistan.

## Tech Stack
- Python + OOP
- LangGraph (Agent Orchestration)
- Groq LLM (Llama 3.3)
- FAISS (Vector Store)
- HuggingFace Embeddings
- Streamlit (UI)

## Agents
| Agent | Role |
|-------|------|
| Router Agent | Classifies query type |
| RAG Agent | Answers from documents |
| Task Agent | Appointment & triage |
| General Agent | General chat |
| Evaluator Agent | Verifies response |

## Design Patterns Used
1. Router Pattern
2. Prompt Chaining
3. Evaluator Pattern

## Setup
```bash
git clone <your-repo>
cd my-ai-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys in .env
streamlit run app.py
```

## Project Structure
```
my-ai-assistant/
├── app.py
├── main.py
├── graph.py
├── agents/
├── tools/
└── knowledge_base/
```