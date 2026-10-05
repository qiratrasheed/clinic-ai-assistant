# Project Report
## 1. Business Scenario

Al-Shifa Medical Clinic is a healthcare facility located in 
Sahiwal, Punjab, Pakistan. The clinic needed an intelligent 
AI assistant that could handle patient queries 24/7 without 
requiring human staff for basic information requests.

**Problems Solved:**
- Patients calling after hours for basic information
- Staff time wasted on repetitive FAQ questions
- Difficulty routing patients to correct doctors
- No automated appointment guidance system

---

## 2. System Architecture

The system uses a multi-agent architecture orchestrated 
by LangGraph with the following flow:
### LangGraph State
```python
class AgentState(TypedDict):
    user_query: str
    query_type: str
    retrieved_context: str
    agent_response: str
    evaluation_result: str
    final_response: str
    messages: List[dict]
```

### Graph Design
- **Entry Point:** Router Agent
- **Conditional Edges:** Router → RAG / Task / General
- **All agents → Evaluator → END**

---

## 3. Agents Description

### Router Agent
Classifies incoming patient queries into three categories
using Llama 3.3 LLM. Uses keyword and semantic understanding
to route correctly.

### RAG Agent
Implements Agentic RAG with three steps:
1. Query Rewrite — improves search quality
2. Document Retrieval — FAISS similarity search
3. Answer Generation — grounded in clinic documents

### Task Agent
Handles appointment scheduling guidance and symptom
triage. Routes patients to correct specialist based
on symptoms described.

### General Agent
Handles greetings, general conversation, and queries
not related to clinic operations.

### Evaluator Agent
Quality control agent that checks every response for:
- Relevance to the question
- Professional and polite tone
- Medical accuracy
- Completeness

---

## 4. Design Patterns Used

### Pattern 1: Router
The Router Agent classifies queries and sends them to
the most appropriate specialized agent using conditional
edges in LangGraph.

### Pattern 2: Prompt Chaining
The RAG Agent implements a 3-step chain:
Query → Rewrite → Retrieve → Generate
Each step feeds into the next.

### Pattern 3: Evaluator
Every response passes through the Evaluator Agent before
reaching the user. If evaluation fails, a fallback
response is provided.

---

## 5. Agentic RAG Implementation

**Vector Store:** FAISS (local, free)
**Embeddings:** HuggingFace all-MiniLM-L6-v2
**Documents:** 4 clinic documents (txt format)
**Chunk Size:** 500 tokens with 50 overlap
**Retrieval:** Top 4 similar chunks

**RAG Pipeline:**
1. Patient asks question
2. Query rewritten for better search
3. FAISS retrieves top 4 relevant chunks
4. LLM generates answer from context only

---

## 6. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Gemini API quota exceeded | Switched to Groq free tier |
| Embedding model not found | Used HuggingFace local embeddings |
| LangGraph import errors | Updated to latest langgraph version |
| Model decommissioned | Updated to llama-3.3-70b-versatile |

---

## 7. Lessons Learned

1. **API Rate Limits** — Always have backup LLM providers
2. **LangGraph State** — Plan state carefully before building
3. **Prompt Engineering** — Clear instructions = better results
4. **OOP Structure** — Base class makes adding agents easy
5. **Testing** — Test each agent individually before wiring

---

## 8. Future Improvements

- Add conversation memory (multi-turn)
- Deploy on Streamlit Cloud
- Add more document types (PDF, CSV)
- Human-in-the-loop for critical decisions
- WhatsApp integration for real patients

---

## 9. Conclusion

This project successfully demonstrates a production-level
multi-agent AI system using LangGraph, Agentic RAG, and
modern LLM integration. The system can handle real patient
queries for a medical clinic with proper routing, retrieval,
and quality verification.

