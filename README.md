# MediChat: Advanced Clinical RAG Chatbot

MediChat is a production-grade Clinical Retrieval-Augmented Generation (RAG) system designed to provide accurate, grounded, and safe medical information. It leverages local Large Language Models (via Ollama) and a high-performance vector store (FAISS) to process clinical queries with state-of-the-art accuracy.

## Key Features

- **Local Inference**: Powered by Ollama (default: `llama3.2:3b`) for privacy and speed.
- **Robust RAG Pipeline**:
  - **Dynamic Fallback**: Automatically scales retrieval depth if initial searches are insufficient.
  - **Semantic Diversity**: Filters redundant context to maximize information density.
  - **Re-ranking**: Multi-facet semantic re-ranking for clinical relevance.
- **Hallucination Control**: Integrated n-gram grounding validator that strips unsupported claims from LLM responses.
- **Context-Aware Conversational Logic**:
  - **Pronoun Resolution**: Resolves ambiguous terms (e.g., "it", "this") based on conversation history.
  - **Topic Shift Detection**: Detects when a user changes subjects to maintain context integrity.
- **Professional UI**: A premium, Apple-style Streamlit dashboard with:
  - Secure hashed-password authentication.
  - Mandatory medical consent workflow.
  - Persistent chat history and search.
- **Comprehensive Evaluation**: Built-in tools for measuring accuracy using **BERTScore** and **ROUGE**.

## How to Run: Full Step-by-Step Guide

Follow these steps in order to set up and run the Clinical Chatbot on your machine.

### 1. Install the Essentials
- **Python**: Download and install from [python.org](https://www.python.org/downloads/). 
  - **Important**: During installation, ensure you check the box **"Add Python to PATH"**.
- **Ollama**: Download and install from [ollama.com](https://ollama.com/). Once installed, open the app and let it run in the background.

### 2. Prepare the AI Model
Open your Command Prompt (cmd) or Terminal and run:
```bash
ollama pull llama3.2:3b
```
This downloads the language model that powers the chatbot. Keep your internet connected until it reaches 100%.

### 3. Get the Code
Download the project files to your computer:
```bash
git clone https://github.com/Harshit-bhardwaj08/Clinical-Chatbot.git

cd Clinical-Chatbot
```

### 4. Setup the Project Environment
In the `medichat` folder, open a terminal and run:
1. **Create a workspace**: `python -m venv venv`
2. **Activate the workspace**:
   - **Windows**: `.\venv\Scripts\activate`
   - **Mac/Linux**: `source venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`

### 5. Configuration (Optional)
The system works with default settings, but you can customize it by creating a `.env` file in the root directory:
```env
OLLAMA_MODEL=llama3.2:3b
DEBUG_MODE=false
```

### 6. Prepare the Medical Data (Ingestion)
Before the AI can answer medical questions, it needs to process the clinical knowledge base:
```bash
python main.py --ingest
```
*Note: This may take 2-5 minutes depending on your computer's speed.*

### 7. Start the Chatbot
You must run the **Backend** and **Frontend** at the same time:

1. **Terminal 1 (Backend)**:
```bash
uvicorn app.api_server:app --port 8000
```
   *Keep this window open! This is the engine of the chatbot.*

2. **Terminal 2 (Frontend)**:
   - Open a **new** terminal window in the same folder.
   - Activate the workspace: `.\venv\Scripts\activate` (or `source venv/bin/activate`).
   - Run the interface:
```bash
streamlit run app/streamlit_app.py
```

### 8. Start Chatting
Your web browser should automatically open to the MediChat interface.
- **Log In**: Use `admin` (username) and `admin123` (password).
- **Consent**: Review and accept the medical disclaimer.
- **Ask**: Try asking "What are the symptoms of appendicitis?"

### 9. Troubleshooting Tips
- **Python not found**: Re-install Python and make sure "Add to PATH" is checked.
- **Connection Refused**: Ensure Ollama is running in your system tray and the Backend terminal (Step 7.1) is active.
- **Module not found**: Ensure you have activated the workspace (`venv`) in every new terminal window you open.

## Evaluation & Testing
Evaluate the RAG system's performance against clinical benchmarks:
```bash
python evaluation.py
```
Run the unit test suite:
```bash
pytest tests/
```

## Project Structure

```text
├── app/                  # Streamlit frontend & assets
├── src/                  # Core RAG logic & backend services
│   ├── rag_chain.py      # The Robust RAG Pipeline (v3)
│   ├── vector_store.py   # FAISS management
│   ├── auth.py           # Secure authentication
│   └── ...
├── data/                 # Local vector store & chat history
├── tests/                # Unit & integration tests
├── main.py               # Application entry point
├── evaluation.py         # RAG accuracy metrics
└── requirements.txt      # Project dependencies
```

## Medical Disclaimer
MediChat is an AI research project and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

---
**Developed with focus on clinical safety and UX excellence.**
