# MediChat: Advanced Clinical RAG Chatbot

## Overview
MediChat is a production-grade Clinical Retrieval-Augmented Generation (RAG) system designed to provide accurate, grounded, and safe medical information. By leveraging local Large Language Models and high-performance vector stores, it processes clinical queries with state-of-the-art accuracy while maintaining strict data privacy through local execution.

## Technical Stack
- **Python**: Primary development language.
- **Streamlit**: Framework used for the interactive web frontend with a premium clinical aesthetic.
- **FastAPI**: High-performance backend API to handle RAG requests and session management.
- **LangChain**: Orchestration framework for building and managing the robust RAG pipeline.
- **FAISS**: Vector database used for ultra-fast and efficient similarity search.
- **Sentence Transformers**: Used to generate high-quality clinical text embeddings (`all-MiniLM-L6-v2`).
- **Ollama**: Local model runner providing privacy-focused LLM capabilities (default: `llama3.2:3b`).
- **Hugging Face Datasets**: Source for clinical knowledge base and benchmark evaluation data.
- **BERTScore / ROUGE**: Libraries used for rigorous quantitative NLP evaluation of model responses.

## Authentication and Security
The application implements a multi-layer security protocol to ensure data integrity and user safety:
- **Secure Login**: Users must authenticate using encrypted credentials.
- **User Registration**: New accounts are protected with mandatory password verification and hashing.
- **Medical Consent**: A blocking consent flow requires users to acknowledge the AI's limitations before access.
- **Granular Access Control**: Chat features are locked behind the authentication and consent layers.
- **Local Data Isolation**: All session data, credentials, and logs are stored exclusively on the user's infrastructure.

## Key Features
- **Local Inference**: Powered by Ollama for maximum privacy and speed.
- **Robust RAG Pipeline**:
  - **Dynamic Fallback**: Automatically scales retrieval depth if initial searches are insufficient.
  - **Semantic Diversity**: Filters redundant context to maximize information density.
  - **Re-ranking**: Multi-facet semantic re-ranking for clinical relevance.
- **Hallucination Control**: Integrated n-gram grounding validator that strips unsupported claims from LLM responses.
- **Context-Aware Conversational Logic**:
  - **Pronoun Resolution**: Resolves ambiguous terms (e.g., "it", "this") based on conversation history.
  - **Topic Shift Detection**: Detects when a user changes subjects to maintain context integrity.
- **Premium UI/UX**: An Apple-style dashboard with independent scrolling, sticky headers, and responsive layout.
- **Comprehensive Evaluation**: Automated benchmarking against clinical datasets using BERTScore and ROUGE.

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
In the project folder, open a terminal and run:
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

### 6. Restart & Refresh (Recommended)
If you have just finished the environment setup, it is highly recommended to close your current terminal windows and open fresh ones. This ensures that all environment variables and paths are correctly initialized.

### 7. Prepare the Medical Data (Ingestion)
Before the AI can answer medical questions, it needs to process the clinical knowledge base:
```bash
python main.py --ingest
```
*Note: This may take 2-5 minutes depending on your computer's speed.*

### 8. Start the Chatbot
First, ensure you are in the same project folder. You must run the **Backend** and **Frontend** at the same time:

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

### 9. Start Chatting
Your web browser should automatically open to the MediChat interface.
- **Log In**: Use `admin` (username) and `admin123` (password).
- **Consent**: Review and accept the medical disclaimer.
- **Ask**: Try asking "What are the symptoms of appendicitis?"

### 10. Troubleshooting Tips
- **Python not found**: Re-install Python and make sure "Add to PATH" is checked.
- **Connection Refused**: Ensure Ollama is running in your system tray and the Backend terminal (Step 8.1) is active.
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

## Ethics and Privacy
- **Data Privacy**: No personal or health-identifiable information is stored or transmitted.
- **Local Sovereignty**: Model execution happens entirely within the user's infrastructure.
- **Transparency**: The mandatory consent flow informs users of the AI-driven nature of the project.
- **Validation**: Input length checks and rate limits are enforced to maintain system integrity.

## Potential Roadmap
- **Enhanced Datasets**: Incorporating more diverse medical journals and textbooks.
- **Advanced Metrics**: Adding human evaluation frameworks for better qualitative analysis.
- **Deployment Scaling**: Providing Docker configurations for enterprise-level hosting.
- **Advanced Auth**: Support for multi-factor authentication and role-based access.

## Conclusion
This Clinical RAG Chatbot offers a secure and reliable framework for medical knowledge retrieval. By combining local LLM capabilities with a grounded retrieval pipeline, it demonstrates a practical solution for health-related information management while ensuring data privacy.

## Medical Disclaimer
MediChat is an AI research project and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

---
**Developed with focus on clinical safety and UX excellence.**
