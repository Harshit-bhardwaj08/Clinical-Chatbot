# Clinical RAG Chatbot

This project implements a local-first Retrieval-Augmented Generation (RAG) system specifically designed for clinical medical question answering. It focuses on providing reliable, grounded, and private access to medical information using open-source models.

---

## Overview

The Clinical RAG Chatbot is a medical assistant that uses Retrieval-Augmented Generation to ground its answers in verified clinical literature. By retrieving relevant data from a local vector database before generating a response, the system significantly reduces the risk of AI hallucinations and ensures that information is based on established medical facts.

The entire system runs locally via Ollama, ensuring that sensitive medical queries are processed on-device. This approach prioritizes data privacy and security, as no information is transmitted to external cloud providers.

---

## Key Features

- **Clinical Question Answering:** Provides detailed information on medical conditions, symptoms, and treatment options.
- **Grounded Responses:** Every answer is derived from specific retrieved documents to maintain high accuracy.
- **Confidence Scoring:** The system assesses and displays a confidence level for each response based on the quality of retrieved context.
- **Source Transparency:** Users can see the exact snippets of medical text used to build each answer.
- **Performance Evaluation:** Built-in tools to calculate BERTScore and ROUGE metrics for response validation.
- **Secure Onboarding:** A professional authentication system with a mandatory medical consent flow.
- **Modern Interface:** A clean, responsive chat UI developed with Streamlit.
- **High-Performance Backend:** A FastAPI-based server optimized for asynchronous query processing.
- **Privacy-First Design:** Complete local processing of all data and model inference.

---

## Technical Stack

- **Python:** Primary development language.
- **Streamlit:** Framework used for the interactive web frontend.
- **FastAPI:** Used to build the high-performance backend API.
- **LangChain:** Utilized for building and managing the RAG pipeline.
- **FAISS:** Vector database used for efficient similarity search.
- **Sentence Transformers:** Used to generate high-quality text embeddings.
- **Ollama:** Local model runner for Large Language Models (default: llama 3.2 3B).
- **Hugging Face Datasets:** Source for clinical training and retrieval data.
- **BERTScore / ROUGE:** Libraries for quantitative NLP evaluation.

---

## Project Structure

- `src/` : Contains core logic for the RAG pipeline, document retrieval, and configuration.
- `app/` : Houses the application layer, including the FastAPI server and the Streamlit UI.
- `data/` : Local storage for the FAISS index and clinical datasets.
- `tests/` : Integration and unit tests for system verification.
- `evaluation.py` : Script for calculating performance and quality metrics.
- `main.py` : CLI entry point for system checks, data ingestion, and testing.

---

## Setup Guide

Follow these steps to configure the environment on your local system:

### 1. Clone the Repository
```bash
git clone https://github.com/Harshit-bhardwaj08/Clinical-Chatbot
cd Project
```

### 2. Configure the Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a file named `.env` in the root directory with the following settings:
- `OLLAMA_MODEL`: Specify the model name (e.g., llama3.2:3b).
- `DEBUG_MODE`: Set to true for detailed pipeline logs.
- `MAX_QUERY_LENGTH`: Character limit for questions (default: 500).

### 5. Initialize Ollama
Download and run Ollama, then pull the required model:
```bash
ollama pull llama3.2:3b
```

### 6. Data Ingestion
Populate the vector database with clinical data:
```bash
python main.py --ingest
```

---

## Running the Project

### Using the Command Line
To run a single query through the terminal:
```bash
python main.py --query "What are common signs of hypertension?"
```

### Starting the Backend
Run the FastAPI server:
```bash
uvicorn app.api_server:app --host 0.0.0.0 --port 8000
```

### Starting the Frontend
Launch the web-based chat interface:
```bash
streamlit run app/streamlit_app.py
```

---

## System Evaluation

The project includes scripts to evaluate answer quality against a reference dataset.

- **Metrics:** Semantic similarity (BERTScore) and n-gram overlap (ROUGE).
- **Default Evaluation:**
  ```bash
  python evaluation.py
  ```
- **Custom Evaluation:**
  ```bash
  python evaluation.py --file data/your_eval_data.json
  ```

---

## Authentication and Security

The application follows a structured security protocol:
1. **Login:** Users must authenticate with a secure username and password.
2. **Registration:** New users can sign up with mandatory password verification.
3. **Medical Consent:** First-time users are required to accept a medical disclaimer.
4. **Access Control:** The chatbot interface is only accessible after consent is granted.
5. **Data Isolation:** All session data and authentication records remain on the local host.

---

## Limitations

- **Data Dependency:** The quality of information is directly linked to the provided medical datasets.
- **Evaluation Scope:** The built-in evaluation is a sanity check; large-scale testing requires more extensive datasets.
- **Medical Disclaimer:** This tool is for informational research only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## Ethics and Privacy

- **Data Privacy:** No personal or health-identifiable information is stored or transmitted.
- **Local Sovereignty:** Model execution happens entirely within the user's infrastructure.
- **Transparency:** The mandatory consent flow informs users of the AI-driven nature of the project.
- **Validation:** Input length checks and rate limits are enforced to maintain system integrity.

---

## Potential Roadmap

- **Enhanced Datasets:** Incorporating more diverse medical journals and textbooks.
- **Advanced Metrics:** Adding human evaluation frameworks for better qualitative analysis.
- **Deployment Scaling:** Providing Docker configurations for enterprise-level hosting.
- **Advanced Auth:** Support for multi-factor authentication and role-based access.

---

## Conclusion

This Clinical RAG Chatbot offers a secure and reliable framework for medical knowledge retrieval. By combining local LLM capabilities with a grounded retrieval pipeline, it demonstrates a practical solution for health-related information management while ensuring data privacy.
