# RAG Agent Project

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) Agent** using the `swarmauri` library and the Groq API. The agent combines a TF-IDF vector store for document retrieval with a conversational AI model to provide intelligent responses based on a curated set of documents about the developer, Kosisochukwu. The project includes multiple files for development, testing, and deployment:

- **`AgentKosiV1.py`**: Initial Python script for setting up the RAG agent with basic document handling and query processing.
- **`AgentKosiV2.py`**: Enhanced version with support for large documents via chunking, an interactive query loop for local testing, and a FastAPI interface for deployment.
- **`RAG_Agent.ipynb`**: Jupyter Notebook providing a step-by-step exploration of the RAG agent's implementation, from document ingestion to LLM integration.
- **`RAG_Agent.py`**: Early standalone script consolidating notebook functionality.

The RAG agent is designed to answer queries about me, my education, hobbies, and more, leveraging both document-based data and the Groq model's general knowledge when needed.

## Features

- **Document Management**: Uses `TfidfVectorStore` to store and retrieve documents, with chunking support for large texts.
- **Conversational AI**: Integrates the Groq API (`llama3-8b-8192` model) for natural language responses.
- **Interactive Querying**: Supports a local loop for continuous questioning and a FastAPI endpoint for web access.
- **Scalability**: Handles large documents about the developer, preserving context with overlapping chunks.
- **Customizable Responses**: Configured to use general knowledge when document data is unavailable, avoiding "not found" messages.

## Prerequisites

To run or deploy this project, you need:

- **Python**: Version 3.8 or higher.
- **Dependencies**:
  - `swarmauri==0.4.1`
  - `python-dotenv==1.0.1`
  - `groq==0.25.0`
  - `fastapi==0.111.0`
  - `uvicorn==0.29.0`
  - `gunicorn==22.0.0`
  - Additional libraries (e.g., `scikit-learn`, `joblib`) for vector store functionality.
- **Groq API Key**: Obtain from [console.groq.com](https://console.groq.com).
- **Docker**: For containerized deployment on Northflank.
- **Jupyter Notebook** (optional): For running `RAG_Agent.ipynb`.

Install dependencies locally with:
```bash
pip install -r requirements.txt
```

## Setup

### Local Development

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ksschkw/rag-agent-project.git
   cd rag-agent-project
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your Groq API key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - Do **not** commit this file to Git (add it to `.gitignore`—see below).

3. **Install Dependencies**:
   Run:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure your virtual environment (e.g., `rag_env`) is activated if using one.

4. **Prepare Documents**:
   - Documents are embedded in the code (e.g., `documents` list in `AgentKosiV2.py`).
   - Modify these lists or add large documents using the `chunk_document` function in `AgentKosiV2.py`.

### Virtual Environment Recommendation

- Use a virtual environment (e.g., `rag_env`) to isolate dependencies:
  ```bash
  python -m venv rag_env
  source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
  pip install -r requirements.txt
  ```
- Add `rag_env/` to `.gitignore` to avoid committing it (see below).

## Usage

### Running Locally

#### Jupyter Notebook (`RAG_Agent.ipynb`)
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `RAG_Agent.ipynb` and execute cells sequentially to:
   - Initialize the vector store with sample documents.
   - Configure conversation context and integrate the Groq API.
   - Test with sample queries.

#### Python Script (`AgentKosiV1.py` or `AgentKosiV2.py`)
1. **Interactive Mode (AgentKosiV2.py)**:
   - Run:
     ```bash
     python AgentKosiV2.py
     ```
   - Enter queries interactively (e.g., "What languages does he write?"). Type `exit` to quit.
   - Example Output:
     ```
     Welcome to the RAG Agent! Type 'exit' to quit.
     Enter your query: What languages does he write?
     Query: What languages does he write?
     RAG Agent Response: Kosisochukwu writes Python, a little JavaScript, and a little Rust.
     ```

2. **Server Mode (AgentKosiV2.py)**:
   - Run:
     ```bash
     python AgentKosiV2.py --server
     ```
   - Access via `http://localhost:5000/query/What+languages+does+he+write?` (e.g., in a browser or with `curl`).
   - Example Response:
     ```json
     {"query": "What languages does he write?", "response": "Kosisochukwu writes Python, a little JavaScript, and a little Rust."}
     ```

3. **Basic Mode (AgentKosiV1.py)**:
   - Run:
     ```bash
     python AgentKosiV1.py
     ```
   - Processes predefined queries and prints responses.

### Example Queries
- "What languages does he write?"
- "What database does he use?"
- "Where is he from?"
- "What is his GitHub?"
- "What does he study?" (e.g., software engineering)
- "Does he containerize his applications?" (e.g., yes, with Docker)

## Project Structure

- `AgentKosiV1.py`: Initial RAG agent implementation.
- `AgentKosiV2.py`: Enhanced version with chunking, interactive loop, and FastAPI support.
- `RAG_Agent.ipynb`: Notebook for development and testing.
- `RAG_Agent.py`: Early standalone script.
- `.env`: Local environment file (not committed).
- `requirements.txt`: Dependency list.
- `Dockerfile`: Container configuration for Northflank deployment.
- `start.sh`: Startup script for Northflank.
- `README.md`: This documentation.

## Deployment

### Northflank with Docker

This project is designed for deployment on Northflank using Docker. Follow these steps:

1. **Prepare Files**:
   - Ensure `Dockerfile`, `start.sh`, and `requirements.txt` are in the project root (details below).
   - Add `GROQ_API_KEY` as a Northflank secret (see Northflank setup).

2. **Push to Git**:
   - Commit and push to your GitHub repository:
     ```bash
     git add .
     git commit -m "Prepare for Northflank deployment"
     git push origin main
     ```

3. **Northflank Setup**:
   - Log in to [northflank.com](https://northflank.com).
   - Create a project (e.g., `RAG-Agent-Kosi`).
   - Add a service: "Container Image" > "Build from Git repository".
   - Configure:
     - Point to the `Dockerfile`.
     - Expose port 5000.
     - Add secret: `GROQ_API_KEY` with your API key.
   - Deploy and test the provided URL.

## Files for Deployment

### `requirements.txt`
```plaintext
swarmauri==0.4.1
python-dotenv==1.0.1
groq==0.25.0
fastapi==0.111.0
uvicorn==0.29.0
gunicorn==22.0.0
scikit-learn==1.6.1
joblib==1.4.2
```

### `Dockerfile`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GROQ_API_KEY=${GROQ_API_KEY}
CMD ["sh", "start.sh"]
```

### `start.sh`
```bash
#!/bin/bash
# Ensure the script is executable on the container
python AgentKosiV2.py --server
```

- Make `start.sh` executable locally:
  ```bash
  chmod +x start.sh
  ```

## Should You Add `rag_env` to `.gitignore`?

Yes, you should add `rag_env/` to your `.gitignore` file. Virtual environments like `rag_env` contain platform-specific files and dependencies that should not be committed to version control. This keeps your repository clean and ensures others (or your deployed environment) use the correct dependency management via `requirements.txt`.

### Updated `.gitignore`
Create or update `.gitignore` in the project root with:
```
# Virtual environment
rag_env/

# Environment variables
.env

# Python build artifacts
__pycache__/
*.pyc

# Docker
*.dockerignore

# IDE-specific files
.vscode/
.idea/
```

## Contributing

Contributions are welcome! Please:
- Fork the repository.
- Create a feature branch (`git checkout -b feature-name`).
- Commit changes (`git commit -m "Add feature-name"`).
- Push and open a pull request.

Report issues or suggest enhancements via GitHub Issues.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, contact me via:
- GitHub: [github.com/Ksschkw](https://github.com/Ksschkw)
- WhatsApp: +2349019549473 (include context when messaging).

## Future Plans

- Enhance document chunking for better context retention.
- Add support for multiple vector stores (e.g., Faiss for scalability).
- Expand deployment options (e.g., AWS, Render).
- Integrate user authentication for the API.
