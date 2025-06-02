# MYRAGAGENT

## Overview

Welcome to **MYRAGAGENT**, a project that implements a **Retrieval-Augmented Generation (RAG) Agent** using the `swarmauri` library and the Groq API from Groq, Inc. This agent combines a TF-IDF vector store for document retrieval with a conversational AI model to provide intelligent, context-aware responses based on personal details about the developer, Kosisochukwu. The project has evolved through multiple iterations, supporting local testing, interactive querying, and deployment via Docker on Northflank.

Key files include:
- **`AgentKosiV1.py`**: Initial version of the RAG agent with basic document handling and query processing.
- **`AgentKosiV2.py`**: Enhanced and final version of the RAG agent, featuring large document chunking, session-based conversation management, an interactive query loop for local testing, and a FastAPI interface for deployment.
- **`RAG_Agent.ipynb`**: Jupyter Notebook for step-by-step development and testing of the RAG agent.
- **`RAG_Agent.py`**: Early standalone script that consolidates the functionality from the notebook.

The RAG agent answers queries about programming skills, database usage, deployment platforms, education, hobbies, and more, leveraging both document data and the Groq model's general knowledge.

## Features

- **Document Management**: Utilizes `TfidfVectorStore` with chunking for large texts, ensuring comprehensive context.
- **Conversational AI**: Integrates the Groq API (`llama3-8b-8192` model) for natural language responses.
- **Interactive Querying**: Offers a local loop for continuous questioning and a web API endpoint.
- **Scalability**: Handles detailed personal profiles with overlapping chunks for better retrieval.
- **Flexible Responses**: Uses general knowledge when document data is unavailable, avoiding "not found" messages.

## Prerequisites

To run or deploy this project, you need:

- **Python**: Version 3.8 or higher (tested with 3.12).
- **Dependencies** (listed in `requirements.txt`):
  - `swarmauri==0.4.1`
  - `python-dotenv==1.0.1`
  - `groq==0.26.0`
  - `fastapi==0.115.12`
  - `uvicorn==0.34.2`
  - `scikit-learn==1.6.1`
  - `joblib==1.5.1`
  - `typing-extensions==4.13.2`
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
   git clone https://github.com/Ksschkw/MYRAGAGENT.git
   cd MYRAGAGENT
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your Groq API key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - Do **not** commit this file (add to `.gitignore`).

3. **Install Dependencies**:
   Run:
   ```bash
   pip install -r requirements.txt
   ```
   Activate your virtual environment (e.g., `rag_env`) if using one:
   ```powershell
   .\rag_env\Scripts\activate
   ```

4. **Prepare Documents**:
   - Documents are embedded in the code (e.g., `documents` list in `AgentKosiV2.py`).
   - Modify or add large documents using the `chunk_document` function.

### Virtual Environment Recommendation

- Create a virtual environment:
  ```bash
  python -m venv rag_env
  source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
  pip install -r requirements.txt
  ```
- Add `rag_env/` to `.gitignore` to avoid committing it.

## Usage

### Running Locally

#### Jupyter Notebook (`RAG_Agent.ipynb`)
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `RAG_Agent.ipynb` and execute cells to:
   - Initialize the vector store.
   - Configure conversation context and integrate the Groq API.
   - Test with sample queries.

#### Python Script (`AgentKosiV1.py` or `AgentKosiV2.py`)
1. **Interactive Mode (AgentKosiV2.py)**:
   - Run:
     ```bash
     python AgentKosiV2.py
     ```
   - Enter queries (e.g., "What languages does he write?"). Type `exit` to quit.
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
   - Access via `http://localhost:5000/query/What+languages+does+he+write?`.
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

4. **Local Testing with `TestClient`**:
   - The interactive loop in `AgentKosiV2.py` uses FastAPI’s `TestClient` to simulate HTTP requests, allowing you to test the API locally without starting a server.

### Deployed Versions

The application is deployed and accessible online:

- **API Endpoint**: Query the RAG agent directly at [https://p01--myragagent--qw5xhkblp8hy.code.run/query/{query}](https://p01--myragagent--qw5xhkblp8hy.code.run/query/{query}). Replace `{query}` with your question. For example:
  - [https://p01--myragagent--qw5xhkblp8hy.code.run/query/What+languages+does+he+write?](https://p01  - https://p01--myragagent--qw5xhkblp8hy.code.run/query/What+languages+does+he+write?)
  - [https://p01--myragagent--qw5xhkblp8hy.code.run/query/What+is+his+GitHub?](https://p01--myragagent--qw5xhkblp8hy.code.run/query/What+is+his+GitHub?)

- **Web GUI**: Access the interactive interface at [https://agentkosi.onrender.com/](https://agentkosi.onrender.com/). Enter your queries in the provided input field to interact with the RAG agent.

### Example Queries
- "What languages does he write?"
- "What database does he use?"
- "Where is he from?"
- "What is his GitHub?"
- "What does he study?" (e.g., software engineering)
- "Does he containerize his applications?" (e.g., yes, with Docker)

## Project Structure

- **`AgentKosiV1.py`**: Initial RAG agent implementation with basic functionality.
- **`AgentKosiV2.py`**: Enhanced version with chunking, session management, interactive loop, and FastAPI.
- **`RAG_Agent.ipynb`**: Notebook for development and testing.
- **`RAG_Agent.py`**: Early standalone script.
- **`start.sh`**: Shell script to start the FastAPI server for deployment.
- **`Dockerfile`**: Configuration for containerizing the application.
- `.env`: Local environment file (not committed).
- `requirements.txt`: Dependency list.
- `README.md`: This documentation.

## Deployment

### Northflank with Docker

This project is containerized for deployment on Northflank using Docker. Follow these steps:

1. **Prepare Files**:
   - Ensure `Dockerfile`, `start.sh`, and `requirements.txt` are in the project root.
   - Add `GROQ_API_KEY` as a Northflank secret.

2. **Build and Test Locally**:
   - Build the Docker image:
     ```bash
     docker build -t rag-agent-kosi .
     ```
   - Run the container:
     ```bash
     docker run -p 5000:5000 -e GROQ_API_KEY=your-api-key-here rag-agent-kosi
     ```
   - Test at `http://localhost:5000/query/What+languages+does+he+write?`.

3. **Push to GitHub**:
   - Commit and push:
     ```bash
     git add .
     git commit -m "Prepare for Northflank deployment"
     git push origin main
     ```

4. **Northflank Setup**:
   - Log in to [northflank.com](https://northflank.com).
   - Create a project (e.g., `MYRAGAGENT`).
   - Add a service: "Container Image" > "Build from Git repository".
   - Configure:
     - Point to the `Dockerfile`.
     - Expose port 5000.
     - Add secret: `GROQ_API_KEY` with your API key.
   - Deploy and test the provided URL.

### Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip
COPY . .
ENV GROQ_API_KEY=${GROQ_API_KEY}
CMD ["sh", "start.sh"]
```

### .dockerignore
```
__pycache__
*.pyc
*.py[cod]
*$py.class
.env
.rag_env/
.git
.gitignore
.vscode/
```

## Contributing

Contributions are welcome! Please:
- Fork the repository.
- Create a feature branch (`git checkout -b feature-name`).
- Commit changes (`git commit -m "Add feature-name"`).
- Push and open a pull request.

Report issues or suggest enhancements via GitHub Issues at [https://github.com/Ksschkw/MYRAGAGENT](https://github.com/Ksschkw/MYRAGAGENT).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, contact Kosisochukwu via:
- GitHub: [github.com/Ksschkw](https://github.com/Ksschkw)
- WhatsApp: +2349019549473 (include context when messaging).

## Future Plans

- **Session Persistence**: Store session data in a database (e.g., SQLite) for long-term conversation history.
- **Improved Chunking**: Experiment with different chunk sizes or overlap strategies for better retrieval accuracy.
- **Multi-Model Support**: Allow switching between different Groq models dynamically.
