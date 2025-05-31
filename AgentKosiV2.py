from swarmauri.standard.documents.concrete.Document import Document
from swarmauri.standard.vector_stores.concrete.TfidfVectorStore import TfidfVectorStore 
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.messages.concrete.HumanMessage import HumanMessage
import os
from swarmauri.standard.llms.concrete.GroqModel import GroqModel as LLM
from swarmauri.standard.conversations.concrete.Conversation import Conversation
from dotenv import load_dotenv
from swarmauri.standard.agents.concrete.RagAgent import RagAgent
from fastapi import FastAPI
import uvicorn
import sys

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GROQ_API_KEY in the .env file")

# Function to chunk large documents
def chunk_document(content, chunk_size=200, overlap=20):
    words = content.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break
    return chunks

# Initialize vector store
vector_store = TfidfVectorStore()

# Small documents
documents = [
    Document(content="He writee Python, a little JavaScript, a little rust, i use dockerfile too"),
    Document(content="He uses sqlite for my database(small projects), json(not really a database) "),
    Document(content="He uses Render, Railway, Northflank and more for deployment"),
    Document(content="He uses docker for containerization"),
    Document(content="His course of study is software engineering"),
    Document(content="He is a Nigerian, currently in Lagos"),
    Document(content="He is from earth, i am not a martian"),
    Document(content="His name is kosisochukwu"),
    Document(content="His github page is at github.com/Ksschkw"),
]

# Add small documents to the vector store
vector_store.add_documents(documents)

# Adding a large document about yourself
large_about_me = """
My name is Kosisochukwu. I am a software engineer from Nigeria, currently living in Lagos.
I have experience in Python, JavaScript, and Rust. I use Docker for containerization and
deployment. My GitHub page is github.com/Ksschkw. I studied software engineering and
have worked on various projects, including small applications using SQLite and larger
systems deployed on platforms like Render, Railway, and Northflank. I enjoy solving
complex problems and have contributed to open-source projects. My interests include
cloud computing, DevOps, and building scalable applications. I have also experimented
with machine learning frameworks like TensorFlow and enjoy learning new technologies. 
I am learning rust to develop with anchor on solana, and maybe for other stuff, 
i mean rust is fast, suports parallelism and concurency, what more could a software engineer ask for ?.
Kosisochukwu, a passionate and driven individual with a knack for all things technical and creative. 
Currently, I'm a software engineering student in my fourth year at the Federal University of Technology Owerri, where I'm honing my skills and preparing to make a significant impact in the tech world.
When I'm not deep into my studies or writing programs, you can find me indulging in puzzles like Sudoku and jigsaws, or relaxing with some anime. These hobbies keep me sharp, focused, and inspired.
One of my core values is the relentless pursuit of excellence—I simply refuse to be average. This mindset pushes me to strive for greatness in everything I do, whether it's acing a project, solving a complex puzzle, or building innovative software solutions.
What gets me out of bed every morning? The thrill of creation and the quest for financial independence. There's nothing quite like the sense of fulfillment I experience when I build something from scratch and watch it work as intended. It's a feeling that drives me to keep pushing boundaries and seeking new challenges.
I'm constantly evolving, learning, and aiming to be the best version of myself. Welcome to my journey. my github page is at github.com/Ksschkw, where you can find some of my projects and contributions. My phone number for whatsApp is 2349019549473
"""
chunks = chunk_document(large_about_me, chunk_size=50, overlap=10)
for i, chunk in enumerate(chunks):
    vector_store.add_documents([Document(content=chunk, metadata={"id": "about_me", "chunk_id": i})])

# Verify documents added
print(f"{len(vector_store.get_all_documents())} documents added to the vector store")

# Initialize the GroqModel
llm = LLM(api_key=API_KEY, name='llama3-8b-8192')

# Function to get allowed models
def get_allowed_models(llm):
    failing_llms = [
        "llama3-70b-8192",
        "llama-3.2-90b-text-preview",
        "mixtral-8x7b-32768",
        "llava-v1.5-7b-4096-preview",
        "llama-guard-3-8b",
        "gemma-7b-it",
    ]
    return [model for model in llm.allowed_models if model not in failing_llms]

# Print model information
print(f"Resource: {llm.resource}")
print(f"Type: {llm.type}")
print(f"Default Name: {llm.name}")
allowed_models = get_allowed_models(llm)
print("Allowed Models: ", allowed_models)

# Create a new system context for the RAG agent
rag_system_context = """Your name is kosisochukwu and you provide answers to the user. 
                        If the information is not available in the provided details, use your general knowledge to answer the question.
                        Never start replies with "According to the provided details" or similar phrases.
                     """

# Create a new conversation for the RAG agent
rag_conversation = MaxSystemContextConversation(system_context=SystemMessage(content=rag_system_context), max_size=100)

# Initialize the RAG Agent
rag_agent = RagAgent(
    llm=llm,
    conversation=rag_conversation,
    system_context=rag_system_context,
    vector_store=vector_store,
)

# FastAPI setup for deployment
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging to confirm middleware application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
logger.info("Applying CORS middleware with allow_origins=['*']")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# Add an explicit OPTIONS endpoint to handle preflight requests
@app.options("/query/{query}")
async def options_handler(query: str):
    logger.info("Handling OPTIONS request for /query/{query}")
    return {"status": "ok"}

# Rest of your code (ensure your existing /query/{query} endpoint remains)

@app.get("/query/{query}")
async def query_endpoint(query: str):
    response = rag_agent.exec(query)
    return {"query": query, "response": response}

# Interactive loop for local testing
def run_interactive_loop():
    print("Welcome to the RAG Agent! Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            response = rag_agent.exec(query)
            print(f"Query: {query}\nRAG Agent Response: {response}\n")
        except Exception as e:
            print(f"Error processing query: {e}\n")

if __name__ == "__main__":
    # Check for command-line argument to determine mode
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Run as a server for deployment
        uvicorn.run(app, host="0.0.0.0", port=5000)
    else:
        # Run in interactive mode for local testing
        run_interactive_loop()