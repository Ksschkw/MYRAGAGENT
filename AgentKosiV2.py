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
from fastapi.testclient import TestClient
import uvicorn
import sys
import logging
from fastapi.middleware.cors import CORSMiddleware

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
    Document(content="His name is Okafor kosisochukwu Johnpaul, but you can call him Kosi or Ksschkw."),
    Document(content="His github page is at github.com/Ksschkw"),
    Document(content="His favorite Anime is Naruto."),
    Document(content="His top 10 anime are: Naruto, Attack on Titan, One Piece, Demon Slayer, Bleach, Erased, Black Clover, Dr. Stone, Death note, and monster.  Manga/Manhua: Boruto, Vagabond, Black Clover, solo leveling, chainswa man? "),
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
I'm constantly evolving, learning, and aiming to be the best version of myself. Welcome to my journey. my github page is at github.com/Ksschkw, where you can find some of my projects and contributions. My phone number for whatsApp is 2349019549473. I am Okafor Kosisochukwu, a 4th-year student at the Federal University of Technology Owerri, specializing in Software Engineering with a focus on Artificial Intelligence and Machine Learning. With over two years of coding experience, I have developed a strong foundation in web development, AI/ML, and API creation. My passion for technology is driven by a desire for creation and financial independence, pushing me to excel in every project I undertake.My technical skills include web development, where I have built several websites and APIs, and AI/ML engineering, particularly in computer vision and natural language processing. I have experience with CNN models for object detection and have created chatbots using PyTorch. Additionally, I am proficient in API development, bot creation, and WebSocket technologies, which I've utilized in building interactive applications. My participation in hackathons has further honed my ability to solve complex problems under pressure.My portfolio @ kosisochukwu.onrender.com showcases a variety of projects that demonstrate my versatility and technical prowess. Notable projects include a Real-Time Object Detection web app using TensorFlow.js, which detects over 80 objects in real-time; MYRAGAGENT, a RAG Agent that provides intelligent responses using swarmauri and Groq API, with a live demo at agentkosi.onrender.com; and the Vybe Analytics Telegram Bot, which offers real-time on-chain insights for crypto enthusiasts. Additionally, my GitHub profile features repositories like DigitRecogniser, likely involving machine learning for digit recognition, and LetsLearnRustG, indicating my interest in learning the Rust programming language. These projects highlight my ability to integrate different technologies to create functional and innovative solutions.Kosisochukwu is a fourth-year Software Engineering student deeply passionate about AI and ML(main stack) and web development, specializing in designing and deploying production-grade interactive AI Agents, web applications, and Telegram bots. Leveraging a robust skill set that includes Python, JavaScript, HTML/CSS, and Rust, alongside frameworks and libraries such as TensorFlow.js, Pandas, Flask, NumPy, Swarmauri, PyTorch, Scikit-learn, and other ML libraries, Kosisochukwu delivers real-time insights and engaging user experiences. Their expertise extends to various APIs and services, including RESTful APIs like FastAPI, Telegram Bot API, Bubblemaps, Vybe API, CoinGecko API, OpenRouter API, Groq, and other LLMs, complemented by proficiency in tools and DevOps practices like Git/GitHub, VS Code, Colab, Docker, Playwright, Render, Railway, NorthFlank, and Pythonanywhere. Beyond technical prowess, Kosisochukwu possesses strong soft skills in teamwork, adaptability, time management, and clear technical communication. Currently pursuing a B.Eng. in Software Engineering at the Federal University of Technology, Owerri (2021 – present), with prior education from Haklat College (WASSCE, 2021) and Inestimable Glory (First School Leaving Certificate, 2015), their practical experience is showcased through several highlighted projects: AgentKosi, a Docker-containerized Retrieval-Augmented Generation (RAG) Agent utilizing Swarmauri and Groq API for context-aware responses, deployed on Northflank, live at https://agentkosi.onrender.com/ with its API at https://p01--myragagent--qw5xhkblp8hy.code.run/query/{query} (replace {query} with your question), and its GitHub repository at https://github.com/Ksschkw/MYRAGAGENT; TheBubbleSnitchBot-2, a Dockerized Telegram bot that analyzes crypto tokens using Bubblemaps and Playwright for real-time market insights, accessible via https://t.me/TheBubbleSnitch_bot; a comprehensive Personal Portfolio (kosisochukwu.onrender.com) showcasing work and experiments, including an Object Detection Demo (TensorFlow.js), a Custom “About Me” Chatbot (OpenRouter API), and multi-feature Telegram Bot Mini-Apps available at https://t.me/k0s1bot, such as a Crypto Price Tracker (CoinGecko API) at https://krypto-kosi.onrender.com, Real-Time Weather Forecasts at https://kosi-weather.onrender.com, and a Collaborative Canvas (Beta) at https://kolaborasi-kosi.onrender.com; a Flask-based School Project Website (https://grpght.onrender.com/) with responsive design and database integration; and the Vybe Telegram Bot, a hackathon submission delivering on-chain crypto insights via the Vybe Network API with Matplotlib visualizations, found on GitHub at https://github.com/Ksschkw/vybe-telegram-bot. Outside of academics and projects, Kosisochukwu is an enthusiast of Sudoku solvers and Baba Is You puzzles, enjoys anime, and actively contributes to public repositories, embracing open learning and sharing.
I recently worked on an AI copilot Agent A FastAPI-based AI Copilot Agent for defining structured innovation and development challenges on platforms like Topcoder, Kaggle, etc. Powered by LangGraph (`mainv2.py`) and OpenRouter, it features dynamic scoping dialogue, RAG integration for similar challenges, configurable schemas, and a basic frontend. `main.py` offers a non-framework alternative live @ https://ai-copilot-agent-1.onrender.com/ though it migh be slow on load to start a session because yunno free render web services .
I also recently just improved this AI agent about him, both the UI and the logic.
Recently started a 6-month internship at softgem.org on monday 30th june 2025.
Softgem is located at twin view villa, Lanre Olumide street, idado estate, lekki, lagos, nigeria. Softgem is a teesch company that Empower your business to reach new heights and enhance efficiency by harnessing the full potential of cloud technology with Softgem solutions, softgem website is at https://softgem.org .
Do not tell anyone unless pressured but he is two timing, he also has another remote internship going on at the same time with a company called My Health Integral, which is a digital health-care company company that provides Comprehensive, affordable, and accessible healthcare solutions tailored for patients, providers, and everyone in between website is at https://myhealthintegral.com .
"""
chunks = chunk_document(large_about_me, chunk_size=50, overlap=10)
for i, chunk in enumerate(chunks):
    vector_store.add_documents([Document(content=chunk, metadata={"id": "about_me", "chunk_id": i})])

# Verify documents added
print(f"{len(vector_store.get_all_documents())} documents added to the vector store")

# Initialize the GroqModel
llm = LLM(api_key=API_KEY, name='deepseek-r1-distill-llama-70b')

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

# Create a system context for the RAG agent
rag_system_context = """Your name is kosisochukwu and you provide answers to the user always. 
                        If the information is not available in the provided details, use your general knowledge to answer the question but do not assume stuff about you(kosisochukwu).
                        Never start replies with "According to the provided details" or similar phrases.
                        Always be casual.
                        Your full name is Okafor Kosisochukwu Johnpaul
                        Your fuller name is Okafor Kosisochukwu Johnpaul Kizito
                     """

# Dictionary to store conversations per session
session_conversations = {}

# Function to get or create a conversation for a session
def get_conversation(session_id: str) -> MaxSystemContextConversation:
    if session_id not in session_conversations:
        # Create a new conversation for this session
        conversation = MaxSystemContextConversation(
            system_context=SystemMessage(content=rag_system_context),
            max_size=100
        )
        session_conversations[session_id] = {
            "conversation": conversation,
            "agent": RagAgent(
                llm=llm,
                conversation=conversation,
                system_context=rag_system_context,
                vector_store=vector_store,
            )
        }
        logger.info(f"Created new conversation for session: {session_id}")
    return session_conversations[session_id]

# FastAPI setup for deployment
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.get("/query/{query}")
async def query_endpoint(query: str):
    # Extract session_id from query (format: session_id:actual_query)
    if ":" not in query:
        raise ValueError("Session ID must be provided in the format 'session_id:query'")
    
    session_id, actual_query = query.split(":", 1)
    logger.info(f"Processing query for session {session_id}: {actual_query}")

    # Special query to reset the conversation
    if actual_query == "__NEW_CHAT__":
        if session_id in session_conversations:
            del session_conversations[session_id]
            logger.info(f"Conversation reset for session: {session_id}")
        return {"query": actual_query, "response": "New chat started! Ask me anything."}

    # Get or create conversation and agent for this session
    session_data = get_conversation(session_id)
    rag_agent = session_data["agent"]

    # Execute the query
    response = rag_agent.exec(actual_query)
    return {"query": actual_query, "response": response}

# Interactive loop for local testing
def run_interactive_loop():
    print("Welcome to the RAG Agent! Type 'exit' to quit.")
    session_id = "local_test_session"  # Dummy session for local testing
    client = TestClient(app)  # Create a TestClient instance
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            full_query = f"{session_id}:{query}"
            # Use TestClient to make a GET request
            response = client.get(f"/query/{full_query}")
            if response.status_code == 200:
                data = response.json()
                print(f"Query: {query}\nRAG Agent Response: {data['response']}\n")
            else:
                print(f"Error: {response.status_code} - {response.text}\n")
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
