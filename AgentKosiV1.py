from swarmauri.standard.documents.concrete.Document import Document
from swarmauri.standard.vector_stores.concrete.TfidfVectorStore import TfidfVectorStore 

# Initialize vector store
vector_store = TfidfVectorStore()

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

# Add documents to the vector_store
vector_store.add_documents(documents)

# Verify that they have been added
# print(f"{len(vector_store.documents)} documents added to the vector store")


from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.messages.concrete.HumanMessage import HumanMessage


## Integrating the LLM
import os
from swarmauri.standard.llms.concrete.GroqModel import GroqModel as LLM
from swarmauri.standard.conversations.concrete.Conversation import Conversation
from dotenv import load_dotenv

# Load envienvironment vaariables
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

# check if API key is set 
if not API_KEY:
    print("Please set the API key in the .env g")

## Function to Get Allowed Models
# Function to gett allowed models filtering out the ones that fail
def get_allowed_models(llm):
    failing_llms=[
        "llama3-70b-8192",
        "llama-3.2-90b-text-preview",
        "mixtral-8x7b-32768",
        "llava-v1.5-7b-4096-preview",
        "llama-guard-3-8b",
        "gemma-7b-it",
    ]
    return [model for model in llm.allowed_models if model not in failing_llms]
# Initialize the GroqModel
llm = LLM(api_key=API_KEY, name='llama3-8b-8192') # I Explicitly set the model name because gemma-7b-it is not supported by the GroqModel any more

# Print model information
print(f"Resource: {llm.resource}")
print(f"Type: {llm.type}")
print(f"Default Name: {llm.name}")

# Get allowed models
allowed_models = get_allowed_models(llm)
print("Allowed Models: ", allowed_models)


#-----------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------#
#----------   -  BUILDING THE RAG AGENT  -  --------#
#---------------------------------------------------#

from swarmauri.standard.agents.concrete.RagAgent import RagAgent

# Creat a new system context for the rag agent
rag_system_context = "You are an assistant that provides answers to the user. You utilize the details below: "
# Create a new conversation for RAG Agent
rag_conversation = MaxSystemContextConversation(system_context=SystemMessage(content=rag_system_context), max_size=100)

# Initialize the RAG Agent by combining LLM, convrsation, and vector store
rag_agent = RagAgent(
    llm=llm,
    conversation=rag_conversation,
    system_context=rag_system_context,
    vector_store=vector_store,
)
# Test the agent with different queries
queries = [
    "What languages does he write?",
    "What database does he use?",
    "Where is he from?",
    "What is his github?",
    "what does he study?",
    "Does he containerize his applications?",
    "Does he deploy his applications?",
]

for query in queries:
    response = rag_agent.exec(query)
    print(f"Query: {query}\nRAG AgentResponse: {response}\n")