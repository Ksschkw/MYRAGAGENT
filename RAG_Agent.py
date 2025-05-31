from swarmauri.standard.documents.concrete.Document import Document
from swarmauri.standard.vector_stores.concrete.TfidfVectorStore import TfidfVectorStore 

# Initialize vector store
vector_store = TfidfVectorStore()

documents = [
    Document(content="His write Python, a little JavaScript, a little rust, i use dockerfile too"),
    Document(content="His use sqlite for my database(small projects), json(not really a database) "),
    Document(content="His use Render, Railway, Northflank and more for deployment"),
    Document(content="His use docker for containerization"),
    Document(content="His course of study is software engineering"),
    Document(content="His am a Nigerian, currently in Lagos"),
    Document(content="His am from earth, i am not a martian"),
    Document(content="His name is kosisochukwu"),
    Document(content="His github page is at github.com/Ksschkw"),
    
]

# Add documents to the vector_store
vector_store.add_documents(documents)

# Verify that they have been added
print(f"{len(vector_store.documents)} documents added to the vector store")

# Retrieve and print all documents
all_docs = vector_store.get_all_documents()

print("All documente in te vector store: ")
for doc in all_docs:
    print(doc.content)

# Configuring the conversation Context

from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.messages.concrete.HumanMessage import HumanMessage

# Create a system message
system_context = SystemMessage(content="Your name is Kosisochukwu")
# Initialize the conversation
conversation = MaxSystemContextConversation(system_context=system_context, max_size=4)
# Add a user message
user_message = HumanMessage(content="What is my name?")
conversation.add_message(user_message)
# Print the current conversation context
print("current conversation history")
for message in conversation.history:
    print(message.content)

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
    ]
    return [model for model in llm.allowed_models if model not in failing_llms]
# Initialize the GroqModel
llm = LLM(api_key=API_KEY)

# Print model information
print(f"Resource: {llm.resource}")
print(f"Type: {llm.type}")
print(f"Default Name: {llm.name}")

# Get allowed models
allowed_models = get_allowed_models(llm)
print("Allowed Models: ", allowed_models)

#-------------------------------------------------------------------------------------------------------------------------------
#EXAMPLES-----------------------------------------------------------------------------------------------------------------------

#--------------------------------------
# EXAMPLE USAGE WITH NO SYSTEM CONTEXT 
#--------------------------------------
# Set the model name to the first available allowed model
llm.name = allowed_models[0]
# Create a conversation
conversation = Conversation()
# Add a human message
input_data = "Hello"
human_message = HumanMessage(content=input_data)
conversation.add_message(human_message)
# Predict response
llm.predict(conversation=conversation)
prediction =  conversation.get_last().content
print(f"Prediction with no system context for {llm.name}: {prediction}")

#------------------------------------
# Example Usage with a System Context
#------------------------------------

system_context =  "You are an assistant that proides answers to the user."
conversation = MaxSystemContextConversation(system_context=SystemMessage(content=system_context), max_size=2)
# Create a human message with the content "HI" and add it to the conversation
human_message = HumanMessage(content="Hi")
conversation.add_message(human_message)
# Predict response
llm.predict(conversation=conversation)
prediction = conversation.get_last().content
print(f"Prediction with system context for {llm.name}: {prediction}")

#-----------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------#
#----------   -  BUILDING THE RAG AGENT  -  --------#
#---------------------------------------------------#

from swarmauri.standard.agents.concrete.RagAgent import RagAgent

# Creat a new system context for the rag agent
rag_system_context = "You are an assistant that provides answers to the user. You utilize the details below: "
# Create a new conversation for RAG Agent
rag_conversation = MaxSystemContextConversation(system_context=SystemMessage(content=rag_system_context), max_size=4)

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
    "Wha is his github?",
]

for query in queries:
    response = rag_agent.exec(query)
    print(f"Query: {query}\nRAG AgentResponse: {response}\n")