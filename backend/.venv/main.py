from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Chat Model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Define Prompt Template
prmpt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""
You are an AI tutor that ONLY answers questions related to IT and Computer Science.
If the user asks about anything else, say:
"I can only answer questions related to IT and Computer Science. Please ask me a question related to IT and Computer Science."

Respond clearly in Markdown and never answer unrelated questions.

{chat_history}
User: {user_input}
AI:
"""
)

# Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Chain
ai_tutor_chain = LLMChain(
    llm=client,
    prompt=prmpt,
    memory=memory
)

# Tools
tools = [
    Tool(
        name="AI Tutor",
        func=lambda user_input: ai_tutor_chain.run({"user_input": user_input}),
        description="Useful for answering questions related to IT and Computer Science."
    )
]

# Initialize Agent
agent_executor = initialize_agent(
    tools=tools,
    llm=client,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Prompt Templates
prompts = {
    "explanation": lambda topic: f"Explain {topic} in simple terms.",
    "study_notes": lambda topic: f"Provide study notes for {topic}.",
    "quiz": lambda topic: f"Create a quiz for {topic} with 5 questions.",
    "hands_on": lambda topic: f"Provide a hands-on project for {topic}.",
    "learning_path": lambda topic: f"Suggest a learning path for {topic}.",
    "summary": lambda topic: f"Summarize the topic {topic}.",
    "custom_question": lambda question: f"Answer the question: {question}."
}


# Request Data Model
class RequestData(BaseModel):
    topic: str = "(Optional)"
    query_type: str = "(Optional)"
    custom_question: str = ""

# Tutor Logic
def ai_tutor(topic: str, query_type: str, custom_question: str) -> str:
    topic_val = topic if topic != "(Optional)" else None
    query_val = query_type if query_type != "(Optional)" else None
    user_input = custom_question.strip()

    if not user_input and not (topic_val and query_val):
        return "Please provide either a custom question or select both a topic and a query type."

    if user_input and query_val and topic_val:
        system_prompt = prompts[query_val](topic_val)
        full_prompt = f"{system_prompt}\n\nAlso consider this specific question: '{user_input}'"
    elif query_val and topic_val:
        full_prompt = prompts[query_val](topic_val)
    else:
        full_prompt = prompts["custom_question"](user_input)

    try:
        response = agent_executor.invoke({"input": full_prompt})["output"]
        return response
    except Exception as e:
        return f"**Error:** {str(e)}"

# API Endpoint
@app.post("/ai_tutor/")
async def ai_tutor_endpoint(request_data: RequestData):
    result = ai_tutor(request_data.topic, request_data.query_type, request_data.custom_question)
    return {"response": result}

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Tutor"}
