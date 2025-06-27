import os
import logging
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ------------------------
# ✅ Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------
# ✅ Load environment variables
# ------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY is missing in your .env file!")

# ------------------------
# ✅ Define models
# ------------------------
class ChatState(TypedDict):
    """State schema for the chatbot."""
    input: str
    output: str
    metadata: Dict[str, Any]  # For future extensibility

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str
    status: str = "success"

# ------------------------
# ✅ Initialize LLM
# ------------------------
def initialize_llm() -> ChatGroq:
    """Initialize and return the Groq LLM client."""
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1024
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

llm = initialize_llm()

# ------------------------
# ✅ Define chatbot node logic
# ------------------------
def ask_ai(state: ChatState) -> ChatState:
    """Send user input to the LLM and return the response."""
    try:
        user_input = state["input"]
        if not user_input.strip():
            raise ValueError("Empty input provided")
            
        logger.info(f"Processing input: {user_input[:50]}...")
        response = llm.invoke(user_input).content
        
        return {
            "output": response,
            "input": user_input,  # Pass through original input
            "metadata": state.get("metadata", {})
        }
    except Exception as e:
        logger.error(f"Error in ask_ai: {e}")
        return {
            "output": "Sorry, I encountered an error processing your request.",
            "input": state["input"],
            "metadata": {"error": str(e)}
        }

# ------------------------
# ✅ Create LangGraph
# ------------------------
def build_chat_graph():
    """Build and compile the chatbot graph."""
    try:
        graph = StateGraph(ChatState)
        graph.add_node("ask_ai", ask_ai)
        graph.set_entry_point("ask_ai")
        graph.add_edge("ask_ai", END)
        return graph.compile()
    except Exception as e:
        logger.error(f"Failed to build chat graph: {e}")
        raise

# ------------------------
# ✅ Initialize rate limiter
# ------------------------
limiter = Limiter(key_func=get_remote_address)

# ------------------------
# ✅ FastAPI app setup
# ------------------------
app = FastAPI(
    title="AI Chatbot API",
    description="API for interacting with Groq LLaMA-3 chatbot",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter

# Add rate limit exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS - ONLY allow your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbot-frontend-gamma-one.vercel.app"],  # Only your frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["*"],
)

# Initialize chatbot graph at startup
@app.on_event("startup")
async def startup_event():
    """Initialize resources when the app starts up."""
    try:
        app.state.chatbot = build_chat_graph()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

# ------------------------
# ✅ Chat endpoint
# ------------------------
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("15/minute")  # 15 requests per minute per IP
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """Handle chat requests and return AI responses."""
    try:
        result = app.state.chatbot.invoke({
            "input": chat_request.message,
            "metadata": {},  # Initialize empty metadata
            "output": ""  # Initialize empty output
        })
        
        return {
            "reply": result["output"],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

# ------------------------
# ✅ CLI Chat Interface
# ------------------------
def run_chat():
    """Run the chatbot in CLI mode."""
    try:
        chatbot = build_chat_graph()
        print("\n🧠 Simple AI Chatbot with Groq LLaMA-3")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    print("⚠️ Please enter a message.")
                    continue

                result = chatbot.invoke({
                    "input": user_input,
                    "metadata": {},
                    "output": ""
                })
                print("\nAI:", result["output"], "\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue

    except Exception as e:
        logger.error(f"CLI error: {e}")
        print("⚠️ Failed to start chatbot. Please check logs for details.")

# ------------------------
# ✅ App entry point
# ------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Run FastAPI server if executed directly
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
    # To run CLI instead, comment above and uncomment:
    # run_chat()