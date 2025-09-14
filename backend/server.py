from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import uuid
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import io

# Import Emergent integrations
# from emergentintegrations.llm.chat import LlmChat, UserMessage
from openai import AsyncOpenAI


# Import our lightweight modules
from database import db
from lightweight_embeddings import embeddings_engine

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure Emergent LLM
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Models
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

# Utility functions - SIMPLIFIED
def create_token(user_id: str) -> str:
    # Super simple token - just prefix + user_id
    return f"simple_token_{user_id}"

def verify_token(token: str) -> str:
    # Super simple token verification
    if token and token.startswith("simple_token_"):
        return token.replace("simple_token_", "")
    raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return verify_token(credentials.credentials)

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# async def generate_answer_with_openai(question: str, context: str) -> str:
#     """Generate answer using Emergent Universal API"""
#     try:
#         # Create a unique session ID for this query
#         session_id = f"docubrain_{uuid.uuid4().hex[:8]}"
        
#         # Initialize the chat with Emergent LLM
#         chat = LlmChat(
#             api_key=EMERGENT_LLM_KEY,
#             session_id=session_id,
#             system_message="You are a helpful assistant that answers questions based on provided document context. Use only the provided information and be concise."
#         ).with_model("openai", "gpt-4o-mini")  # Using default model as recommended
        
#         # Create the prompt
#         prompt = f"""Based on the context below, answer the question concisely. Use only the provided information.

# Context:
# {context}

# Question: {question}

# Answer:"""
        
#         # Create user message
#         user_message = UserMessage(text=prompt)
        
#         # Send message and get response
#         response = await chat.send_message(user_message)
        
#         return response
        
#     except Exception as e:
#         print(f"Error generating response with Emergent LLM: {e}")
#         return f"Error generating response: {str(e)}"








async def generate_answer_with_openai(question: str, context: str) -> str:
    """Generate answer using OpenAI API directly"""
    try:
        client = AsyncOpenAI(api_key=os.environ.get("EMERGENT_LLM_KEY"))

        prompt = f"""Based on the context below, answer the question concisely. 
        Use only the provided information.

        Context:
        {context}

        Question: {question}

        Answer:"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided document context. Use only the provided information and be concise."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return f"Error generating response: {str(e)}"






# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    await db.init_db()

# Authentication endpoints
@api_router.post("/auth/register")
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    user_id = str(uuid.uuid4())
    api_key = f"sk-docubrain-{uuid.uuid4().hex[:20]}"
    
    user = {
        "user_id": user_id,
        "username": user_data.username,
        "password": user_data.password,  # Store plain password for simplicity
        "api_key": api_key,
        "created_at": datetime.utcnow()
    }
    
    success = await db.create_user(user)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to create user")
    
    token = create_token(user_id)
    
    return {
        "success": True,
        "message": "Registration successful!",
        "user_id": user_id,
        "token": token,
        "api_key": api_key
    }

@api_router.post("/auth/login")
async def login(user_data: UserLogin):
    # Find user
    user = await db.get_user_by_username(user_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Simple password check
    if user_data.password != user["password"]:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_token(user["user_id"])
    
    return {
        "success": True,
        "message": "Login successful!",
        "user_id": user["user_id"],
        "token": token,
        "api_key": user["api_key"]
    }

# Document endpoints
@api_router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    content = await file.read()
    text = extract_text_from_pdf(content)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Process document with lightweight embeddings
    chunks = chunk_text(text)
    embeddings = embeddings_engine.get_embeddings_tfidf(chunks)
    
    # Save to database
    doc_id = str(uuid.uuid4())
    document = {
        "id": doc_id,
        "user_id": user_id,
        "filename": file.filename,
        "content": text,
        "chunks": chunks,
        "embeddings": embeddings,
        "upload_time": datetime.utcnow(),
        "chunk_count": len(chunks),
        "status": "completed"
    }
    
    success = await db.create_document(document)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save document")
    
    return {"message": "Document uploaded and processed successfully", "document_id": doc_id}

@api_router.post("/documents/text")
async def add_text_document(
    title: str = Form(...),
    content: str = Form(...),
    user_id: str = Depends(get_current_user)
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    # Process text with lightweight embeddings
    chunks = chunk_text(content)
    embeddings = embeddings_engine.get_embeddings_tfidf(chunks)
    
    # Save to database
    doc_id = str(uuid.uuid4())
    document = {
        "id": doc_id,
        "user_id": user_id,
        "filename": f"{title}.txt",
        "content": content,
        "chunks": chunks,
        "embeddings": embeddings,
        "upload_time": datetime.utcnow(),
        "chunk_count": len(chunks),
        "status": "completed"
    }
    
    success = await db.create_document(document)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save document")
    
    return {"message": "Text document processed successfully", "document_id": doc_id}

@api_router.get("/documents")
async def get_documents(user_id: str = Depends(get_current_user)):
    documents = await db.get_user_documents(user_id)
    return documents

# Query endpoint
@api_router.post("/query", response_model=QueryResponse)
async def query_documents(query: QueryRequest, user_id: str = Depends(get_current_user)):
    # Get user documents with content
    documents = await db.get_user_documents_with_content(user_id)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents found. Please upload some documents first.")
    
    # Find relevant chunks across all documents using lightweight embeddings
    all_relevant_chunks = []
    
    for doc in documents:
        relevant_chunks = embeddings_engine.find_relevant_chunks(
            query.question, 
            doc["chunks"], 
            doc["embeddings"]
        )
        
        for chunk in relevant_chunks:
            chunk['filename'] = doc['filename']
            all_relevant_chunks.append(chunk)
    
    if not all_relevant_chunks:
        return QueryResponse(
            answer="I couldn't find relevant information in your documents to answer this question.",
            sources=[]
        )
    
    # Sort by relevance and take top 5
    all_relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
    top_chunks = all_relevant_chunks[:5]
    
    # Create context for Emergent LLM
    context = "\n\n".join([chunk['content'] for chunk in top_chunks])
    
    # Generate answer using Emergent Universal API
    answer = await generate_answer_with_openai(query.question, context)
    
    # Prepare sources
    sources = [
        {
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "relevance_score": chunk["relevance_score"]
        }
        for chunk in top_chunks
    ]
    
    return QueryResponse(answer=answer, sources=sources)

# External API endpoint
@api_router.post("/external/query")
async def external_query(
    api_key: str = Form(...),
    question: str = Form(...)
):
    # Find user by API key
    user = await db.get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Use the regular query logic
    query_request = QueryRequest(question=question)
    return await query_documents(query_request, user["user_id"])

# Include the router in the main app
app.include_router(api_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "DocuBrain API is running! 🧠",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /api/auth/register",
            "login": "POST /api/auth/login", 
            "upload": "POST /api/documents/upload",
            "query": "POST /api/query",
            "docs": "/docs"
        }
    }

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)