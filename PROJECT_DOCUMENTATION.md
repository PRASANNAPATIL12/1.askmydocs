# DocuBrain - Complete Project Documentation

## 🧠 Project Overview

**DocuBrain** is an intelligent document management and question-answering system that transforms your documents into a private, searchable knowledge base. Users can upload PDF files or add text documents, then ask natural language questions to get AI-powered answers based on their document content.

### Key Features
- 📄 **Document Upload**: Support for PDF files and direct text input
- 🤖 **AI-Powered Q&A**: Ask questions about your documents and get intelligent answers
- 🔐 **User Authentication**: Secure user registration and login system
- 🌐 **External API**: RESTful API for third-party integrations
- 💾 **Document Management**: View, organize, and manage uploaded documents
- 🔍 **Smart Search**: Advanced text similarity search using TF-IDF embeddings

---

## 🏗️ System Architecture

### Technology Stack

#### Backend
- **Framework**: FastAPI (Python)
- **Database**: MongoDB Atlas
- **AI Integration**: Emergent Universal API (GPT-4o-mini)
- **Document Processing**: PyPDF2 for PDF text extraction
- **Embeddings**: TF-IDF vectorization with scikit-learn
- **Authentication**: Simple token-based authentication

#### Frontend
- **Framework**: React 18
- **Routing**: React Router DOM
- **HTTP Client**: Axios
- **Styling**: Tailwind CSS
- **UI Components**: Custom components with Radix UI
- **State Management**: React Context API

#### Infrastructure
- **Backend Hosting**: Render.com
- **Frontend Hosting**: Vercel
- **Database**: MongoDB Atlas (Cloud)

---

## 📁 Project Structure

```
DocuBrain/
├── backend/                    # FastAPI Backend
│   ├── server.py              # Main FastAPI application
│   ├── database.py            # MongoDB operations
│   ├── lightweight_embeddings.py # TF-IDF embeddings engine
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── frontend/                  # React Frontend
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styles
│   │   └── index.js          # Entry point
│   ├── public/               # Static assets
│   ├── package.json          # Node.js dependencies
│   └── .env                  # Environment variables
├── contracts.md              # API documentation
└── PROJECT_DOCUMENTATION.md  # This file
```

---

## 🔧 Core Components Deep Dive

### 1. Backend Components

#### `server.py` - Main Application Server
**Purpose**: Central FastAPI application handling all API endpoints

**Key Functions**:
- `create_token(user_id)`: Generates simple authentication tokens
- `verify_token(token)`: Validates authentication tokens
- `extract_text_from_pdf(file_content)`: Extracts text from PDF files using PyPDF2
- `chunk_text(text, chunk_size=500)`: Splits text into chunks for processing
- `generate_answer_with_emergent_llm(question, context)`: Generates AI responses using Emergent Universal API

**API Endpoints**:
- `POST /api/auth/register`: User registration
- `POST /api/auth/login`: User authentication
- `POST /api/documents/upload`: PDF file upload and processing
- `POST /api/documents/text`: Text document creation
- `GET /api/documents`: Retrieve user's documents
- `POST /api/query`: Ask questions about documents
- `POST /api/external/query`: External API access with API key

#### `database.py` - MongoDB Operations
**Purpose**: Handles all database operations using MongoDB Motor (async driver)

**Key Functions**:
- `init_db()`: Establishes MongoDB connection
- `create_user(user_data)`: Creates new user records
- `get_user_by_username(username)`: Retrieves user by username
- `get_user_by_api_key(api_key)`: Retrieves user by API key
- `create_document(doc_data)`: Stores document data and embeddings
- `get_user_documents(user_id)`: Gets user's document list
- `get_user_documents_with_content(user_id)`: Gets documents with full content for querying

**Database Schema**:
```javascript
// Users Collection
{
  user_id: "uuid",
  username: "string",
  password: "string", // Plain text (simplified for demo)
  api_key: "sk-docubrain-xxxx",
  created_at: "ISO_datetime"
}

// Documents Collection
{
  id: "uuid",
  user_id: "uuid",
  filename: "string",
  content: "full_text_content",
  chunks: ["array", "of", "text", "chunks"],
  embeddings: [[float, array], [float, array]], // TF-IDF vectors
  upload_time: "ISO_datetime",
  chunk_count: integer,
  status: "completed"
}
```

#### `lightweight_embeddings.py` - Text Processing Engine
**Purpose**: Handles text embeddings and similarity search using TF-IDF

**Key Functions**:
- `get_embeddings_tfidf(texts)`: Generates TF-IDF embeddings for text chunks
- `get_query_embedding(query)`: Creates embedding for search queries
- `find_relevant_chunks(query, chunks, embeddings)`: Finds most similar text chunks
- `_simple_keyword_search(query, chunks)`: Fallback keyword-based search

**How It Works**:
1. **Text Vectorization**: Uses scikit-learn's TfidfVectorizer with 1000 max features
2. **Similarity Calculation**: Cosine similarity between query and document embeddings
3. **Ranking**: Returns top-k most relevant chunks with relevance scores
4. **Fallback**: Simple keyword matching if TF-IDF fails

### 2. Frontend Components

#### `App.js` - Main React Application
**Purpose**: Complete frontend application with routing and state management

**Key Components**:

**AuthProvider & AuthContext**:
- Manages user authentication state
- Handles login/logout operations
- Persists authentication in localStorage

**Login Component**:
- Dual-mode form (login/register)
- Form validation and error handling
- Success notifications for registration

**Dashboard Component**:
- Document management interface
- PDF and text upload functionality
- Question-answering interface
- API key display for external integrations

**ProtectedRoute Component**:
- Route protection for authenticated users
- Loading states during authentication check

---

## 🔄 System Workflow

### 1. User Registration & Authentication Flow
```
1. User visits application → Login page displayed
2. User clicks "Register" → Switches to registration form
3. User submits credentials → POST /api/auth/register
4. Backend validates → Creates user in MongoDB
5. Success response → User sees success popup
6. User clicks "Login" → Switches to login form
7. User submits credentials → POST /api/auth/login
8. Backend validates → Returns token and user data
9. Frontend stores token → Redirects to dashboard
```

### 2. Document Upload & Processing Flow

#### PDF Upload Process:
```
1. User selects PDF file → File validation (PDF only)
2. FormData created → POST /api/documents/upload
3. Backend receives file → PyPDF2 extracts text
4. Text chunked (500 chars) → TF-IDF embeddings generated
5. Document data saved → MongoDB stores everything
6. Success response → Frontend updates document list
```

#### Text Document Process:
```
1. User enters title/content → Form validation
2. FormData created → POST /api/documents/text
3. Backend processes text → Chunking and embeddings
4. Document saved → MongoDB storage
5. Success response → UI updated
```

### 3. Question-Answering Flow
```
1. User enters question → Form submission
2. POST /api/query → Backend retrieves user documents
3. For each document:
   a. Calculate query embedding
   b. Compare with document embeddings
   c. Find relevant chunks (cosine similarity)
4. Top 5 chunks selected → Context created
5. Emergent LLM called → AI generates answer
6. Response with answer and sources → Frontend displays result
```

### 4. External API Integration Flow
```
1. External system calls → POST /api/external/query
2. API key validation → User lookup by API key
3. Query processing → Same as internal query flow
4. JSON response → Answer and sources returned
```

---

## 🔐 Security & Authentication

### Authentication System
- **Simple Token-Based**: Uses `simple_token_{user_id}` format
- **Token Validation**: Middleware checks Bearer tokens
- **Session Management**: Frontend stores tokens in localStorage
- **API Key Access**: Each user gets unique API key for external access

### Security Considerations
- Plain text passwords (simplified for demo - should use hashing in production)
- CORS enabled for all origins (should be restricted in production)
- No rate limiting (should be added for production)
- Input validation on file types and content

---

## 🤖 AI Integration Details

### Emergent Universal API Integration
- **Model Used**: GPT-4o-mini (OpenAI)
- **Session Management**: Unique session ID per query
- **System Prompt**: Instructs AI to use only provided context
- **Error Handling**: Graceful fallback on API failures

### Prompt Engineering
```
System Message: "You are a helpful assistant that answers questions based on provided document context. Use only the provided information and be concise."

User Prompt Template:
"""
Based on the context below, answer the question concisely. Use only the provided information.

Context:
{relevant_document_chunks}

Question: {user_question}

Answer:
"""
```

---

## 📊 Performance & Scalability

### Current Implementation
- **TF-IDF Embeddings**: Lightweight, fast processing
- **In-Memory Vectorization**: Suitable for moderate document sizes
- **MongoDB**: Scalable document storage
- **Async Operations**: Non-blocking I/O operations

### Optimization Strategies
- **Chunking Strategy**: 500-character chunks for balanced context
- **Top-K Selection**: Returns top 5 most relevant chunks
- **Caching**: TF-IDF vectorizer fitted once per session
- **Fallback Mechanisms**: Simple keyword search if embeddings fail

### Scalability Considerations
- **Database Indexing**: Add indexes on user_id and filename
- **Caching Layer**: Redis for frequently accessed documents
- **Vector Database**: Consider Pinecone/Weaviate for large-scale deployments
- **Load Balancing**: Multiple backend instances for high traffic

---

## 🧪 Testing & Quality Assurance

### Backend Testing Coverage
- ✅ API endpoint functionality
- ✅ Database operations
- ✅ Authentication flow
- ✅ Document processing pipeline
- ✅ AI integration
- ✅ Error handling

### Frontend Testing Areas
- User authentication flow
- Document upload functionality
- Query submission and response handling
- Error state management
- Responsive design

---

## 🌍 Environment Configuration

### Backend Environment Variables (.env)
```bash
MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/
DATABASE_NAME=askmydocs
EMERGENT_LLM_KEY=sk-emergent-xxxxx
CORS_ORIGINS="*"
```

### Frontend Environment Variables (.env)
```bash
REACT_APP_BACKEND_URL=https://your-backend-url.com
```

---

## 🚀 Deployment Architecture

### Production Setup
- **Backend**: Render.com (Python web service)
- **Frontend**: Vercel (Static site deployment)
- **Database**: MongoDB Atlas (Cloud database)
- **CDN**: Vercel's global CDN for frontend assets

### CI/CD Pipeline
- GitHub repository integration
- Automatic deployments on push
- Environment variable management
- Health checks and monitoring

---

## 🔧 Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB Atlas account
- Emergent Universal API key

### Local Development
```bash
# Backend setup
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8001

# Frontend setup
cd frontend
yarn install
yarn start
```

---

## 📝 API Documentation

### Authentication Endpoints
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### Document Management
- `POST /api/documents/upload` - Upload PDF files
- `POST /api/documents/text` - Add text documents
- `GET /api/documents` - List user documents

### Query System
- `POST /api/query` - Ask questions (authenticated)
- `POST /api/external/query` - External API access

### Response Formats
```javascript
// Authentication Response
{
  "success": true,
  "user_id": "uuid",
  "token": "simple_token_uuid",
  "api_key": "sk-docubrain-xxxx"
}

// Query Response
{
  "answer": "AI generated answer",
  "sources": [
    {
      "filename": "document.pdf",
      "chunk_index": 0,
      "relevance_score": 0.85
    }
  ]
}
```

---

## 🐛 Troubleshooting Guide

### Common Issues
1. **MongoDB Connection Issues**: Check connection string and network access
2. **PDF Processing Errors**: Ensure valid PDF format and readable text
3. **AI API Failures**: Verify Emergent Universal API key and quotas
4. **CORS Errors**: Check frontend URL configuration
5. **Authentication Issues**: Verify token format and expiration

### Debug Steps
1. Check server logs for detailed error messages
2. Verify environment variables are loaded correctly
3. Test API endpoints individually
4. Validate database connectivity
5. Confirm AI API integration

---

## 🔮 Future Enhancements

### Planned Features
- **Advanced Security**: Password hashing, JWT tokens, rate limiting
- **Enhanced UI**: Drag-and-drop uploads, better document visualization
- **Advanced Search**: Semantic search, filters, search history
- **Multi-format Support**: Word documents, Excel files, images with OCR
- **Collaboration**: Shared documents, team workspaces
- **Analytics**: Usage statistics, query analytics
- **Mobile App**: React Native mobile application

### Technical Improvements
- **Vector Database**: Migration to specialized vector storage
- **Caching Layer**: Redis for performance optimization
- **Microservices**: Split into smaller, focused services
- **Real-time Updates**: WebSocket integration for live updates
- **Batch Processing**: Async document processing for large files

---

## 📞 Support & Maintenance

### Monitoring
- Application health checks
- Database performance monitoring
- AI API usage tracking
- Error logging and alerting

### Backup Strategy
- Automated MongoDB backups
- Code repository backups
- Configuration backup

This documentation provides a comprehensive understanding of the DocuBrain system for developers, maintainers, and stakeholders.