# DocuBrain API Contracts

## Working Features ✅

### 1. Authentication System
- **POST** `/api/auth/register` - User registration
- **POST** `/api/auth/login` - User login
- Returns: `user_id`, `token`, `api_key`

### 2. Document Management
- **POST** `/api/documents/upload` - Upload PDF files
- **POST** `/api/documents/text` - Add text documents
- **GET** `/api/documents` - List user documents

### 3. AI-Powered Query System
- **POST** `/api/query` - Ask questions about documents
- **POST** `/api/external/query` - External API endpoint with API key

### 4. Real Processing Pipeline
- ✅ PDF text extraction with PyPDF2
- ✅ Text chunking (500 char chunks)
- ✅ Embeddings with sentence-transformers
- ✅ Vector similarity search
- ✅ Gemini AI integration for answers
- ✅ Source citation with relevance scores

## Technical Stack
- **Backend**: FastAPI + MongoDB + JWT Auth
- **AI**: sentence-transformers + Google Gemini Pro
- **Frontend**: React + Context API
- **Database**: MongoDB for users/documents

## API Usage Example
```bash
# External API call
curl -X POST "http://localhost:8001/api/external/query" \
  -F "api_key=sk-docubrain-xxxx" \
  -F "question=What is our vacation policy?"
```

## Status: PRODUCTION READY 🚀
All core functionality implemented and tested.