import motor.motor_asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

MONGO_URL = os.environ.get('MONGO_URL')
DATABASE_NAME = os.environ.get('DATABASE_NAME', 'askmydocs')

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        
    async def init_db(self):
        """Initialize MongoDB connection"""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
            self.db = self.client[DATABASE_NAME]
            # Test the connection
            await self.client.admin.command('ping')
            print(f"✅ Successfully connected to MongoDB database: {DATABASE_NAME}")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise e
    
    async def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a new user"""
        try:
            # Convert datetime to ISO string for MongoDB
            if isinstance(user_data.get('created_at'), datetime):
                user_data['created_at'] = user_data['created_at'].isoformat()
            
            result = await self.db.users.insert_one(user_data)
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            user = await self.db.users.find_one({"username": username})
            if user:
                # Remove MongoDB _id field for consistency
                user.pop('_id', None)
            return user
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
    
    async def get_user_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get user by API key"""
        try:
            user = await self.db.users.find_one({"api_key": api_key})
            if user:
                # Remove MongoDB _id field for consistency
                user.pop('_id', None)
            return user
        except Exception as e:
            print(f"Error getting user by API key: {e}")
            return None
    
    async def create_document(self, doc_data: Dict[str, Any]) -> bool:
        """Create a new document"""
        try:
            # Convert datetime to ISO string for MongoDB
            if isinstance(doc_data.get('upload_time'), datetime):
                doc_data['upload_time'] = doc_data['upload_time'].isoformat()
            
            result = await self.db.documents.insert_one(doc_data)
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error creating document: {e}")
            return False
    
    async def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        try:
            cursor = self.db.documents.find(
                {"user_id": user_id},
                {"id": 1, "filename": 1, "upload_time": 1, "chunk_count": 1, "status": 1, "_id": 0}
            ).sort("upload_time", -1)
            
            documents = await cursor.to_list(length=None)
            return documents
        except Exception as e:
            print(f"Error getting user documents: {e}")
            return []
    
    async def get_user_documents_with_content(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user with full content for querying"""
        try:
            cursor = self.db.documents.find(
                {"user_id": user_id},
                {"id": 1, "filename": 1, "content": 1, "chunks": 1, "embeddings": 1, "_id": 0}
            )
            
            documents = await cursor.to_list(length=None)
            return documents
        except Exception as e:
            print(f"Error getting user documents with content: {e}")
            return []

# Global database instance
db = Database()