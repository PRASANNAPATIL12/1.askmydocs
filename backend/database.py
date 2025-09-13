import aiosqlite
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./docubrain.db')
DB_PATH = DATABASE_URL.replace('sqlite:///', '')

class Database:
    def __init__(self):
        self.db_path = DB_PATH
        
    async def init_db(self):
        """Initialize database with required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Users table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Documents table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunks TEXT NOT NULL,
                    embeddings TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            await db.commit()
    
    async def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a new user"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO users (user_id, username, password, api_key, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_data['user_id'],
                    user_data['username'],
                    user_data['password'],
                    user_data['api_key'],
                    user_data['created_at'].isoformat()
                ))
                await db.commit()
                return True
        except aiosqlite.IntegrityError:
            return False
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT user_id, username, password, api_key, created_at
                FROM users WHERE username = ?
            ''', (username,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        'user_id': row[0],
                        'username': row[1], 
                        'password': row[2],
                        'api_key': row[3],
                        'created_at': row[4]
                    }
                return None
    
    async def get_user_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get user by API key"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT user_id, username, password, api_key, created_at
                FROM users WHERE api_key = ?
            ''', (api_key,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        'user_id': row[0],
                        'username': row[1],
                        'password': row[2],
                        'api_key': row[3],
                        'created_at': row[4]
                    }
                return None
    
    async def create_document(self, doc_data: Dict[str, Any]) -> bool:
        """Create a new document"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO documents (id, user_id, filename, content, chunks, embeddings, upload_time, chunk_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_data['id'],
                    doc_data['user_id'],
                    doc_data['filename'],
                    doc_data['content'],
                    json.dumps(doc_data['chunks']),
                    json.dumps(doc_data['embeddings']),
                    doc_data['upload_time'].isoformat(),
                    doc_data['chunk_count'],
                    doc_data['status']
                ))
                await db.commit()
                return True
        except Exception:
            return False
    
    async def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT id, filename, upload_time, chunk_count, status
                FROM documents WHERE user_id = ?
                ORDER BY upload_time DESC
            ''', (user_id,)) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'filename': row[1],
                        'upload_time': row[2],
                        'chunk_count': row[3],
                        'status': row[4]
                    }
                    for row in rows
                ]
    
    async def get_user_documents_with_content(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user with full content for querying"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT id, filename, content, chunks, embeddings
                FROM documents WHERE user_id = ?
            ''', (user_id,)) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'filename': row[1],
                        'content': row[2],
                        'chunks': json.loads(row[3]),
                        'embeddings': json.loads(row[4])
                    }
                    for row in rows
                ]

# Global database instance
db = Database()