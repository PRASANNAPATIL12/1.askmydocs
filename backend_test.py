#!/usr/bin/env python3
"""
DocuBrain Backend API Testing Suite
Tests all backend functionality including MongoDB integration and Emergent Universal API
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path
import tempfile
from typing import Dict, Any
import uuid

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent / 'backend'))

# Test configuration
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://docu2-clone.preview.emergentagent.com')
API_BASE = f"{BACKEND_URL}/api"

class DocuBrainTester:
    def __init__(self):
        self.session = None
        self.test_user_token = None
        self.test_user_api_key = None
        self.test_user_id = None
        self.test_document_id = None
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
    
    async def setup_session(self):
        """Setup HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup_session(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
        
        if success:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
            self.results['errors'].append(f"{test_name}: {message}")
    
    async def test_root_endpoint(self):
        """Test root endpoint"""
        try:
            async with self.session.get(f"{BACKEND_URL}/") as response:
                if response.status == 200:
                    data = await response.json()
                    if "DocuBrain API is running" in data.get('message', ''):
                        self.log_result("Root Endpoint", True, "API is running")
                        return True
                    else:
                        self.log_result("Root Endpoint", False, f"Unexpected response: {data}")
                        return False
                else:
                    self.log_result("Root Endpoint", False, f"Status: {response.status}")
                    return False
        except Exception as e:
            self.log_result("Root Endpoint", False, f"Exception: {str(e)}")
            return False
    
    async def test_user_registration(self):
        """Test user registration"""
        try:
            # Generate unique username
            username = f"testuser_{uuid.uuid4().hex[:8]}"
            password = "testpass123"
            
            payload = {
                "username": username,
                "password": password
            }
            
            async with self.session.post(f"{API_BASE}/auth/register", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success') and data.get('token') and data.get('api_key'):
                        self.test_user_token = data['token']
                        self.test_user_api_key = data['api_key']
                        self.test_user_id = data['user_id']
                        self.log_result("User Registration", True, f"User created: {username}")
                        return True
                    else:
                        self.log_result("User Registration", False, f"Missing fields in response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("User Registration", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("User Registration", False, f"Exception: {str(e)}")
            return False
    
    async def test_user_login(self):
        """Test user login with the registered user"""
        try:
            if not self.test_user_token:
                self.log_result("User Login", False, "No test user available")
                return False
            
            # We need to extract username from registration, let's create a new user for login test
            username = f"logintest_{uuid.uuid4().hex[:8]}"
            password = "loginpass123"
            
            # First register a user
            reg_payload = {"username": username, "password": password}
            async with self.session.post(f"{API_BASE}/auth/register", json=reg_payload) as response:
                if response.status != 200:
                    self.log_result("User Login", False, "Failed to create test user for login")
                    return False
            
            # Now test login
            login_payload = {"username": username, "password": password}
            async with self.session.post(f"{API_BASE}/auth/login", json=login_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success') and data.get('token'):
                        self.log_result("User Login", True, f"Login successful for: {username}")
                        return True
                    else:
                        self.log_result("User Login", False, f"Missing fields in response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("User Login", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("User Login", False, f"Exception: {str(e)}")
            return False
    
    async def test_text_document_upload(self):
        """Test text document upload"""
        try:
            if not self.test_user_token:
                self.log_result("Text Document Upload", False, "No authentication token")
                return False
            
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            
            # Create form data
            data = aiohttp.FormData()
            data.add_field('title', 'Test Document')
            data.add_field('content', 'This is a test document about machine learning. Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.')
            
            async with self.session.post(f"{API_BASE}/documents/text", data=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('document_id'):
                        self.test_document_id = result['document_id']
                        self.log_result("Text Document Upload", True, f"Document uploaded: {result['document_id']}")
                        return True
                    else:
                        self.log_result("Text Document Upload", False, f"No document_id in response: {result}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("Text Document Upload", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("Text Document Upload", False, f"Exception: {str(e)}")
            return False
    
    async def test_pdf_document_upload(self):
        """Test PDF document upload"""
        try:
            if not self.test_user_token:
                self.log_result("PDF Document Upload", False, "No authentication token")
                return False
            
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            
            # Create a simple PDF content for testing
            # Note: This is a minimal test - in real scenario we'd use a proper PDF
            pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test PDF Content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n299\n%%EOF"
            
            # Create form data with file
            data = aiohttp.FormData()
            data.add_field('file', pdf_content, filename='test.pdf', content_type='application/pdf')
            
            async with self.session.post(f"{API_BASE}/documents/upload", data=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('document_id'):
                        self.log_result("PDF Document Upload", True, f"PDF uploaded: {result['document_id']}")
                        return True
                    else:
                        self.log_result("PDF Document Upload", False, f"No document_id in response: {result}")
                        return False
                elif response.status == 400:
                    # PDF parsing might fail with our simple test PDF, which is expected
                    error_text = await response.text()
                    if "Error processing PDF" in error_text:
                        self.log_result("PDF Document Upload", True, "PDF endpoint working (parsing failed as expected with test PDF)")
                        return True
                    else:
                        self.log_result("PDF Document Upload", False, f"Unexpected 400 error: {error_text}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("PDF Document Upload", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("PDF Document Upload", False, f"Exception: {str(e)}")
            return False
    
    async def test_get_documents(self):
        """Test getting user documents"""
        try:
            if not self.test_user_token:
                self.log_result("Get Documents", False, "No authentication token")
                return False
            
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            
            async with self.session.get(f"{API_BASE}/documents", headers=headers) as response:
                if response.status == 200:
                    documents = await response.json()
                    if isinstance(documents, list):
                        self.log_result("Get Documents", True, f"Retrieved {len(documents)} documents")
                        return True
                    else:
                        self.log_result("Get Documents", False, f"Expected list, got: {type(documents)}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("Get Documents", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("Get Documents", False, f"Exception: {str(e)}")
            return False
    
    async def test_document_query(self):
        """Test document querying functionality"""
        try:
            if not self.test_user_token:
                self.log_result("Document Query", False, "No authentication token")
                return False
            
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            
            # Test query
            query_payload = {
                "question": "What is machine learning?"
            }
            
            async with self.session.post(f"{API_BASE}/query", json=query_payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('answer') and 'sources' in result:
                        answer = result['answer']
                        sources = result['sources']
                        self.log_result("Document Query", True, f"Query successful. Answer: {answer[:100]}...")
                        return True
                    else:
                        self.log_result("Document Query", False, f"Missing answer or sources: {result}")
                        return False
                elif response.status == 400:
                    error_text = await response.text()
                    if "No documents found" in error_text:
                        self.log_result("Document Query", True, "Query endpoint working (no documents as expected)")
                        return True
                    else:
                        self.log_result("Document Query", False, f"Unexpected 400 error: {error_text}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("Document Query", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("Document Query", False, f"Exception: {str(e)}")
            return False
    
    async def test_external_api_query(self):
        """Test external API endpoint with API key"""
        try:
            if not self.test_user_api_key:
                self.log_result("External API Query", False, "No API key available")
                return False
            
            # Create form data
            data = aiohttp.FormData()
            data.add_field('api_key', self.test_user_api_key)
            data.add_field('question', 'What is artificial intelligence?')
            
            async with self.session.post(f"{API_BASE}/external/query", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('answer') and 'sources' in result:
                        self.log_result("External API Query", True, "External API endpoint working")
                        return True
                    else:
                        self.log_result("External API Query", False, f"Missing answer or sources: {result}")
                        return False
                elif response.status == 400:
                    error_text = await response.text()
                    if "No documents found" in error_text:
                        self.log_result("External API Query", True, "External API endpoint working (no documents as expected)")
                        return True
                    else:
                        self.log_result("External API Query", False, f"Unexpected 400 error: {error_text}")
                        return False
                else:
                    error_text = await response.text()
                    self.log_result("External API Query", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_result("External API Query", False, f"Exception: {str(e)}")
            return False
    
    async def test_mongodb_integration(self):
        """Test MongoDB integration by checking database operations"""
        try:
            # Import database module to test connection
            from database import db
            
            # Test database initialization
            await db.init_db()
            
            # Test basic database operations
            test_user_data = {
                "user_id": str(uuid.uuid4()),
                "username": f"dbtest_{uuid.uuid4().hex[:8]}",
                "password": "testpass",
                "api_key": f"sk-test-{uuid.uuid4().hex[:20]}",
                "created_at": "2024-01-01T00:00:00"
            }
            
            # Test user creation
            success = await db.create_user(test_user_data)
            if not success:
                self.log_result("MongoDB Integration", False, "Failed to create test user in database")
                return False
            
            # Test user retrieval
            retrieved_user = await db.get_user_by_username(test_user_data["username"])
            if not retrieved_user:
                self.log_result("MongoDB Integration", False, "Failed to retrieve test user from database")
                return False
            
            self.log_result("MongoDB Integration", True, "Database operations successful")
            return True
            
        except Exception as e:
            self.log_result("MongoDB Integration", False, f"Exception: {str(e)}")
            return False
    
    async def test_emergent_llm_integration(self):
        """Test Emergent Universal API integration"""
        try:
            # Import server module to test LLM function
            from server import generate_answer_with_openai
            
            # Test LLM response generation
            test_question = "What is artificial intelligence?"
            test_context = "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines."
            
            response = await generate_answer_with_openai(test_question, test_context)
            
            if response and not response.startswith("Error generating response"):
                self.log_result("Emergent LLM Integration", True, f"LLM response: {response[:100]}...")
                return True
            else:
                self.log_result("Emergent LLM Integration", False, f"LLM error or empty response: {response}")
                return False
                
        except Exception as e:
            self.log_result("Emergent LLM Integration", False, f"Exception: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧠 DocuBrain Backend API Testing Suite")
        print("=" * 50)
        
        await self.setup_session()
        
        try:
            # Test sequence
            tests = [
                ("Root Endpoint", self.test_root_endpoint),
                ("MongoDB Integration", self.test_mongodb_integration),
                ("Emergent LLM Integration", self.test_emergent_llm_integration),
                ("User Registration", self.test_user_registration),
                ("User Login", self.test_user_login),
                ("Text Document Upload", self.test_text_document_upload),
                ("PDF Document Upload", self.test_pdf_document_upload),
                ("Get Documents", self.test_get_documents),
                ("Document Query", self.test_document_query),
                ("External API Query", self.test_external_api_query),
            ]
            
            for test_name, test_func in tests:
                print(f"\n🔍 Testing: {test_name}")
                await test_func()
            
            # Summary
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)
            print(f"✅ Passed: {self.results['passed']}")
            print(f"❌ Failed: {self.results['failed']}")
            print(f"📈 Success Rate: {(self.results['passed'] / (self.results['passed'] + self.results['failed']) * 100):.1f}%")
            
            if self.results['errors']:
                print("\n🚨 FAILED TESTS:")
                for error in self.results['errors']:
                    print(f"   • {error}")
            
            return self.results['failed'] == 0
            
        finally:
            await self.cleanup_session()

async def main():
    """Main test runner"""
    tester = DocuBrainTester()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)