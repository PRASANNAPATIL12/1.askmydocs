#!/usr/bin/env python3
"""
Direct Integration Testing for MongoDB and Emergent Universal API
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent / 'backend'))

async def test_mongodb_direct():
    """Test MongoDB connection and operations directly"""
    try:
        from database import db
        
        print("🔍 Testing MongoDB Integration...")
        
        # Test connection
        await db.init_db()
        print("✅ MongoDB connection successful")
        
        # Test user operations
        test_user = {
            "user_id": "test-integration-user",
            "username": "integration_test_user",
            "password": "testpass",
            "api_key": "sk-test-integration-key",
            "created_at": "2024-01-01T00:00:00"
        }
        
        # Create user
        success = await db.create_user(test_user)
        if success:
            print("✅ User creation in MongoDB successful")
        else:
            print("❌ User creation failed")
            return False
        
        # Retrieve user
        retrieved = await db.get_user_by_username(test_user["username"])
        if retrieved and retrieved["user_id"] == test_user["user_id"]:
            print("✅ User retrieval from MongoDB successful")
        else:
            print("❌ User retrieval failed")
            return False
        
        # Test document operations
        test_doc = {
            "id": "test-doc-id",
            "user_id": "test-integration-user",
            "filename": "test.txt",
            "content": "Test document content",
            "chunks": ["Test document content"],
            "embeddings": [[0.1, 0.2, 0.3]],
            "upload_time": "2024-01-01T00:00:00",
            "chunk_count": 1,
            "status": "completed"
        }
        
        # Create document
        success = await db.create_document(test_doc)
        if success:
            print("✅ Document creation in MongoDB successful")
        else:
            print("❌ Document creation failed")
            return False
        
        # Retrieve documents
        docs = await db.get_user_documents("test-integration-user")
        if docs and len(docs) > 0:
            print("✅ Document retrieval from MongoDB successful")
        else:
            print("❌ Document retrieval failed")
            return False
        
        print("🎉 MongoDB integration fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB integration failed: {str(e)}")
        return False

async def test_emergent_api_direct():
    """Test Emergent Universal API integration directly"""
    try:
        from server import generate_answer_with_openai
        
        print("\n🔍 Testing Emergent Universal API Integration...")
        
        # Test LLM response
        test_question = "What is machine learning?"
        test_context = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
        
        response = await generate_answer_with_openai(test_question, test_context)
        
        if response and not response.startswith("Error generating response"):
            print("✅ Emergent Universal API response successful")
            print(f"   Response: {response[:150]}...")
            
            # Check if response is relevant
            if any(keyword in response.lower() for keyword in ['machine learning', 'artificial intelligence', 'data', 'learn']):
                print("✅ Response content is relevant and accurate")
            else:
                print("⚠️  Response may not be fully relevant but API is working")
            
            return True
        else:
            print(f"❌ Emergent Universal API failed: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Emergent Universal API integration failed: {str(e)}")
        return False

async def test_embeddings_engine():
    """Test lightweight embeddings engine"""
    try:
        from lightweight_embeddings import embeddings_engine
        
        print("\n🔍 Testing Lightweight Embeddings Engine...")
        
        # Test TF-IDF embeddings
        test_texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing deals with text and speech"
        ]
        
        embeddings = embeddings_engine.get_embeddings_tfidf(test_texts)
        
        if embeddings and len(embeddings) == len(test_texts):
            print("✅ TF-IDF embeddings generation successful")
            print(f"   Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
        else:
            print("❌ TF-IDF embeddings generation failed")
            return False
        
        # Test query embedding and similarity search
        query = "What is artificial intelligence?"
        relevant_chunks = embeddings_engine.find_relevant_chunks(query, test_texts, embeddings)
        
        if relevant_chunks and len(relevant_chunks) > 0:
            print("✅ Similarity search working")
            print(f"   Found {len(relevant_chunks)} relevant chunks")
            for chunk in relevant_chunks:
                print(f"   - Score: {chunk['relevance_score']:.3f}, Content: {chunk['content'][:50]}...")
        else:
            print("❌ Similarity search failed")
            return False
        
        print("🎉 Embeddings engine fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings engine failed: {str(e)}")
        return False

async def main():
    """Run all integration tests"""
    print("🧠 DocuBrain - Direct Integration Testing")
    print("=" * 50)
    
    results = []
    
    # Test MongoDB
    mongodb_result = await test_mongodb_direct()
    results.append(("MongoDB Integration", mongodb_result))
    
    # Test Emergent API
    emergent_result = await test_emergent_api_direct()
    results.append(("Emergent Universal API", emergent_result))
    
    # Test Embeddings
    embeddings_result = await test_embeddings_engine()
    results.append(("Lightweight Embeddings", embeddings_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📈 Success Rate: {(passed/total*100):.1f}% ({passed}/{total})")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATIONS WORKING PERFECTLY!")
        print("✅ Ready for production use")
    else:
        print(f"\n⚠️  {total-passed} integration(s) need attention")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)