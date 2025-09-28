#!/usr/bin/env python3

import sys
import os
import time
import json
from typing import Dict, Any, List

# Add the parent directory to path to import the client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lm_studio_client import LMStudioClient

def test_client_connection():
    """Test basic client connection"""
    print("=== Testing LM Studio Client Connection ===")
    
    # Initialize client
    client = LMStudioClient("http://localhost:1234")
    
    # Test connection
    status = client.test_connection()
    print(f"Connection Status: {json.dumps(status, indent=2)}")
    
    return status["health_status"]

def test_list_models():
    """Test listing available models"""
    print("\n=== Testing Model Listing ===")
    
    client = LMStudioClient("http://localhost:1234")
    models = client.list_models()
    
    print(f"Found {len(models)} models:")
    for i, model in enumerate(models[:5]):  # Show first 5 models
        print(f"  {i+1}. {model.get('id', 'unknown')}")
    
    return len(models) > 0

def test_chat_completion():
    """Test chat completion"""
    print("\n=== Testing Chat Completion ===")
    
    client = LMStudioClient("http://localhost:1234")
    
    messages = [
        {"role": "user", "content": "Hello! What can you tell me about artificial intelligence?"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            model="local-model",
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            reply = response['choices'][0]['message']['content']
            print(f"Response: {reply}")
            return True
        else:
            print("No response generated")
            return False
            
    except Exception as e:
        print(f"Chat completion failed: {e}")
        return False

def test_streaming_chat():
    """Test streaming chat completion"""
    print("\n=== Testing Streaming Chat ===")
    
    client = LMStudioClient("http://localhost:1234")
    
    messages = [
        {"role": "user", "content": "Explain machine learning in one sentence."}
    ]
    
    try:
        stream = client.chat_completion(
            messages=messages,
            model="local-model",
            max_tokens=50,
            temperature=0.7,
            stream=True
        )
        
        print("Streaming response:")
        full_response = ""
        for chunk in stream:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    token = delta['content']
                    print(token, end='', flush=True)
                    full_response += token
        
        print(f"\nFull response: {full_response}")
        return True
        
    except Exception as e:
        print(f"Streaming chat failed: {e}")
        return False

def test_text_completion():
    """Test text completion"""
    print("\n=== Testing Text Completion ===")
    
    client = LMStudioClient("http://localhost:1234")
    
    prompt = "The future of robotics is"
    
    try:
        response = client.generate_text(
            prompt=prompt,
            model="local-model",
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            completion = response['choices'][0]['text']
            print(f"Completion: {completion}")
            return True
        else:
            print("No completion generated")
            return False
            
    except Exception as e:
        print(f"Text completion failed: {e}")
        return False

def test_embeddings():
    """Test embeddings generation"""
    print("\n=== Testing Embeddings ===")
    
    client = LMStudioClient("http://localhost:1234")
    
    text = "This is a test sentence for embeddings."
    
    try:
        embeddings = client.get_embeddings(text)
        
        print(f"Generated embeddings with {len(embeddings)} dimensions")
        print(f"First 10 dimensions: {embeddings[:10]}")
        return len(embeddings) > 0
        
    except Exception as e:
        print(f"Embeddings generation failed: {e}")
        return False

def test_image_preparation():
    """Test image preparation functionality"""
    print("\n=== Testing Image Preparation ===")
    
    client = LMStudioClient("http://localhost:1234")
    
    # Create a test image file
    test_image_path = "test_image.jpg"
    
    try:
        # Try to prepare a non-existent image (should handle gracefully)
        image_handle = client.prepare_image("non_existent_image.jpg")
        if image_handle is None:
            print("✓ Correctly handled non-existent image")
        
        # Test base64 image preparation
        test_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        image_handle = client.prepare_image_base64(test_base64)
        
        if image_handle and 'image_url' in image_handle:
            print("✓ Base64 image preparation successful")
            return True
        else:
            print("✗ Base64 image preparation failed")
            return False
            
    except Exception as e:
        print(f"Image preparation test failed: {e}")
        return False

def run_all_tests():
    """Run all client tests"""
    print("Starting LM Studio Client Tests...")
    
    tests = [
        ("Client Connection", test_client_connection),
        ("Model Listing", test_list_models),
        ("Chat Completion", test_chat_completion),
        ("Streaming Chat", test_streaming_chat),
        ("Text Completion", test_text_completion),
        ("Embeddings", test_embeddings),
        ("Image Preparation", test_image_preparation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status}: {test_name}\n")
        except Exception as e:
            print(f"✗ ERROR: {test_name} - {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=== TEST SUMMARY ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    run_all_tests()
