#!/usr/bin/env python3

import requests
import json
import logging
import base64
import mimetypes
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class LMStudioClient:
    """
    Client optimized for LM Studio API endpoints with image support
    """
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
        # LM Studio specific endpoints
        self.endpoints = {
            'chat': '/v1/chat/completions',
            'completions': '/v1/completions',
            'models': '/v1/models',
            'embeddings': '/v1/embeddings',
#            'files': '/v1/files'  # For file uploads
        }
        
        self.session = requests.Session()
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        self.session.headers.update(headers)
        self.logger = logging.getLogger('LMStudioClient')
    
#    def upload_image(self, image_path: str) -> Optional[str]:
#        """
#        Upload image to LM Studio and return file ID
#        """
#        try:
#            endpoint = f"{self.base_url}{self.endpoints['files']}"
#            
#            # Read image file
#            with open(image_path, 'rb') as f:
#                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
#                
#                # Upload file
#                response = self.session.post(
#                    endpoint,
#                    files=files,
#                    timeout=self.timeout
#                )
#                response.raise_for_status()
#                
#                file_data = response.json()
#                file_id = file_data.get('id')
#                
#                self.logger.info(f"Uploaded image: {image_path}, file ID: {file_id}")
#                return file_id
#                
#        except Exception as e:
#            self.logger.error(f"Failed to upload image {image_path}: {e}")
#            return None
#    
    def prepare_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Prepare image for LM Studio - using base64 data URI format
        This is the format LM Studio typically expects
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = 'image/jpeg'  # default
            
            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create data URI format that LM Studio expects
            image_handle = {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:{mime_type};base64,{base64_image}"
                }
            }
            
            self.logger.info(f"Prepared image: {image_path}")
            return image_handle
            
        except Exception as e:
            self.logger.error(f"Failed to prepare image {image_path}: {e}")
            return None
    
    def prepare_image_base64(self, base64_string: str, mime_type: str = 'image/jpeg') -> Dict[str, Any]:
        """
        Prepare image from base64 string using data URI format
        """
        image_handle = {
            'type': 'image_url',
            'image_url': {
                'url': f"data:{mime_type};base64,{base64_string}"
            }
        }
        
        self.logger.info("Prepared image from base64 string")
        return image_handle
    
    def chat_completion(self,
                       messages: List[Dict[str, Any]],
                       model: str = "local-model",
                       max_tokens: int = 500,
                       temperature: float = 0.7,
                       stream: bool = False,
                       timeout: int = 30) -> Dict[str, Any]:
        """
        Chat completion using LM Studio's /v1/chat/completions endpoint
        Now supports images in messages using multimodal format
        """
        endpoint = f"{self.base_url}{self.endpoints['chat']}"
        
        # Format messages for multimodal input
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg["role"]}
            
            # Handle multimodal content (text + images)
            if "images" in msg and msg["images"]:
                content = []
                
                # Add images first
                for image in msg["images"]:
                    content.append(image)
                
                # Add text content if exists
                if "content" in msg and msg["content"]:
                    content.append({"type": "text", "text": msg["content"]})
                
                formatted_msg["content"] = content
            else:
                # Regular text message
                formatted_msg["content"] = msg["content"]
            
            formatted_messages.append(formatted_msg)
        
        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        self.logger.debug(f"Chat payload: {json.dumps(payload, indent=2)}")
        
        try:
            self.logger.info(f"Sending chat request to {endpoint}")
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Chat completion successful")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Chat completion request failed: {e}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response content: {e.response.text}")
            return self._mock_chat_completion(messages)
    
    def chat_with_image(self,
                       prompt: str,
                       image_handle: Dict[str, Any],
                       model: str = "local-model",
                       max_tokens: int = 500,
                       temperature: float = 0.7,
                       stream: bool = False,
                       timeout: int = 30) -> Dict[str, Any]:
        """
        Convenience method for chat with single image
        """
        message = {
            "role": "user",
            "content": prompt,
            "images": [image_handle]
        }
        
        return self.chat_completion(
            messages=[message],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            timeout=timeout
        )
        
    def generate_text(self,
                     prompt: str,
                     model: str = "local-model",
                     max_tokens: int = 500,
                     temperature: float = 0.7,
                     stream: bool = False,
                     timeout: int = 30) -> Dict[str, Any]:
        """
        Text completion using LM Studio's /v1/completions endpoint
        """
        endpoint = f"{self.base_url}{self.endpoints['completions']}"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        try:
            self.logger.info(f"Sending completion request to {endpoint}")
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Text generation successful")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Text generation request failed: {e}")
            return self._mock_text_generation(prompt)
    
    def list_models(self, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Get available models from LM Studio
        """
        endpoint = f"{self.base_url}{self.endpoints['models']}"
        
        try:
            self.logger.info(f"Fetching models from {endpoint}")
            response = self.session.get(endpoint, timeout=timeout)
            response.raise_for_status()
            models = response.json().get('data', [])
            self.logger.info(f"Found {len(models)} models")
            return models
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Models request failed: {e}")
            return []
    
    def get_embeddings(self,
                      input_text: str,
                      model: str = "local-model",
                      timeout: int = 30) -> List[float]:
        """
        Get text embeddings using LM Studio's /v1/embeddings endpoint
        """
        endpoint = f"{self.base_url}{self.endpoints['embeddings']}"
        
        payload = {
            "model": model,
            "input": input_text
        }
        
        try:
            self.logger.info(f"Getting embeddings from {endpoint}")
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            embeddings = result.get('data', [{}])[0].get('embedding', [])
            self.logger.info(f"Got embeddings of length {len(embeddings)}")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Embeddings request failed: {e}")
            return []
    
    def health_check(self) -> bool:
        """
        Health check using models endpoint (GET /v1/models)
        """
        endpoint = f"{self.base_url}{self.endpoints['models']}"
        
        try:
            self.logger.debug(f"Health check at {endpoint}")
            response = self.session.get(endpoint, timeout=5)
            
            # LM Studio returns 200 with models list if healthy
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and isinstance(data['data'], list):
                    self.logger.info(f"Health check successful, found {len(data['data'])} models")
                    return True
            return False
            
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Health check failed: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection and return detailed status
        """
        health_status = self.health_check()
        models = self.list_models()
        
        return {
            "base_url": self.base_url,
            "health_status": health_status,
            "available_models": len(models),
            "model_names": [model.get('id', 'unknown') for model in models[:3]],  # First 3 models
            "endpoints": self.endpoints
        }
    
    def _mock_chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Mock response for chat completion"""
        last_message = messages[-1]["content"] if messages else "Hello"
        response_text = f"Mock response to: '{last_message}'"
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(msg["content"].split()) for msg in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": sum(len(msg["content"].split()) for msg in messages) + len(response_text.split())
            }
        }
    
    def _mock_text_generation(self, prompt: str) -> Dict[str, Any]:
        """Mock response for text generation"""
        response_text = f"Mock completion for: '{prompt}'"
        
        return {
            "choices": [{
                "text": response_text,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        

    def _mock_chat_completion(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock response for chat completion with image support"""
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "Hello")
        
        # Check if there are images
        has_images = "images" in last_message and last_message["images"]
        
        if has_images:
            response_text = f"Mock response to image with prompt: '{content}'"
        else:
            response_text = f"Mock response to: '{content}'"
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(msg.get("content", "").split()) for msg in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": sum(len(msg.get("content", "").split()) for msg in messages) + len(response_text.split())
            }
        }

