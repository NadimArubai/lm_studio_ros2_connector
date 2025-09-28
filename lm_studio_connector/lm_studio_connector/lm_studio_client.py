#!/usr/bin/env python3

import requests
import json
import logging
import base64
import mimetypes
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path

class LMStudioClient:
    """
    Client optimized for LM Studio API endpoints with image support and streaming
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
        }
        
        self.session = requests.Session()
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        self.session.headers.update(headers)
        self.logger = logging.getLogger('LMStudioClient')
    
    def prepare_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Prepare image for LM Studio - using base64 data URI format"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = 'image/jpeg'
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
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
        """Prepare image from base64 string using data URI format"""
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
                       timeout: int = 30) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Chat completion with streaming support
        Returns either a complete response or a stream iterator
        """
        endpoint = f"{self.base_url}{self.endpoints['chat']}"
        
        # Format messages for multimodal input
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg["role"]}
            
            if "images" in msg and msg["images"]:
                content = []
                for image in msg["images"]:
                    content.append(image)
                if "content" in msg and msg["content"]:
                    content.append({"type": "text", "text": msg["content"]})
                formatted_msg["content"] = content
            else:
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
            if stream:
                return self._handle_streaming_request(endpoint, payload, timeout, "chat")
            else:
                return self._handle_standard_request(endpoint, payload, timeout, "chat")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Chat completion request failed: {e}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response content: {e.response.text}")
            return self._mock_chat_completion(messages)
    
    def _handle_standard_request(self, endpoint: str, payload: dict, timeout: int, request_type: str) -> Dict[str, Any]:
        """Handle standard non-streaming request"""
        self.logger.info(f"Sending {request_type} request to {endpoint}")
        response = self.session.post(
            endpoint,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        self.logger.info(f"{request_type} request successful")
        return result
    
    def _handle_streaming_request(self, endpoint: str, payload: dict, timeout: int, request_type: str) -> Iterator[Dict[str, Any]]:
        """Handle streaming request"""
        self.logger.info(f"Sending streaming {request_type} request to {endpoint}")
        
        response = self.session.post(
            endpoint,
            json=payload,
            timeout=timeout,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse streaming chunk: {data}")
    
    def generate_text(self,
                     prompt: str,
                     model: str = "local-model",
                     max_tokens: int = 500,
                     temperature: float = 0.7,
                     stream: bool = False,
                     timeout: int = 30) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Text completion with streaming support
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
            if stream:
                return self._handle_streaming_request(endpoint, payload, timeout, "completion")
            else:
                return self._handle_standard_request(endpoint, payload, timeout, "completion")
            
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
    
    def _mock_chat_completion(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock response for chat completion with image support"""
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "Hello")
        
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
        
