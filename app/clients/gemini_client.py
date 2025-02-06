"""Gemini Vertex AI 客户端"""
import json
import os
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient

class GeminiClient(BaseClient):
    def __init__(self, project_id: str, location: str, 
                 api_key: str = None,  # 保留API key用于兼容
                 api_url: str = None):
        self.project_id = project_id
        self.location = location
        self.api_url = api_url or f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/gemini-1.5-pro:streamGenerateContent"
        super().__init__(api_key or "", self.api_url)  # Vertex AI使用Bearer token认证

    async def _get_headers(self) -> dict:
        """使用gcloud CLI认证获取头信息"""
        from google.auth import default
        from google.auth.transport.requests import Request

        # 自动检测gcloud凭证
        credentials, _ = default()
        if not credentials.valid:
            credentials.refresh(Request())
        
        return {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }

    async def stream_chat(self, messages: list, model: str = "gemini-1.5-pro") -> AsyncGenerator[tuple[str, str], None]:
        # Vertex AI需要项目信息
        headers = await self._get_headers()
        
        # 转换消息格式
        vertex_messages = [{
            "role": "user" if msg["role"] == "assistant" else "model",
            "parts": [{"text": msg["content"]}]
        } for msg in messages]

        data = {
            "contents": vertex_messages,
            "generationConfig": {
                "maxOutputTokens": 8192
            }
        }
        
        async for chunk in self._make_request(headers, data):
            try:
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith("{"):
                    data = json.loads(chunk_str)
                    for candidate in data.get("candidates", []):
                        content = candidate.get("content", {})
                        for part in content.get("parts", []):
                            if text := part.get("text"):
                                yield "answer", text
            except Exception as e:
                logger.error(f"处理Vertex AI响应失败: {e}") 