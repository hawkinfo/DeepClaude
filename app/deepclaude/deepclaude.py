"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""
import json
import time
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient, GeminiClient


class DeepClaude:
    """处理 DeepSeek 和 Claude API 的流式输出衔接"""
    
    def __init__(self, deepseek_api_key: str, assistant_api_key: str, 
                 deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions", 
                 assistant_api_url: str = None,
                 assistant_type: str = "gemini",  # 默认使用Gemini
                 is_openrouter: bool = False):
        """初始化 API 客户端
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            assistant_api_key: Claude API密钥
        """
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
        
        # 根据类型初始化不同客户端
        if assistant_type == "claude":
            self.assistant_client = ClaudeClient(assistant_api_key, assistant_api_url, is_openrouter)
        else:  # 默认使用Gemini
            self.assistant_client = GeminiClient(assistant_api_key, assistant_api_url)
    
    async def chat_completions_with_stream(self, messages: list, 
                                         deepseek_model: str = "deepseek-reasoner",
                                         assistant_model: str = None) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程"""
        logger.info(f"开始处理聊天请求，消息数量: {len(messages)}")
        logger.debug(f"消息内容: {messages}")
        
        try:
            # 创建消息队列
            output_queue = asyncio.Queue()
            chat_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())
            
            async def process_deepseek():
                try:
                    logger.info("开始处理 DeepSeek 请求")
                    async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model):
                        if content_type == "content":
                            logger.debug(f"收到 DeepSeek 内容: {content}")
                            response = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": deepseek_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": content
                                    }
                                }]
                            }
                            await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                        elif content_type == "reasoning":
                            logger.debug(f"收到 DeepSeek 推理内容: {content}")
                except Exception as e:
                    logger.error(f"处理 DeepSeek 请求时发生错误: {e}")
                    raise
                finally:
                    await output_queue.put(None)
            
            async def process_assistant():
                try:
                    logger.info("开始处理 Assistant 请求")
                    async for content_type, content in self.assistant_client.stream_chat(messages, assistant_model):
                        if content_type == "answer":
                            logger.debug(f"收到 Assistant 回复: {content}")
                            response = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": assistant_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": content
                                    }
                                }]
                            }
                            await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                except Exception as e:
                    logger.error(f"处理 Assistant 请求时发生错误: {e}")
                    raise
                finally:
                    await output_queue.put(None)
            
            # 创建并发任务
            deepseek_task = asyncio.create_task(process_deepseek())
            assistant_task = asyncio.create_task(process_assistant())
            
            # 等待两个任务完成
            finished_tasks = 0
            while finished_tasks < 2:
                try:
                    item = await output_queue.get()
                    if item is None:
                        finished_tasks += 1
                        logger.debug(f"任务完成计数: {finished_tasks}")
                    else:
                        yield item
                except Exception as e:
                    logger.error(f"处理输出队列时发生错误: {e}")
                    break
            
            logger.info("流式处理完成")
            yield b'data: [DONE]\n\n'
            
        except Exception as e:
            logger.error(f"处理聊天请求时发生未捕获的异常: {e}")
            raise