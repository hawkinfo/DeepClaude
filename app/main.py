import os
from dotenv import load_dotenv
import asyncio
import json
from app.deepclaude.deepclaude import DeepClaude
from app.clients.gemini_client import GeminiClient
from app.clients.claude_client import ClaudeClient
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

async def main():
    # 初始化配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
    USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")
    USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"

    # 初始化DeepClaude
    if USE_GEMINI:
        assistant_client_class = GeminiClient
        assistant_api_url = f"projects/{GOOGLE_PROJECT_ID}/locations/{GOOGLE_LOCATION}"
    else:
        assistant_client_class = ClaudeClient
        assistant_api_url = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")

    deep_claude = DeepClaude(
        deepseek_api_key=DEEPSEEK_API_KEY,
        assistant_api_key="",  # Gemini不需要API key
        deepseek_api_url=DEEPSEEK_API_URL,
        assistant_api_url=f"projects/{GOOGLE_PROJECT_ID}/locations/{GOOGLE_LOCATION}",
        assistant_type="gemini",  # 明确指定使用Gemini
        is_openrouter=USE_OPENROUTER
    )

    print("DeepClaude 命令行交互模式已启动，输入'quit'退出")

    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ")
            if user_input.lower() == 'quit':
                print("退出对话")
                break

            # 构建消息
            messages = [{"role": "user", "content": user_input}]

            # 获取流式响应
            print("\n助手: ", end="", flush=True)
            async for chunk in deep_claude.chat_completions_with_stream(
                messages=messages,
                deepseek_model=DEEPSEEK_MODEL,
                assistant_model=CLAUDE_MODEL if not USE_GEMINI else "gemini-1.5-pro"
            ):
                # 解析并显示响应内容
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith('data: '):
                    data = chunk_str[6:]
                    if data == '[DONE]':
                        break
                    try:
                        json_data = json.loads(data)
                        content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

            print()  # 换行

        except KeyboardInterrupt:
            print("\n退出对话")
            break
        except Exception as e:
            logger.error(f"发生错误: {e}")
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())