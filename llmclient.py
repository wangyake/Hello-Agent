import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("MODEL_ID")
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
    
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """生成思考"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=temperature,
                stream=True
            )
            print(f"✔ 大语言模型响应成功：")
            collected_content = []
            for chunk in response:
                # choices是一个数组，表示多条候选答案，大部分情况只取第一个
                # delta是一个对象，表示当前响应的增量内容
                # 非流式用 response.choices[0].message.content
                # 流式用 chunk.choices[0].delta.content
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"❌ 调用LLM API时出错: {e}")
            return None

if __name__ == "__main__":
    try:
        llm = HelloAgentsLLM()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        response = llm.think(messages)
        if response:
            print("\n\n--- 完成模型响应 ---")
            print(response)
    except ValueError as e:
        print(e)