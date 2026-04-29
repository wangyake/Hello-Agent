"""
Reason+Act
“思考-行动-观察”循环
智能体将不断重复这个 Thought -> Action -> Observation 的循环，将新的观察结果追加到历史记录中，
形成一个不断增长的上下文，直到它在Thought中认为已经找到了最终答案，
然后输出结果。这个过程形成了一个强大的协同效应：推理使得行动更具目的性，而行动则为推理提供了事实依据。

具体的，在每个时间步，根据初始问题 q 和 之前所有的“行动-观察”历史轨迹 ((a1, o1), ..., (at-1, ot-1))
生成当前的思考 tht 和 当前的行动 at。
随后 环境工具T会执行行动 at，返回结果ot = T(at)
智能体将使用一个预训练的模型 π 来生成当前的行动 at 和 观察 ot。

这种机制特别适用于以下场景：
  需要外部知识的任务：如查询实时信息（天气、新闻、股价）、搜索专业领域的知识等。
  需要精确计算的任务：将数学问题交给计算器工具，避免LLM的计算错误。
  需要与API交互的任务：如操作数据库、调用某个服务的API来完成特定功能。

因此我们将构建一个具备使用外部工具能力的ReAct智能体，来回答一个大语言模型仅凭自身知识库无法直接回答的问题。
例如：“华为最新的手机是哪一款？它的主要卖点是什么？” 
这个问题需要智能体理解自己需要上网搜索，调用工具搜索结果并总结答案。

一个良好定义的工具应包含以下三个核心要素：
名称 (Name)： 一个简洁、唯一的标识符，供智能体在 Action 中调用，例如 Search。
描述 (Description)： 一段清晰的自然语言描述，说明这个工具的用途。这是整个机制中最关键的部分，因为大语言模型会依赖这段描述来判断何时使用哪个工具。
执行逻辑 (Execution Logic)： 真正执行任务的函数或方法。
我们的第一个工具是 search 函数，它的作用是接收一个查询字符串，然后返回搜索结果
"""
from serpapi import SerpApiClient
import os
from dotenv import load_dotenv

# 加载 .env 文件，让直接运行时也能读取到环境变量
load_dotenv()

def search(query: str) -> str:
    """使用SerpAPI搜索查询字符串"""
    print(f'🔍 正在执行 [SerpApi] 网页搜索： {query}')
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        params = {
            'engine': 'google',
            'q': query,
            'api_key': api_key,
            'gl': 'cn',
            'hl': 'zh-CN',
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误：{e}"


"""
构建通用的工具执行器
"""
from typing import List, Dict, Any

class ToolExecutor:
    def __init__(self):
        self.tools : Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, description: str, func: callable):
        """注册一个工具"""
        if name in self.tools:
            print(f'工具 {name} 已注册，将被覆盖')
        self.tools[name] = {
            "description": description,
            "func": func,
        }
        print(f'工具 {name} 已注册')

    def getTool(self, name: str) -> callable:
        """获取一个工具"""
        return self.tools.get(name, {}).get("func")
    
    def getAvailableTools(self) -> str:
        """获取所有已注册的工具"""
        return "\n".join([
            f"{name}: {info['description']}"
            for name, info in self.tools.items()
        ])

# 测试
if __name__ == "__main__":
    executor = ToolExecutor()
    executor.register("Search", "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。", search)
    
    print("\n--- 可用工具 ---")
    print(executor.getAvailableTools())

    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "What's the latest GPU model from NVIDIA?"

    tool_function = executor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")