from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

tavily_client = TavilyClient(api_key=f"{os.getenv('TAVILY_API_KEY')}")

@dataclass
class Message:
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

# helper
def dict_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: dict_to_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return dict_to_serializable(obj.__dict__)
    elif isinstance(obj, (list, tuple)):
        return [dict_to_serializable(i) for i in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
        
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: List[Message], max_tokens: int = 1000, **kwargs) -> tuple[str | dict, str, bool]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=max_tokens,
            **kwargs
        )
        if response.choices[0].message.tool_calls is None:
            return response.choices[0].message.content, response.usage.total_tokens, False
        else:
            return response.choices[0].message, response.usage.total_tokens, True
        
class Tool:
    def __init__(
        self,
        description: List[Dict[str, Any]],
        function: Callable
    ):
        self.description = description
        self.function = function

class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str,
        tool: Tool = None
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.tool = tool

    def generate(self, prompt: str, **kwargs) -> str:
        self.messages.append({"role": "user", "content": prompt})
        total_tokens = 0
        if self.tool:
            completion, tokens, is_tool_call = self.llm.generate(
                prompt=self.messages,
                tools=self.tool.description
            )
            total_tokens += tokens

            if is_tool_call:
                self.messages.append(completion)

                # Just pretty print the messages
                serializable_dict = dict_to_serializable(self.messages)
                print('tool call start: ')                
                print(json.dumps(serializable_dict, indent=2, ensure_ascii=False))

                if len(completion.tool_calls) > 1:
                    for tool_call in completion.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        result = self.tool.function(args["query"])
                        print('search result: ')
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                        self.messages.append({                             
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })
                    # Just pretty print the messages
                    serializable_dict = dict_to_serializable(self.messages)
                    print('multiple tool call: ')
                    print(json.dumps(serializable_dict, indent=2, ensure_ascii=False))
                else:
                    tool_call = completion.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    result = self.tool.function(args["query"])
                    print('search result: ')
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    self.messages.append({                             
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    # Just pretty print the messages
                    serializable_dict = dict_to_serializable(self.messages)
                    print('single tool call: ')
                    print(json.dumps(serializable_dict, indent=2, ensure_ascii=False))
                    
                completion_2, tokens, _ = self.llm.generate(
                    prompt=self.messages,
                )
                total_tokens += tokens

                return completion_2, total_tokens
            else:
                return completion, total_tokens
        else:
            completion, tokens, _ = self.llm.generate(
                prompt=self.messages                
            )
            total_tokens += tokens
            return completion, total_tokens
        
def search_tavily(query, max_results=3, **kwargs):
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        **kwargs        
    )
    return response['results']

def extract_tavily(urls, **kwargs):
    response = tavily_client.extract(
        urls=urls,
        **kwargs        
    )
    return response

search_tool = [{
    "type": "function",
    "function": {
        "name": "search_tavily",
        "description": "Search the web for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query to search the web for"
                }
            },
            "required": [
                "query"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]