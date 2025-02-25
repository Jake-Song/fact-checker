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

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: List[Message], **kwargs) -> tuple[str | dict, bool]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            **kwargs
        )
        if response.choices[0].message.tool_calls is None:
            return response.choices[0].message.content, False
        else:
            return response.choices[0].message, True
        
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

        if self.tool:
            completion, is_tool_call = self.llm.generate(
                prompt=self.messages,
                tools=self.tool.description
            )
            
            if is_tool_call:
                self.messages.append(completion)
                print('tool call start: ', self.messages)

                if len(completion.tool_calls) > 1:
                    for tool_call in completion.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        result = self.tool.function(args["query"])
                        print('search result: ', result)
                        self.messages.append({                             
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })
                    print('multiple tool call: ', self.messages)
                else:
                    tool_call = completion.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    result = self.tool.function(args["query"])
                    print('search result: ', result)
                    self.messages.append({                             
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    print('single tool call: ', self.messages)
                    
                completion_2, _ = self.llm.generate(
                    prompt=self.messages,
                )

                return completion_2
            else:
                return completion
        else:
            completion, _ = self.llm.generate(
                prompt=self.messages                
            )
            return completion
        
def search_tavily(query, max_results=2, **kwargs):
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