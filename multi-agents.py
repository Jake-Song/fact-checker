from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"

@dataclass
class Message:
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

class Agent:
    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm_provider: LLMProvider,
        system_prompt: str
    ):
        self.name = name
        self.role = role
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.conversation_history: List[Message] = []

    async def process_message(self, message: Message) -> Message:
        # Add message to conversation history
        self.conversation_history.append(message)
        
        # Construct prompt from history
        prompt = self._construct_prompt()
        
        # Generate response using LLM
        response_content = await self.llm_provider.generate(prompt)
        
        # Create response message
        response = Message(
            role="assistant",
            content=response_content,
            metadata={"agent_name": self.name, "agent_role": self.role.value}
        )
        
        # Add response to history
        self.conversation_history.append(response)
        
        return response

    def _construct_prompt(self) -> str:
        prompt = f"{self.system_prompt}\n\n"
        for msg in self.conversation_history:
            prompt += f"{msg.role}: {msg.content}\n"
        return prompt

class AgentTeam:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.coordinator: Optional[Agent] = None

    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent
        if agent.role == AgentRole.COORDINATOR:
            self.coordinator = agent

    async def process_task(self, task: str) -> List[Message]:
        if not self.coordinator:
            raise ValueError("Team must have a coordinator agent")

        # Initialize task with coordinator
        initial_message = Message(role="user", content=task)
        coordinator_response = await self.coordinator.process_message(initial_message)

        responses = [coordinator_response]
        
        # Parse coordinator's response to determine next steps
        try:
            instructions = json.loads(coordinator_response.content)
            for step in instructions["steps"]:
                agent_name = step["agent"]
                agent_task = step["task"]
                
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    message = Message(role="user", content=agent_task)
                    response = await agent.process_message(message)
                    responses.append(response)
                    
                    # Send result back to coordinator
                    coord_message = Message(
                        role="user",
                        content=f"Result from {agent_name}: {response.content}"
                    )
                    coord_response = await self.coordinator.process_message(coord_message)
                    responses.append(coord_response)
        
        except json.JSONDecodeError:
            # Handle case where coordinator response isn't JSON
            pass

        return responses

# Example implementation of an LLM provider
class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

# Example usage
async def main():
    # Initialize providers
    openai_provider = OpenAIProvider()

    # Create agents with different roles
    coordinator = Agent(
        name="coordinator",
        role=AgentRole.COORDINATOR,
        llm_provider=openai_provider,
        system_prompt="You are the coordinator. Analyze tasks and delegate to specialized agents."
    )

    researcher = Agent(
        name="researcher",
        role=AgentRole.RESEARCHER,
        llm_provider=openai_provider,
        system_prompt="You are the researcher. Find and analyze relevant information."
    )

    planner = Agent(
        name="planner",
        role=AgentRole.PLANNER,
        llm_provider=openai_provider,
        system_prompt="You are the planner. Create detailed action plans."
    )

    # Create team
    team = AgentTeam()
    team.add_agent(coordinator)
    team.add_agent(researcher)
    team.add_agent(planner)

    # Process a task
    task = "Research and create a plan for implementing a new microservice architecture"
    responses = await team.process_task(task)

    # Handle responses
    for response in responses:
        print(f"{response.metadata['agent_name']}: {response.content}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())