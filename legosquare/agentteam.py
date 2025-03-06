import asyncio
import uuid
import logging
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
import aiohttp
from legosquare.lib import Message, TaskGraph, KnowledgeBase
from legosquare.agent import Agent
from legosquare.market import DecentralizedTaskMarket

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentTeam:
    """A team of collaborative agents working together"""
    def __init__(self, team_id: str, name: str, ollama_base_url: str = "http://localhost:11434"):
        self.id = team_id
        self.name = name
        self.agents = {}  # agent_id -> Agent mapping
        self.task_market = DecentralizedTaskMarket()
        self.message_queue = asyncio.Queue()  # Central message queue
        self.team_knowledge = KnowledgeBase()  # Shared knowledge
        self.running = False
        self.performance_metrics = {
            "tasks_completed": 0,
            "total_tasks": 0,
            "average_completion_time": 0,
            "messages_exchanged": 0
        }
        self.ollama_base_url = ollama_base_url
        self.ollama_model = "llama3.2:latest"
    
    async def add_agent(self, agent: Agent, subscribe_to: List[str] = None) -> None:
        """Add an agent to the team"""
        self.agents[agent.id] = agent
        await agent.connect_to_task_market(self.task_market, subscribe_to or ["general"])
    
    async def start(self) -> None:
        """Start the team's operation"""
        self.running = True
        
        # Start message processing
        asyncio.create_task(self._process_messages())
        
        logger.info(f"Team {self.name} ({self.id}) started with {len(self.agents)} agents")
    
    async def stop(self) -> None:
        """Stop the team's operation"""
        self.running = False
        logger.info(f"Team {self.name} ({self.id}) stopped")
    
    async def submit_task(self, description: str, complexity: float = 0.5, priority: int = 1) -> str:
        """Submit a new task to the team"""
        task_id = str(uuid.uuid4())
        
        # Publish to task market
        await self.task_market.publish_task(
            task_id=task_id,
            description=description,
            task_type="general",
            complexity=complexity,
            priority=priority
        )
        
        self.performance_metrics["total_tasks"] += 1
        logger.info(f"Task {task_id} submitted to team {self.name}: {description}")
        
        return task_id
    
    async def submit_complex_task(self, description: str) -> str:
        """Submit a complex task that will be broken down into subtasks"""
        # Create a task graph for the complex task
        root_task_id = str(uuid.uuid4())
        task_graph = await self._create_task_graph(root_task_id, description)
        
        # Publish to task market
        graph_id = await self.task_market.publish_task_graph(task_graph)
        
        self.performance_metrics["total_tasks"] += len(task_graph.nodes)
        logger.info(f"Complex task {root_task_id} submitted to team {self.name} with {len(task_graph.nodes)} subtasks")
        
        return graph_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task"""
        if task_id in self.task_market.available_tasks:
            return self.task_market.available_tasks[task_id]
        
        # Check if it's a task graph
        if task_id in self.task_market.task_graphs:
            return await self.task_market.get_task_graph_status(task_id)
        
        return {"error": "Task not found"}
    
    async def send_message_to_team(self, content: str, sender: str = "User") -> None:
        """Send a message to all agents in the team"""
        message = Message(content=content, sender_id=sender)
        await self.message_queue.put(message)
        
        logger.info(f"Message from {sender} to team {self.name}: {content}")
    
    async def get_team_performance(self) -> Dict[str, Any]:
        """Get team performance metrics"""
        # Update with agent-specific metrics
        agent_metrics = {}
        for agent_id, agent in self.agents.items():
            agent_metrics[agent_id] = {
                "tasks_completed": agent.metrics["tasks_completed"],
                "messages_sent": agent.metrics["messages_sent"],
                "messages_received": agent.metrics["messages_received"],
                "average_response_time": agent.metrics["average_response_time"]
            }
        
        return {
            "team_metrics": self.performance_metrics,
            "agent_metrics": agent_metrics
        }
    
    async def _process_messages(self) -> None:
        """Process messages in the queue"""
        while self.running:
            try:
                message = await self.message_queue.get()
                
                # If the message has a specific recipient, deliver it directly
                if message.recipient_id and message.recipient_id in self.agents:
                    response = await self.agents[message.recipient_id].receive_message(message)
                    if response:
                        await self.message_queue.put(response)
                    
                # Otherwise, broadcast to all agents
                else:
                    for agent_id, agent in self.agents.items():
                        if agent_id != message.sender_id:
                            response = await agent.receive_message(message)
                            if response:
                                await self.message_queue.put(response)
                
                self.performance_metrics["messages_exchanged"] += 1
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _create_task_graph(self, root_task_id: str, description: str) -> TaskGraph:
        """Create a task graph by decomposing a complex task using an LLM."""
        # Create the task graph with the root task
        task_graph = TaskGraph(root_task_id, description)
        
        # Use LLM to decompose the task
        subtasks_with_deps = await self._decompose_task_with_llm(description)
        
        # Process the decomposition and build the graph
        for subtask in subtasks_with_deps:
            subtask_id = str(uuid.uuid4())
            task_graph.add_node(subtask_id, subtask["description"])
            
            # Store the subtask ID for dependency mapping
            subtask["id"] = subtask_id
        
        # Add dependencies after all nodes are created
        for subtask in subtasks_with_deps:
            for dep_index in subtask.get("depends_on", []):
                if 0 <= dep_index < len(subtasks_with_deps):
                    dep_subtask = subtasks_with_deps[dep_index]
                    task_graph.add_dependency(subtask["id"], dep_subtask["id"])
        
        return task_graph
    
    async def _decompose_task_with_llm(self, description: str) -> List[Dict[str, Any]]:
        """Use Ollama to decompose a task into subtasks with dependencies."""
        prompt = f"""
        You are an expert task planner. You need to decompose this high-level task into smaller subtasks:
        
        TASK: {description}
        
        Break this task down into 3-7 subtasks. For each subtask:
        1. Provide a clear, actionable description
        2. Specify which other subtasks (if any) it depends on
        
        Return your answer as a JSON array where each object has:
        - "description": the subtask description (string)
        - "depends_on": array of indices of prerequisite subtasks (empty if none)
        
        Example:
        [
            {{"description": "Research existing solutions", "depends_on": []}},
            {{"description": "Define requirements specification", "depends_on": [0]}},
            {{"description": "Implement core functionality", "depends_on": [1]}}
        ]
        
        Ensure the dependency structure is valid (no cycles) and that tasks are ordered from prerequisites to dependent tasks.
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Ollama API: {error_text}")
                    
                    result = await response.json()
                    llm_response = result.get("response", "")
            
            # Extract JSON array from response
            # First try to find JSON between triple backticks
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find just a JSON array
                json_match = re.search(r"\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]", llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback to using the whole response
                    json_str = llm_response
            
            # Parse the JSON response
            subtasks = json.loads(json_str)
            
            # Validate the structure
            for subtask in subtasks:
                if "description" not in subtask:
                    subtask["description"] = "Undefined subtask"
                if "depends_on" not in subtask:
                    subtask["depends_on"] = []
            
            return subtasks
            
        except json.JSONDecodeError:
            # If LLM response doesn't parse as JSON, create a fallback decomposition
            print(f"Failed to parse LLM response as JSON. Using fallback decomposition.")
            print(f"LLM response: {llm_response}")
            return [
                {"description": f"Analyze requirements for: {description}", "depends_on": []},
                {"description": f"Generate solution options for: {description}", "depends_on": [0]},
                {"description": f"Implement selected solution for: {description}", "depends_on": [1]}
            ]
        except Exception as e:
            print(f"Error using LLM for task decomposition: {str(e)}")
            # Fallback to a basic decomposition if the LLM call fails
            return [
                {"description": f"Analyze requirements for: {description}", "depends_on": []},
                {"description": f"Generate solution options for: {description}", "depends_on": [0]},
                {"description": f"Implement selected solution for: {description}", "depends_on": [1]}
            ]

    def set_ollama_model(self, model_name: str):
        """Configure which Ollama model to use for task decomposition"""
        self.ollama_model = model_name