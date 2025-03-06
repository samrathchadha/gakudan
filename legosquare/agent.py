import asyncio
import uuid
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from abc import ABC, abstractmethod
import ollama  # For LLM integration
import logging
from legosquare.lib import TaskGraph, Message, Skill, TaskBid, AgentCapability, TaskProfile, KnowledgeBase
from legosquare.market import DecentralizedTaskMarket

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent(ABC):
    """Enhanced abstract base class for collaborative agents"""
    def __init__(self, agent_id: str, name: str = None, specialization: str = None):
        self.id = agent_id
        self.name = name or f"Agent-{agent_id[:8]}"
        self.specialization = specialization
        self.knowledge_base = KnowledgeBase()
        self.skills = {}  # Name -> Skill mapping
        self.collaborators = set()  # Other agents this agent works with
        self.trust_levels = {}  # Agent ID -> trust level (0.0 to 1.0)
        self.learning_rate = 0.1  # How quickly the agent adapts its strategies
        self.task_market = None  # Reference to the task market
        
        # Self-tracked performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "total_tokens_used": 0,
            "reasoning_tasks_completed": 0,
            "tasks_accepted": 0,
            "tasks_completed": 0,
            "total_reasoning_time": 0,
            "average_response_time": 0,
            "successful_collaborations": 0,
        }
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process an incoming message and optionally generate a response"""
        pass
    
    @abstractmethod
    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Perform reasoning on a task and return result"""
        pass
    
    async def connect_to_task_market(self, task_market: DecentralizedTaskMarket, subscribe_to: List[str] = None) -> None:
        """Connect the agent to a task market and subscribe to relevant task types"""
        self.task_market = task_market
        
        if subscribe_to:
            await task_market.subscribe_to_tasks(self.id, subscribe_to)
    
    async def send_message(self, content: str, recipient_id: str) -> Message:
        """Create and send a message to another agent"""
        start_time = time.time()
        
        message = Message(
            content=content,
            sender_id=self.id,
            recipient_id=recipient_id,
        )
        
        # Update metrics
        self.metrics["messages_sent"] += 1
        self.metrics["total_tokens_used"] += message.token_count
        
        # Update average response time
        if self.metrics["messages_sent"] > 0:
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["messages_sent"] - 1) + 
                (time.time() - start_time)) / self.metrics["messages_sent"]
            )
        
        # Add to knowledge base
        self.knowledge_base.add_to_history(message)
        
        return message
    
    async def receive_message(self, message: Message) -> Optional[Message]:
        """Process an incoming message and possibly return a response"""
        if message.recipient_id not in [self.id, None]:
            raise ValueError(f"Message {message.id} is not intended for this agent")
        
        # Update metrics
        self.metrics["messages_received"] += 1
        
        # Add to knowledge base
        self.knowledge_base.add_to_history(message)
        
        # Update trust model based on message quality
        if message.sender_id in self.trust_levels:
            # Simple trust update based on message usefulness
            current_trust = self.trust_levels.get(message.sender_id, 0.5)
            self.trust_levels[message.sender_id] = (
                current_trust * (1 - self.learning_rate) 
            )
        
        # Process message and generate response
        response = await self.process_message(message)
        return response
    
    async def add_skill(self, skill: Skill) -> None:
        """Add a new skill to the agent's capabilities"""
        self.skills[skill.name] = skill
    
    async def use_skill(self, skill_name: str, *args, **kwargs) -> Any:
        """Use a specific skill"""
        if skill_name not in self.skills:
            raise ValueError(f"Skill {skill_name} not found")
        
        return await self.skills[skill_name].execute(*args, **kwargs)
    
    async def check_task_market(self) -> List[Dict[str, Any]]:
        """Check the task market for available tasks"""
        if not self.task_market:
            logger.warning(f"Agent {self.id} not connected to a task market")
            return []
        
        return await self.task_market.get_task_notifications(self.id)
    
    async def evaluate_task(self, task_info: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate a task and return the confidence level and estimated time"""
        # Check task type against specialization
        if self.specialization and task_info.get("type") == self.specialization:
            confidence_boost = 0.2
        else:
            confidence_boost = 0
        
        # Calculate confidence - base level plus boosts for specialization and past tasks
        confidence = 0.5 + confidence_boost
        confidence = min(0.95, max(0.1, confidence))  # Clamp to reasonable range
        
        # Estimate time based on complexity
        complexity = task_info.get("complexity", 0.5)
        estimated_time = 10 + (complexity * 20)  # Simple formula: 10 to 30 seconds
        
        return confidence, estimated_time
    
    async def bid_on_task(self, task_id: str) -> bool:
        """Bid on a task in the marketplace"""
        if not self.task_market:
            logger.warning(f"Agent {self.id} not connected to a task market")
            return False
        
        # Get task information
        task_notifications = await self.check_task_market()
        task_info = next((task for task in task_notifications if task["id"] == task_id), None)
        
        if not task_info:
            logger.warning(f"Task {task_id} not found in available tasks for agent {self.id}")
            return False
        
        # Evaluate the task
        confidence, estimated_time = await self.evaluate_task(task_info)
        
        # Create and place bid
        bid = TaskBid(
            agent_id=self.id,
            task_id=task_id,
            confidence=confidence,
            estimated_time=estimated_time
        )
        
        success = await self.task_market.place_bid(bid)
        if success:
            logger.info(f"Agent {self.id} placed bid on task {task_id} with confidence {confidence:.2f}")
        
        return success
    
    async def accept_task(self, task_id: str) -> bool:
        """Accept a task assigned by the task market"""
        if not self.task_market:
            logger.warning(f"Agent {self.id} not connected to a task market")
            return False
        
        # Check if task is assigned to this agent
        task_notifications = await self.check_task_market()
        task_info = next((task for task in task_notifications if task["id"] == task_id 
                         and task.get("assigned_to") == self.id), None)
        
        if not task_info:
            logger.warning(f"Task {task_id} not assigned to agent {self.id}")
            return False
        
        # Update metrics
        self.metrics["tasks_accepted"] += 1
        
        return True
    
    async def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark a task as completed in the task market"""
        if not self.task_market:
            logger.warning(f"Agent {self.id} not connected to a task market")
            return False
        
        success = await self.task_market.complete_task(task_id, result)
        
        if success:
            # Update metrics
            self.metrics["tasks_completed"] += 1
            logger.info(f"Agent {self.id} completed task {task_id}")
        
        return success
    
    async def create_task_graph(self, task: str) -> TaskGraph:
        """Create a task dependency graph from a high-level task"""
        # Generate a unique ID for the root task
        root_task_id = str(uuid.uuid4())
        
        # Create a new task graph with the root task
        task_graph = TaskGraph(root_task_id, task)

class LLMAgent(Agent):
    """An agent that uses a Large Language Model for reasoning and processing messages."""
    
    def __init__(
        self, 
        agent_id: str, 
        name: str = None, 
        specialization: str = None,
        model: str = "llama3.2:latest", 
        context_window_size: int = 10
    ):
        super().__init__(agent_id, name, specialization)
        self.model = model
        self.context_window = []
        self.context_window_size = context_window_size
        self.llm_client = self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        # This method allows for easier mocking in tests and future client changes
        return ollama
        
    def _update_context(self, sender_id: str, content: str) -> None:
        self.context_window.append(f"[{sender_id}]: {content}")
        
        # Maintain context window size limit
        if len(self.context_window) > self.context_window_size:
            self.context_window = self.context_window[-self.context_window_size:]
    
    def _prepare_context_dict(self, message: Message) -> Dict[str, Any]:
        return {
            "history": "\n".join(self.context_window),
            "sender": message.sender_id,
        }
        
    async def process_message(self, message: Message) -> Optional[Message]:
        # Add incoming message to context
        self._update_context(message.sender_id, message.content)
        
        # Prepare context for reasoning
        context = self._prepare_context_dict(message)
        
        # Use LLM for reasoning and track performance
        start_time = time.time()
        
        try:
            response_content = await self.reason(message.content, context)
            
            # Update metrics
            reasoning_time = time.time() - start_time
            self.metrics["reasoning_tasks_completed"] += 1
            self.metrics["total_reasoning_time"] += reasoning_time
            
            # If we have a valid response, add to context and return
            if response_content:
                self._update_context(self.id, response_content)
                return await self.send_message(response_content, message.sender_id)
            
        except Exception as e:
            logger.error(f"Error during message processing: {str(e)}")
            
        return None
    
    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Construct prompt with context if available
            prompt = self._build_prompt(task, context)
            
            # Execute LLM inference
            return await self._execute_llm_call(prompt)
            
        except Exception as e:
            logger.error(f"LLM reasoning error: {str(e)}")
            raise
    
    async def _execute_llm_call(self, prompt: str) -> str:
        """Separated LLM call for better testability and potential async implementation"""
        response = self.llm_client.generate(
            model=self.model,
            prompt=prompt,
        )
        return response['response'].strip()
    
    def _build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        context_str = ""
        if context:
            context_str = f"Context:\n{json.dumps(context, indent=2)}\n\n"
        
        return f"{context_str}Task: {task}\n\nResponse:"
    
    

class SpecializedAgent(LLMAgent):
    """An agent with specialized capabilities for specific domains"""
    def __init__(self, agent_id: str, name: str, specialization: str, 
                 skills: Dict[str, float], specialty_areas: Dict[str, float], 
                 model: str = "llama3.2:latest"):
        super().__init__(agent_id, name, specialization, model)
        self.skill_weights = skills
        self.specialty_weights = specialty_areas
        self.capability = AgentCapability(agent_id, skills, specialty_areas)
        
    async def evaluate_task(self, task_info: Dict[str, Any]) -> Tuple[float, float]:
        """More sophisticated task evaluation based on agent capabilities"""
        # Create a task profile for the task
        description = task_info.get("description", "")
        
        # Extract required skills and specializations from task description
        # In a real system, this would use more sophisticated methods like NLP
        required_skills = {}
        required_specializations = {}
        
        for skill, weight in self.skill_weights.items():
            if skill.lower() in description.lower():
                required_skills[skill] = 0.8  # Default weight if mentioned
                
        for specialty, weight in self.specialty_weights.items():
            if specialty.lower() in description.lower():
                required_specializations[specialty] = 0.8  # Default weight if mentioned
        
        # Ensure there's at least something to match against
        if not required_skills and not required_specializations:
            # Fallback to generic requirements
            required_skills = {next(iter(self.skill_weights.keys()), "general"): 0.5}
            
        # Create task profile
        task_profile = TaskProfile(
            task_id=task_info.get("id", "unknown"),
            description=description,
            required_skills=required_skills,
            required_specializations=required_specializations
        )
        
        # Calculate match score
        match_score = self._calculate_capability_match(task_profile)
        
        # Estimate completion time based on complexity and match score
        complexity = task_info.get("complexity", 0.5)
        base_time = 10 + (complexity * 30)  # Base time from 10 to 40 seconds
        
        # Adjust time based on match score (better match = faster completion)
        estimated_time = base_time * (1.5 - match_score)
        estimated_time = max(5, min(60, estimated_time))  # Clamp between 5 and 60 seconds
        
        return match_score, estimated_time
    
    def _calculate_capability_match(self, task_profile: TaskProfile) -> float:
        """Calculate how well this agent's capabilities match a task profile"""
        agent_vector = np.array(self.capability.vector)
        task_vector = np.array(task_profile.vector)
        
        # If vectors have different lengths, pad the shorter one
        if len(agent_vector) > len(task_vector):
            padding = np.zeros(len(agent_vector) - len(task_vector))
            task_vector = np.append(task_vector, padding)
        elif len(task_vector) > len(agent_vector):
            padding = np.zeros(len(task_vector) - len(agent_vector))
            agent_vector = np.append(agent_vector, padding)
            
        # Calculate cosine similarity
        if np.sum(task_vector) == 0:
            return 0.5  # Default score for empty task requirements
            
        dot_product = np.dot(agent_vector, task_vector)
        norm_agent = np.linalg.norm(agent_vector)
        norm_task = np.linalg.norm(task_vector)
        
        if norm_agent == 0 or norm_task == 0:
            return 0.5  # Default score for zero vectors
            
        similarity = dot_product / (norm_agent * norm_task)
        
        # Scale similarity from [-1, 1] to [0, 1]
        adjusted_score = (similarity + 1) / 2
        
        # Boost score if specialization directly matches
        if self.specialization and self.specialization.lower() in task_profile.description.lower():
            adjusted_score = min(0.95, adjusted_score + 0.2)
            
        return adjusted_score

