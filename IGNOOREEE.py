import asyncio
import uuid
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum, auto
import ollama  # For LLM integration
import logging
from sentence_transformers import SentenceTransformer
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskGraph:
    def __init__(self, root_id: str, description: str):
        self.root_id = root_id
        self.description = description
        self.nodes = {root_id: description}
        self.edges = {}  # Maps task_id -> list of dependencies
    
    def add_node(self, task_id: str, description: str):
        self.nodes[task_id] = description
        if task_id not in self.edges:
            self.edges[task_id] = []
    
    def add_dependency(self, task_id: str, depends_on_id: str):
        if task_id not in self.edges:
            self.edges[task_id] = []
        if depends_on_id not in self.edges:
            self.edges[depends_on_id] = []
        self.edges[task_id].append(depends_on_id)


class AgentCapability:
    """Represents an agent's capability with associated weights."""

    def __init__(self, agent_id: str, skill_weights: Dict[str, float], specialization_weights: Dict[str,float]):
        self.agent_id = agent_id
        self.skill_weights = skill_weights  # skill_name -> weight
        self.specialization_weights = specialization_weights  # specialization_name -> weight
        self.vector = self._create_capability_vector()
        
    def _create_capability_vector(self) -> List[float]:
        """Create vector representation of agent's capabilities."""
        # Concatenate skill weights and specialization weights into a single vector
        return list(self.skill_weights.values()) + list(self.specialization_weights.values())

class TaskProfile:
    """Represents the requirements of a task."""
    def __init__(self, task_id: str, description: str, required_skills: Dict[str, float], required_specializations: Dict[str,float]):
        self.task_id = task_id
        self.description = description
        self.required_skills = required_skills  # skill_name -> weight
        self.required_specializations = required_specializations # specialization_name -> weight
        self.vector = self._create_task_vector()
        
    def _create_task_vector(self) -> List[float]:
        """Create vector representation of task requirements."""
        # Concatenate required skill weights and required specialization weights into a single vector
        return list(self.required_skills.values()) + list(self.required_specializations.values())

class TaskStatus(Enum):
    """Represents the status of a task in the task graph."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()

class TaskGraph:
    """Represents a task and its subtasks with dependencies."""
    def __init__(self, root_task_id: str, description: str):
        self.root_task_id = root_task_id
        self.description = description
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []  # (parent, child)
        self.add_node(root_task_id, description)

    def add_node(self, task_id: str, description: str, dependencies: Optional[List[str]] = None):
        """Add a task to the graph, optionally with dependencies."""
        self.nodes[task_id] = {
            "description": description,
            "status": TaskStatus.PENDING, 
            "result": None, 
            "assigned_to": None,
            "created_at": time.time(),
            "completed_at": None
        }
        
        if dependencies:
            for dep in dependencies:
                self.add_dependency(task_id, dep)

    def add_dependency(self, task_id: str, depends_on: str):
        """Add a dependency edge between two tasks."""
        if depends_on not in self.nodes:
            raise ValueError(f"Dependency task {depends_on} doesn't exist in the graph")
            
        self.edges.append((depends_on, task_id))

    def get_dependencies(self, task_id: str) -> List[str]:
        """Get the dependencies of a given task."""
        return [dep for dep, t in self.edges if t == task_id]
    
    def get_dependents(self, task_id: str) -> List[str]:
        """Get the dependents of a given task."""
        return [t for dep, t in self.edges if dep == task_id]
    
    def get_all_nodes(self) -> List[str]:
        """Get all task IDs in the graph."""
        return list(self.nodes.keys())
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        if task_id not in self.nodes:
            return False
        
        self.nodes[task_id]["assigned_to"] = agent_id
        self.nodes[task_id]["status"] = TaskStatus.IN_PROGRESS
        return True
    
    def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark a task as completed with its result."""
        if task_id not in self.nodes:
            return False
        
        self.nodes[task_id]["status"] = TaskStatus.COMPLETED
        self.nodes[task_id]["result"] = result
        self.nodes[task_id]["completed_at"] = time.time()
        return True
    
    def fail_task(self, task_id: str, reason: str) -> bool:
        """Mark a task as failed."""
        if task_id not in self.nodes:
            return False
        
        self.nodes[task_id]["status"] = TaskStatus.FAILED
        self.nodes[task_id]["result"] = {"error": reason}
        self.nodes[task_id]["completed_at"] = time.time()
        return True
    
    def get_available_tasks(self) -> List[str]:
        """Get tasks that are ready to be worked on (all dependencies are completed)."""
        available_tasks = []
        
        for task_id, node in self.nodes.items():
            if node["status"] == TaskStatus.PENDING:
                dependencies = self.get_dependencies(task_id)
                if not dependencies or all(self.nodes[dep]["status"] == TaskStatus.COMPLETED for dep in dependencies):
                    available_tasks.append(task_id)
                    
        return available_tasks
    
    def is_completed(self) -> bool:
        """Check if the entire task graph is completed."""
        return all(node["status"] == TaskStatus.COMPLETED for node in self.nodes.values())
    
    def get_completion_status(self) -> Dict[str, int]:
        """Get the count of tasks in each status."""
        status_counts = {status: 0 for status in TaskStatus}
        for node in self.nodes.values():
            status_counts[node["status"]] += 1
        return {status.name: count for status, count in status_counts.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the task graph to a dictionary representation."""
        return {
            "root_task_id": self.root_task_id,
            "description": self.description,
            "nodes": {
                task_id: {
                    **node,
                    "status": node["status"].name,
                    "dependencies": self.get_dependencies(task_id),
                    "dependents": self.get_dependents(task_id)
                } for task_id, node in self.nodes.items()
            }
        }

@dataclass
class AgentResult:
    agent_id: str
    task_id: str
    result: Any
    execution_time: float


@dataclass
class Message:
    """Container for messages exchanged between agents"""
    content: str
    sender_id: str
    recipient_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    
    def __post_init__(self):
        if not self.token_count:
            self.token_count = self._count_tokens()
    
    def _count_tokens(self) -> int:
        # Simple token counting logic based on whitespace
        return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp,
            "token_count": self.token_count
        }

class VectorStore:
    """Vector database for semantic storage and retrieval"""
    def __init__(self, dimension: int = 768, similarity_threshold: float = 0.7, model_name: str = "all-MiniLM-L6-v2"):
        self.vectors = {}  # id -> vector mapping
        self.data = {}     # id -> data mapping
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        
        # Verify the model's output dimension matches the expected dimension
        if self.model.get_sentence_embedding_dimension() != dimension:
            print(f"Warning: Model output dimension ({self.model.get_sentence_embedding_dimension()}) " 
                  f"differs from specified dimension ({dimension}). Updating dimension.")
            self.dimension = self.model.get_sentence_embedding_dimension()
    
    async def add(self, id: str, data: Any, vector: Optional[List[float]] = None) -> str:
        """Add data with vector representation"""
        if vector is None:
            vector = await self._generate_embedding(str(data))
        
        self.vectors[id] = vector
        self.data[id] = data
        return id
    
    async def search(self, query: Union[str, List[float]], top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search for similar items by vector similarity"""
        if isinstance(query, str):
            query_vector = await self._generate_embedding(query)
        else:
            query_vector = query
            
        results = []
        for id, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity >= self.similarity_threshold:
                results.append((id, self.data[id], similarity))
        
        return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text using a sentence transformer model"""
        # Convert to list to ensure it's serializable
        return self.model.encode(text).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1, v2 = np.array(vec1), np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class KnowledgeBase:
    """Enhanced knowledge storage with vector capabilities for agents"""
    def __init__(self, vector_dimension: int = 768):
        self.facts = VectorStore(dimension=vector_dimension)
        self.context = {}
        self.history = []
        self.summaries = []  # Periodic summaries of conversations
        self.shared_beliefs = VectorStore(dimension=vector_dimension)  # Team consensus knowledge
    
    async def add_fact(self, key: str, value: Any, confidence: float = 1.0) -> None:
        data = {"value": value, "confidence": confidence, "timestamp": time.time()}
        await self.facts.add(key, data)
    
    async def query_facts(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        return await self.facts.search(query, top_k)
    
    def update_context(self, context_update: Dict[str, Any]) -> None:
        self.context.update(context_update)
    
    def get_context(self) -> Dict[str, Any]:
        return self.context
    
    def add_to_history(self, message: Message) -> None:
        self.history.append(message)
        
        # Create periodic summaries to prevent history from getting too large
        if len(self.history) % 20 == 0:
            summary = self._summarize_recent_history(10)
            self.summaries.append(summary)
    
    async def get_relevant_history(self, query: str, max_messages: int = 10) -> List[Message]:
        """Get messages relevant to a query using semantic search"""
        # Convert recent history to searchable format
        recent_history = self.history[-100:] if len(self.history) > 100 else self.history
        
        # For each message, create a searchable representation and find relevance
        query_vector = await self.facts._generate_embedding(query)
        
        scored_messages = []
        for message in recent_history:
            # Create a searchable representation of the message
            message_text = f"{message.sender_id}: {message.content}"
            message_vector = await self.facts._generate_embedding(message_text)
            
            # Calculate similarity to query
            similarity = self.facts._cosine_similarity(query_vector, message_vector)
            scored_messages.append((message, similarity))
        
        # Sort by relevance and return top messages
        relevant_messages = [msg for msg, score in sorted(
            scored_messages, key=lambda x: x[1], reverse=True
        )[:max_messages]]
        
        return relevant_messages
    
    def _summarize_recent_history(self, num_messages: int) -> Dict[str, Any]:
        """Create a summary of recent messages"""
        recent = self.history[-num_messages:] if self.history else []
        
        # Simple summarization logic - count messages by participants
        participant_counts = {}
        for msg in recent:
            participant_counts[msg.sender_id] = participant_counts.get(msg.sender_id, 0) + 1
        
        return {
            "timestamp": time.time(),
            "message_count": len(recent),
            "participants": list(participant_counts.keys()),
            "message_distribution": participant_counts,
            "summary": f"Discussion between {', '.join(participant_counts.keys())} with {len(recent)} messages"
        }

class Skill:
    """Reusable capability that can be attached to agents"""
    def __init__(self, name: str, function: Callable, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    async def execute(self, *args, **kwargs):
        """Execute the skill's function"""
        return await self.function(*args, **kwargs)

class TaskBid:
    """Represents an agent's bid to work on a task"""
    def __init__(self, agent_id: str, task_id: str, confidence: float, estimated_time: float):
        self.agent_id = agent_id
        self.task_id = task_id
        self.confidence = confidence  # 0.0 to 1.0
        self.estimated_time = estimated_time  # Seconds
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bid to dictionary representation"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "confidence": self.confidence,
            "estimated_time": self.estimated_time,
            "timestamp": self.timestamp
        }

class DecentralizedTaskMarket:
    """A decentralized marketplace for task bidding and assignment"""
    def __init__(self):
        self.available_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> task details
        self.bids: Dict[str, List[TaskBid]] = {}  # task_id -> list of bids
        self.task_graphs: Dict[str, TaskGraph] = {}  # task_graph_id -> TaskGraph
        self.subscribed_agents: Dict[str, Set[str]] = {}  # agent_id -> set of task types interested in
        
    async def publish_task(self, task_id: str, description: str, task_type: str = "general",
                         complexity: float = 0.5, priority: int = 1, 
                         task_graph_id: Optional[str] = None) -> str:
        """Publish a task to the marketplace"""
        task_info = {
            "id": task_id,
            "description": description,
            "type": task_type,
            "complexity": complexity,
            "priority": priority,
            "published_at": time.time(),
            "status": "open",
            "task_graph_id": task_graph_id
        }
        
        self.available_tasks[task_id] = task_info
        self.bids[task_id] = []
        
        # Notify subscribed agents
        await self._notify_subscribed_agents(task_info)
        
        return task_id
    
    async def place_bid(self, bid: TaskBid) -> bool:
        """Place a bid on a task"""
        if bid.task_id not in self.available_tasks:
            return False
        
        if self.available_tasks[bid.task_id]["status"] != "open":
            return False
        
        self.bids[bid.task_id].append(bid)
        return True
    
    async def subscribe_to_tasks(self, agent_id: str, task_types: List[str]) -> None:
        """Subscribe an agent to be notified of certain task types"""
        if agent_id not in self.subscribed_agents:
            self.subscribed_agents[agent_id] = set()
            
        self.subscribed_agents[agent_id].update(task_types)
    
    async def get_task_notifications(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all open tasks that match an agent's subscriptions"""
        if agent_id not in self.subscribed_agents:
            return []
        
        interested_types = self.subscribed_agents[agent_id]
        matching_tasks = []
        
        for task_id, task_info in self.available_tasks.items():
            if task_info["status"] == "open" and (
                "general" in interested_types or task_info["type"] in interested_types
            ):
                matching_tasks.append(task_info)
        
        return matching_tasks
    
    async def get_bids_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all bids for a specific task"""
        if task_id not in self.bids:
            return []
        
        return [bid.to_dict() for bid in self.bids[task_id]]
    
    async def award_task(self, task_id: str, agent_id: str) -> bool:
        """Award a task to a specific agent"""
        if task_id not in self.available_tasks:
            return False
        
        if self.available_tasks[task_id]["status"] != "open":
            return False
        
        # Update task status
        self.available_tasks[task_id]["status"] = "assigned"
        self.available_tasks[task_id]["assigned_to"] = agent_id
        self.available_tasks[task_id]["assigned_at"] = time.time()
        
        # If part of a task graph, update the task graph
        task_graph_id = self.available_tasks[task_id].get("task_graph_id")
        if task_graph_id and task_graph_id in self.task_graphs:
            self.task_graphs[task_graph_id].assign_task(task_id, agent_id)
        
        return True
    
    async def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark a task as completed with its result"""
        if task_id not in self.available_tasks:
            return False
        
        if self.available_tasks[task_id]["status"] != "assigned":
            return False
        
        # Update task status
        self.available_tasks[task_id]["status"] = "completed"
        self.available_tasks[task_id]["completed_at"] = time.time()
        self.available_tasks[task_id]["result"] = result
        
        # If part of a task graph, update the task graph
        task_graph_id = self.available_tasks[task_id].get("task_graph_id")
        if task_graph_id and task_graph_id in self.task_graphs:
            self.task_graphs[task_graph_id].complete_task(task_id, result)
            
            # Check if new tasks are available in the graph
            await self._publish_available_tasks_from_graph(task_graph_id)
        
        return True
    
    async def publish_task_graph(self, task_graph: TaskGraph) -> str:
        """Publish an entire task graph to the marketplace"""
        graph_id = str(uuid.uuid4())
        self.task_graphs[graph_id] = task_graph
        
        # Publish all available tasks from the graph
        await self._publish_available_tasks_from_graph(graph_id)
        
        return graph_id
    
    async def get_task_graph_status(self, graph_id: str) -> Dict[str, Any]:
        """Get the status of a task graph"""
        if graph_id not in self.task_graphs:
            return {"error": "Task graph not found"}
        
        graph = self.task_graphs[graph_id]
        return {
            "id": graph_id,
            "description": graph.description,
            "completion_status": graph.get_completion_status(),
            "is_completed": graph.is_completed()
        }
    
    async def _publish_available_tasks_from_graph(self, graph_id: str) -> None:
        """Publish all available tasks from a task graph"""
        if graph_id not in self.task_graphs:
            return
        
        graph = self.task_graphs[graph_id]
        available_tasks = graph.get_available_tasks()
        
        for task_id in available_tasks:
            # Skip if already published
            if task_id in self.available_tasks:
                continue
                
            node = graph.nodes[task_id]
            await self.publish_task(
                task_id=task_id,
                description=node["description"],
                task_type="subtask",  # Mark as subtask
                complexity=0.5,  # Default complexity
                priority=1,  # Default priority
                task_graph_id=graph_id
            )
    
    async def _notify_subscribed_agents(self, task_info: Dict[str, Any]) -> None:
        """Notify subscribed agents about a new task"""
        # In a real system, this would send notifications to agents
        # For now, just log the notification
        task_type = task_info["type"]
        agents_to_notify = []
        
        for agent_id, subscriptions in self.subscribed_agents.items():
            if "general" in subscriptions or task_type in subscriptions:
                agents_to_notify.append(agent_id)
        
        if agents_to_notify:
            logger.info(f"Notifying agents {agents_to_notify} about new task {task_info['id']}")

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

# Demo implementation of the collaborative agent system
async def run_demo():
    logger.info("Starting collaborative agent demo")
    
    # Create a team
    team = AgentTeam(str(uuid.uuid4()), "Research Team")
    
    # Create specialized agents with different capabilities
    researcher = SpecializedAgent(
        agent_id=str(uuid.uuid4()),
        name="ResearchBot",
        specialization="research",
        skills={"data_analysis": 0.9, "information_retrieval": 0.8, "critical_thinking": 0.7},
        specialty_areas={"academic_research": 0.9, "literature_review": 0.8}
    )
    
    writer = SpecializedAgent(
        agent_id=str(uuid.uuid4()),
        name="WriterBot",
        specialization="writing",
        skills={"writing": 0.9, "editing": 0.8, "storytelling": 0.7},
        specialty_areas={"technical_writing": 0.9, "creative_writing": 0.6}
    )
    
    analyst = SpecializedAgent(
        agent_id=str(uuid.uuid4()),
        name="AnalystBot",
        specialization="analysis",
        skills={"data_analysis": 0.9, "critical_thinking": 0.8, "problem_solving": 0.9},
        specialty_areas={"statistical_analysis": 0.9, "pattern_recognition": 0.8}
    )
    
    # Add agents to the team with their subscriptions
    await team.add_agent(researcher, ["research", "information", "data_collection"])
    await team.add_agent(writer, ["writing", "content", "documentation"])
    await team.add_agent(analyst, ["analysis", "data", "evaluation"])
    
    # Start the team
    await team.start()
    
    # Send a message to the team
    await team.send_message_to_team("Hello team! I need help with a research project on climate change.")
    
    # Submit a simple task
    simple_task_id = await team.submit_task(
        "Find 3 recent studies on rising sea levels",
        complexity=0.6,
        priority=2
    )
    
    # Submit a complex task
    complex_task_id = await team.submit_complex_task(
        "Create a comprehensive report on climate change impacts in coastal cities"
    )
    
    # Wait a moment for the system to process
    await asyncio.sleep(2)
    
    # Check task statuses
    simple_task_status = await team.get_task_status(simple_task_id)
    logger.info(f"Simple task status: {simple_task_status}")
    
    complex_task_status = await team.get_task_status(complex_task_id)
    logger.info(f"Complex task status: {complex_task_status}")
    
    # Simulate agents bidding on tasks
    for agent_id, agent in team.agents.items():
        available_tasks = await agent.check_task_market()
        for task in available_tasks:
            await agent.bid_on_task(task["id"])
    
    # Wait for bids to be processed
    await asyncio.sleep(1)
    
    # Simulate task awards - award to highest bidder
    for task_id in [simple_task_id]:
        bids = await team.task_market.get_bids_for_task(task_id)
        if bids:
            # Sort by confidence
            sorted_bids = sorted(bids, key=lambda x: x["confidence"], reverse=True)
            best_bid = sorted_bids[0]
            await team.task_market.award_task(task_id, best_bid["agent_id"])
            logger.info(f"Task {task_id} awarded to {best_bid['agent_id']}")
    
    # Simulate task completion
    for agent_id, agent in team.agents.items():
        available_tasks = await agent.check_task_market()
        for task in available_tasks:
            if task.get("assigned_to") == agent_id and task.get("status") == "assigned":
                # Agent completes the task
                result = await agent.reason(task["description"])
                await agent.complete_task(task["id"], result)
    
    # Wait for the system to process
    await asyncio.sleep(2)
    
    # Get team performance
    performance = await team.get_team_performance()
    logger.info(f"Team performance: {json.dumps(performance, indent=2)}")
    
    # Check final task statuses
    simple_task_status = await team.get_task_status(simple_task_id)
    logger.info(f"Final simple task status: {simple_task_status}")
    
    complex_task_status = await team.get_task_status(complex_task_id)
    logger.info(f"Final complex task status: {complex_task_status}")
    
    logger.info("Demo completed")

if __name__ == "__main__":
    asyncio.run(run_demo()) 