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
