import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
import logging
from legosquare.lib import TaskBid, TaskGraph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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