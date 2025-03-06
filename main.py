import asyncio
import uuid
import json
import logging
from legosquare import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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