from rich.console import Console
from rich.markdown import Markdown
import threading
import queue
import logging
import colorlog
import uuid
from typing import List, Dict, Any
import networkx as nx
import matplotlib.pyplot as plt
import json
from google import genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configure Colorful Logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s][%(threadName)s] %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# Constants for prompts
THOUGHT_GENERATOR_SYSTEM_PROMPT = """Generate an appropriate number of sub-prompts to solve the given problem and explore the problem from specific perspectives and personality traits that are unique
- the point of the prompts is when their answers are summarized, the main propmt is very well explained, design them with that in mind.
- make sure each prompt is very unique, dont make the prompts tasks.
- the goal is to aswer the question cohesively and focus on all essential perspectives.
- the prompts you give after defining personalities and perspective should be as close to main prompt as possible. phrase the prompt as a question
- boldly give personalities
- the personality should bring unique perspective but shouldn't be TOO creative so you know roles that contribute is the priority
- personalities could be designed taking into perspective what other personalities may need to complement them
- do not ask questions that can be answered by yes or no
- ensure there is a new line between points
- do not include any introductory messages
- DO NOT GIVE THEM NAMES
- format start:
    1)personality:prompt
    2)personality:prompt
    3)personality:prompt
    4)personality:prompt
    ..
- format end
do not deviate from format at all
"""

SUB_THOUGHT_GENERATOR_SYSTEM_PROMPT = """
IF you feel like this needs further deliberation, you can simply respond with what personalities should deliberate it and what prompt they should use. they will already be given your current answer, you can ask them further questoins on it
PLEASE NOTE: there is no need to generate them, if you think the depth of exploration here is sufficient you are not required to further elaborate
Generate a MAXIMUM of 3 sub-prompts to solve the given problem and explore the problem from specific perspectives and personality traits that are unique
- the point of the prompts is when their answers are summarized, the main propmt is very well explained, design them with that in mind.
- make sure each prompt is very unique, dont make the prompts tasks.
- the goal is to aswer the question cohesively and focus on all essential perspectives.
- the prompts you give after defining personalities and perspective should be as close to main prompt as possible. phrase the prompt as a question
- boldly give personalities
- DO NOT GIVE THEM NAMES
- the personality should bring unique perspective but shouldn't be TOO creative so you know roles that contribute is the priority
- personalities could be designed taking into perspective what other personalities may need to complement them
- do not ask questions that can be answered by yes or no
- ensure there is a new line between points if given
- do not include any introductory messages
- format start:
    1)personality:prompt
    2)personality:prompt
    ..
- format end
do not deviate from format at all
"""

SUB_PROMPT_SYSTEM_INSTRUCTION = "You are an AI assistant. Provide EXTREMELY CONCISE and PRECISE responses. Do not answer with yes or no, build unique content. Provide strong tangible solutions to the task at hand. do not include any introductory messages or conclusive messages"

COMBINER_SYSTEM_PROMPT = 'You have been given several perspectives from many different people.INCLUDE EVERY GOOD POINT ANYONE HAS MENTIONED. You will now create a formal plan with all the work everyone has done. Dont summarize, synthesize a coherent synchronous answer that is a final answer to the original prompt, not just a summary of the answers you have. DO NOT GIVE ANY TOPICS. JUST GIVE IT INSTRUCTIONS ON HOW TO DO NOT WHAT TO DO IT ON. do not include any introductory messages or conclusive messages. You do not need to be concise.'

# RAG Knowledge Base
class KnowledgeBase:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.lock = threading.Lock()

    def add_document(self, text, node_id):
        with self.lock:
            self.documents.append({"text": text, "node_id": node_id})
            # Recompute vectors
            if len(self.documents) > 1:
                texts = [doc["text"] for doc in self.documents]
                self.vectors = self.vectorizer.fit_transform(texts)
            elif len(self.documents) == 1:
                texts = [self.documents[0]["text"]]
                self.vectors = self.vectorizer.fit_transform(texts)

    def query(self, query_text, top_k=1):
        with self.lock:
            if not self.documents:
                return []

            query_vector = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()

            # Get top-k most similar documents
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                results.append({
                    "text": self.documents[idx]["text"],
                    "node_id": self.documents[idx]["node_id"],
                    "similarity": similarities[idx]
                })

            return results

# Graph for tracking prompt relationships
class PromptGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.lock = threading.Lock()
        self.knowledge_base = KnowledgeBase()

    def add_node(self, node_id, **attributes):
        with self.lock:
            if node_id in self.graph:
                # Update existing node attributes
                for key, value in attributes.items():
                    self.graph.nodes[node_id][key] = value
            else:
                # Add new node
                self.graph.add_node(node_id, **attributes)

            # If response is provided, add to knowledge base
            if 'response' in attributes:
                prompt_text = self.graph.nodes[node_id].get('prompt', '')
                response_text = attributes['response']
                combined_text = f"{prompt_text}\n{response_text}"
                self.knowledge_base.add_document(combined_text, node_id)

    def add_edge(self, parent_id, child_id):
        with self.lock:
            self.graph.add_edge(parent_id, child_id)
            # Fix depth issue: ensure child depth is greater than parent
            if parent_id in self.graph and child_id in self.graph:
                parent_depth = self.graph.nodes[parent_id].get('depth', 0)
                child_depth = self.graph.nodes[child_id].get('depth', 0)

                if child_depth <= parent_depth:
                    self.graph.nodes[child_id]['depth'] = parent_depth + 1

    def get_children(self, node_id):
        with self.lock:
            return list(self.graph.successors(node_id))

    def get_parent(self, node_id):
        with self.lock:
            parents = list(self.graph.predecessors(node_id))
            return parents[0] if parents else None

    def query_knowledge_base(self, query_text):
        return self.knowledge_base.query(query_text)

    def visualize(self, output_file="prompt_graph.png"):
        with self.lock:
            plt.figure(figsize=(12, 10))

            # Create node colors based on depth
            node_colors = []
            for node in self.graph.nodes():
                if node == 'root':
                    node_colors.append('red')
                else:
                    depth = self.graph.nodes[node].get('depth', 0)
                    if depth == 0:
                        node_colors.append('orange')
                    elif depth == 1:
                            node_colors.append('yellow')
                    else:
                        node_colors.append('green')

            # Create node labels
            node_labels = {}
            for node in self.graph.nodes():
                if node == 'root':
                    node_labels[node] = 'Main Prompt'
                else:
                    label = self.graph.nodes[node].get('prompt', '')
                    if label and len(label) > 20:
                        label = label[:17] + "..."
                    node_labels[node] = f"{node[:4]}...: {label}"

            pos = nx.spring_layout(self.graph, seed=42)
            nx.draw(self.graph, pos, with_labels=False, node_color=node_colors,
                    node_size=500, arrows=True, arrowsize=15)
            nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=8)

            d = nx.json_graph.node_link_data(self.graph)  # node-link format to serialize
            json.dump(d, open("out.json", "w"))

            plt.title("Prompt Hierarchy Graph")
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Graph visualization saved to {output_file}")

    def to_dict(self):
        with self.lock:
            nodes = {}
            for node in self.graph.nodes():
                nodes[node] = {
                    'attributes': dict(self.graph.nodes[node]),
                    'children': list(self.graph.successors(node)),
                    'parent': list(self.graph.predecessors(node))
                }
            return nodes

def generate_system_prompts(query: str, client):
    global THOUGHT_GENERATOR_SYSTEM_PROMPT, SUB_PROMPT_SYSTEM_INSTRUCTION, COMBINER_SYSTEM_PROMPT

    try:
        # Generate thought generator prompt
        thought_generator_response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a prompt engineer. Create a system prompt for generating diverse thought approaches. BE VERY CONCISE. SHORT ANSWERS ONLY. LIKE 3 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"
            ),
            contents=[f"Based on this question: '{query}', create a DIRECTION for team members to be made in. Do not tell what team members should be chosen just make reccomendations and hints. You are writing system instructions for the recruiter of the team"]
        )
        new_thought_generator = thought_generator_response.text

        # Generate sub-prompt system instruction
        sub_prompt_response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a prompt engineer. Create VERY concise system instruction. LIKE 4 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"
            ),
            contents=[f"Based on this question: '{query}', create a brief system instruction for responding to sub-prompts. Your prompt should be in the ddirection of HOW to answer, not WHAT to answer. Each system has very specific tasks so this is a guideline of HOW to answer, not what topics to mention. do NOT be generic"]
        )
        new_sub_prompt = sub_prompt_response.text

        # Generate combiner prompt
        combiner_response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a prompt engineer. Create a system prompt for synthesizing multiple perspectives. LIKE 4 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"
            ),
            contents=[f"Based on this question: '{query}', create a system prompt for combining multiple perspectives into one coherent answer. Your system prompt should be more like on how to plan the answer NOT on what to write"]
        )
        new_combiner = combiner_response.text

        if new_thought_generator.strip():
            THOUGHT_GENERATOR_SYSTEM_PROMPT = f"""
            Generate an appropriate number of sub-prompts to solve the given problem and explore the problem from specific perspectives and personality traits that are unique
            {new_thought_generator}
            - the point of the prompts is when their answers are summarized, the main propmt is very well explained, design them with that in mind.
            - make sure each prompt is very unique, dont make the prompts tasks.
            - the goal is to aswer the question cohesively and focus on all essential perspectives.
            - the prompts you give after defining personalities and perspective should be as close to main prompt as possible. phrase the prompt as a question
            - boldly give personalities
            - the personality should bring unique perspective but shouldn't be TOO creative so you know roles that contribute is the priority
            - personalities could be designed taking into perspective what other personalities may need to complement them
            - do not ask questions that can be answered by yes or no
            - do not include any introductory messages
            - format start:
                1)personality:prompt
                2)personality:prompt
                3)personality:prompt
                4)personality:prompt
                ..
            - format end
            do not deviate from format at all
            """

        if new_sub_prompt.strip():
            SUB_PROMPT_SYSTEM_INSTRUCTION += new_sub_prompt

        if new_combiner.strip():
            COMBINER_SYSTEM_PROMPT += new_combiner

    except Exception as e:
        logging.error(f"Error generating system prompts: {e}")

    return f"Modifications to prompts\nTHOUGHT_GENERATOR_SYSTEM_PROMPT: {new_thought_generator}\nSUB_PROMPT_SYSTEM_INSTRUCTION: {new_sub_prompt}\nCOMBINER_SYSTEM_PROMPT: {new_combiner}"

def mprint(string):
    print("\n\n")
    Console().print(Markdown(string))
    print("\n\n")

class GeminiPromptProcessor:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.thread_results = {}
        self.combined_result = None
        self.task_queue = queue.Queue()
        self.all_results = []
        self.result_queue = queue.Queue()
        self.max_depth = 2  # Limit recursion depth to prevent infinite loops
        self.colors = [
            '\033[31m',  # Red
            '\033[91m',  # Bright Red
            '\033[33m',  # Yellow
            '\033[93m',  # Bright Yellow
            '\033[34m',  # Blue
            '\033[94m',  # Bright Blue
            '\033[35m',  # Magenta
            '\033[95m',  # Bright Magenta
            '\033[36m',  # Cyan
            '\033[96m',  # Bright Cyan
            '\033[32m',  # Green
            '\033[92m',  # Bright Green
            '\033[37m',  # White
            '\033[97m',  # Bright White
        ]
        self.color_lock = threading.Lock()
        self.color_index = 0
        self.active_threads = 0
        self.active_threads_lock = threading.Lock()
        self.all_tasks_completed = threading.Event()
        self.rate_limit = 30 # prompts per minute
        self.last_minute_prompts = []

        # Initialize our graph tracking system
        self.prompt_graph = PromptGraph()
        # Add root node
        self.prompt_graph.add_node('root', prompt='Main Prompt', depth=-1)

    def get_next_color(self):
        with self.color_lock:
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            return color

    def add_to_queue(self, sub_prompt, parent_id, depth=0):
        if depth >= self.max_depth:
            logger.warning(f"Max depth limit reached for task: {sub_prompt}")
            return False

        color = self.get_next_color()
        new_id = str(uuid.uuid4())
        logger.info(f"{color}Adding new task to queue from parent {parent_id}: {sub_prompt}\033[0m")

        # Add to our graph
        if len(sub_prompt.split("\n")) > 1:
            self.prompt_graph.add_node(new_id, prompt=sub_prompt.split("\n")[-1], depth=depth+1, color=color)
        self.prompt_graph.add_node(new_id, prompt=sub_prompt, depth=depth, color=color)
        self.prompt_graph.add_edge(parent_id, new_id)

        self.task_queue.put({
            'id': new_id,
            'sub_prompt': sub_prompt,
            'parent_id': parent_id,
            'depth': depth + 1,
            'color': color
        })

        return True

    def generate_thought_approaches(self, main_prompt: str) -> List[Dict[str, str]]:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction=THOUGHT_GENERATOR_SYSTEM_PROMPT
                ),
                contents=[f"main prompt: how to fix {main_prompt}\ngenerate unique sub-prompts. one per line "]
            )
            approaches_content = response.text
            mprint(approaches_content)

            approach_list = []
            for line in approaches_content.split('\n'):
                if line.strip():
                    node_id = str(uuid.uuid4())
                    # Add to our graph
                    self.prompt_graph.add_node(node_id, prompt=line.strip(), depth=0)
                    self.prompt_graph.add_edge('root', node_id)

                    approach_list.append({
                        'id': node_id,
                        'prompt': line.strip()
                    })

            return approach_list
        except Exception as e:
            logger.error(f"Error generating thought approaches: {e}")
            # Return some fallback approaches with IDs for testing
            fallbacks = ["a", "b", "c", "d"]
            approach_list = []
            for fallback in fallbacks:
                node_id = str(uuid.uuid4())
                self.prompt_graph.add_node(node_id, prompt=fallback, depth=0)
                self.prompt_graph.add_edge('root', node_id)
                approach_list.append({
                    'id': node_id,
                    'prompt': fallback
                })
            return approach_list

    def check_rate_limit(self):
        current_time = time.time()
        self.last_minute_prompts = [t for t in self.last_minute_prompts if current_time - t < 60]
        if len(self.last_minute_prompts) >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_minute_prompts[0])
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
        self.last_minute_prompts.append(time.time())

    def process_sub_prompt(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        sub_prompt = task_info['sub_prompt']
        depth = task_info.get('depth', 0)
        parent_id = task_info.get('parent_id', 'root')
        color = task_info.get('color', '\033[37m')  # Default to white
        node_id = task_info.get('id', str(uuid.uuid4()))

        thread_logger = colorlog.getLogger(node_id)
        thread_logger.setLevel(logging.INFO)

        try:
            with self.active_threads_lock:
                self.active_threads += 1

            thread_logger.info(f"{color}Starting processing of task (depth: {depth}, parent: {parent_id})\033[0m")

            # Check knowledge base for relevant information
            kb_results = self.prompt_graph.query_knowledge_base(sub_prompt)
            additional_context = ""
            if kb_results:
                relevant_node = kb_results[0]["node_id"]
                relevant_text = kb_results[0]["text"]
                additional_context = f"Related information: {relevant_text}\n\n"
                thread_logger.info(f"{color}Found relevant information from node {relevant_node}\033[0m")

            # Generate response using Gemini
            self.check_rate_limit()
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction=SUB_PROMPT_SYSTEM_INSTRUCTION
                ),
                contents=[additional_context + sub_prompt]
            )
            response_text = response.text

            if depth < self.max_depth:
                self.check_rate_limit()
                sub_prompt_response = self.client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SUB_THOUGHT_GENERATOR_SYSTEM_PROMPT
                    ),
                    contents=[
                        f"Previous response: {response_text}",
                        "Generate a maximum of three sub prompts, reply with JUST NO if you dont need sub prompts at all"
                    ]
                )
                r2 = sub_prompt_response.text
                if len(r2.split("\n")) == 1 and "no" in r2.lower():
                    mprint(f"Thread {node_id} did not require further analysis")
                else:
                    for new_prompt in r2.split("\n"):
                        self.add_to_queue(response_text + "\n" + new_prompt, node_id, depth)

            result = {
                'id': node_id,
                'parent_id': parent_id,
                'sub_prompt': sub_prompt,
                'response': response_text,
                'depth': depth,
                'color': color
            }

            # Update node with response
            self.prompt_graph.add_node(node_id, response=response_text)

            mprint(f"{color}[DEPTH: {depth}] {sub_prompt}" + "\n\n" + response_text + "\033[0m")

            return result
        except Exception as e:
            thread_logger.error(f"{color}Error processing sub-prompt: {e}\033[0m")
            # Update node with error
            self.prompt_graph.add_node(node_id, error=str(e))

            return {
                'id': node_id,
                'parent_id': parent_id,
                'sub_prompt': sub_prompt,
                'error': str(e),
                'depth': depth,
                'color': color
            }
        finally:
            with self.active_threads_lock:
                self.active_threads -= 1
                if self.active_threads == 0 and self.task_queue.empty():
                    self.all_tasks_completed.set()

    def worker(self):
        """Worker function that processes tasks from the queue."""
        while True:
            try:
                task = self.task_queue.get(block=False)
                result = self.process_sub_prompt(task)
                self.result_queue.put(result)
                self.task_queue.task_done()
            except queue.Empty:
                # Check if we should exit
                if self.all_tasks_completed.is_set():
                    import time
                    time.sleep(3)
                    if self.all_tasks_completed.is_set():
                        break
                # No tasks available, wait a bit
                import time
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.task_queue.task_done()

    def combine_responses(self, responses: List[Dict[str, Any]], main_prompt: str) -> Dict[str, Any]:
        """Combine all responses with a hierarchical approach."""
        # Format responses using our graph
        formatted_responses = []

        # Helper function to recursively format responses using the graph
        def format_responses_from_graph(node_id='root', depth=0):
            children = self.prompt_graph.get_children(node_id)
            if not children:
                return []

            results = []
            for i, child_id in enumerate(children, 1):
                indent = "  " * depth
                node_data = self.prompt_graph.graph.nodes[child_id]
                prompt = node_data.get('prompt', 'N/A')
                response_text = node_data.get('response', 'No response')

                results.append(f"{indent}Perspective {i} (Level {depth}): {prompt}\n{indent}Response: {response_text}")

                # Process children
                child_results = format_responses_from_graph(child_id, depth + 1)
                results.extend(child_results)

            return results

        # Start with root responses
        formatted_responses = format_responses_from_graph('root')

        combination_prompt = (
            f"Main Prompt: {main_prompt}\n\n"
            f"Hierarchical Responses:\n" +
            "\n".join(formatted_responses) +
            "\n\nSynthesize these responses into a comprehensive summary. "
            "Pay attention to the hierarchy of responses, where deeper levels "
            "are more specific explorations of their parent perspectives. "
            "Create a cohesive narrative that incorporates insights from all levels."
        )

        try:
            self.check_rate_limit()
            combined_response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction=COMBINER_SYSTEM_PROMPT
                ),
                contents=[combination_prompt]
            )

            logger.info("\033[37m[COMBINER] Comprehensive synthesis complete\033[0m")

            # Visualize the graph before returning
            self.prompt_graph.visualize()

            return {
                'main_prompt': main_prompt,
                'combined_response': combined_response.text,
                'graph': self.prompt_graph.to_dict()
            }
        except Exception as e:
            logger.error(f"Response combination error: {e}")
            return {'error': str(e)}

    def process_main_prompt(self, main_prompt: str):
        """Main orchestration method for prompt processing with dynamic thread creation."""
        import time

        # Reset state for new run
        self.all_results = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.all_tasks_completed = threading.Event()
        self.active_threads = 0

        # Reset the graph
        self.prompt_graph = PromptGraph()
        self.prompt_graph.add_node('root', prompt=main_prompt, depth=-1)

        # Generate thought approach sub-prompts
        print(generate_system_prompts(main_prompt, self.client))
        sub_prompts = self.generate_thought_approaches(main_prompt)

        # Initialize the task queue with root tasks
        for i, sub_prompt_info in enumerate(sub_prompts):
            color = self.colors[i % len(self.colors)]
            self.task_queue.put({
                'id': sub_prompt_info['id'],
                'sub_prompt': sub_prompt_info['prompt'],
                'parent_id': 'root',
                'depth': 0,
                'color': color
            })

        # Create worker threads
        num_workers = min(10, len(sub_prompts) * 2)  # Adjust as needed
        workers = []

        for _ in range(num_workers):
            worker = threading.Thread(target=self.worker)
            worker.daemon = True
            worker.start()
            workers.append(worker)

        # Wait for completion or timeout
        max_wait_time = 6000  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # Check if all tasks are completed
            with self.active_threads_lock:
                if self.active_threads == 0 and self.task_queue.empty():
                    self.all_tasks_completed.set()
                    time.sleep(3)
                    if self.active_threads == 0 and self.task_queue.empty():
                        break

            # Get results
            while not self.result_queue.empty():
                self.all_results.append(self.result_queue.get())

            time.sleep(0.1)

        # Signal workers to exit
        self.all_tasks_completed.set()

        # Get any remaining results
        while not self.result_queue.empty():
            self.all_results.append(self.result_queue.get())

        # Wait for workers to complete
        for worker in workers:
            worker.join(timeout=1)

        # Combine responses
        mprint("# Final response is getting evaluated")
        combined_result = self.combine_responses(self.all_results, main_prompt)

        return self.all_results, combined_result

def main():
    api_key = "AIzaSyBe8kjRD-siRLDQh30xVRka5TmrsAZVwYc"
    processor = GeminiPromptProcessor(api_key)

    while True:
        Console().clear()
        main_prompt = input("Enter a problem that can be solved via planning: ")
        mprint(f"# {main_prompt}?")
        results, combined_output = processor.process_main_prompt(main_prompt)

        # Final output processing
        logger.info("\033[37m[FINAL RESULT] Comprehensive Analysis Complete\033[0m")
        mprint(combined_output['combined_response'])

        # Show graph info
        logger.info("\033[37m[GRAPH] Graph structure saved and visualized\033[0m")

        while True:
            choice = input("Would you like to enter a new question? [y/n]:")
            if choice.lower() == 'y':
                break

if __name__ == "__main__":
    main()
