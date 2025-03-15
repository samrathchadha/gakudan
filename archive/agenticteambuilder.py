from rich.console import Console
from rich.markdown import Markdown
console = Console()
import ollama
import threading
import queue
import logging
import colorlog
import uuid
from typing import List, Dict, Any

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

SUB_PROMPT_SYSTEM_INSTRUCTION = "You are an AI assistant. Provide EXTREMELY CONCISE and PRECISE responses. Do not answer with yes or no, build unique content. Provide strong tangible solutions to the task at hand. do not include any introductory messages or conclusive messages"

COMBINER_SYSTEM_PROMPT = 'You have been given several perspectives from many different people.INCLUDE EVERY GOOD POINT ANYONE HAS MENTIONED. You will now create a formal plan with all the work everyone has done. Dont summarize, synthesize a coherent synchronous answer that is a final answer to the original prompt, not just a summary of the answers you have.  do not include any introductory messages or conclusive messages. You do not need to be concise'



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

def generate_system_prompts(query: str, model: str = "llama3.2"):
    global THOUGHT_GENERATOR_SYSTEM_PROMPT, SUB_PROMPT_SYSTEM_INSTRUCTION, COMBINER_SYSTEM_PROMPT
    
    try:
        thought_generator_response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a prompt engineer. Create a system prompt for generating diverse thought approaches. BE VERY CONCISE. SHORT ANSWERS ONLY. LIKE 3 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"},
                {"role": "user", "content": f"Based on this question: '{query}', create a DIRECTION for team members to be made in. Do not tell what team members should be chosen just make reccomendations and hints. You are writing system instructions for the recruiter of the team"}
            ]
        )
        new_thought_generator = thought_generator_response.message.content
        
        # Generate the sub-prompt system instruction
        sub_prompt_response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a prompt engineer. Create VERY concise system instruction. LIKE 4 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"},
                {"role": "user", "content": f"Based on this question: '{query}', create a brief system instruction for responding to sub-prompts. Your prompt should be in the ddirection of HOW to answer, not WHAT to answer. Each system has very specific tasks so this is a guideline of HOW to answer, not what topics to mention. do NOT be generic"}
            ]
        )
        new_sub_prompt = sub_prompt_response.message.content
        
        # Generate the combiner prompt
        combiner_response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a prompt engineer. Create a system prompt for synthesizing multiple perspectives. LIKE 4 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"},
                {"role": "user", "content": f"Based on this question: '{query}', create a system prompt for combining multiple perspectives into one coherent answer. Your system prompt should be more like on how to plan the answer NOT on what to write"}
            ]
        )
        new_combiner = combiner_response.message.content
        
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
    console.print(Markdown(string))
    print("\n\n")

class OllamaPromptProcessor:
    def __init__(self, base_model='llama3.2'):
        self.base_model = base_model
        self.thread_results = {}
        self.combined_result = None
    
    def generate_thought_approaches(self, main_prompt: str) -> List[str]:
        try:
            approaches = ollama.chat(model=self.base_model, messages=[
                {'role': 'system', 'content': THOUGHT_GENERATOR_SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Main Prompt: how to fix {main_prompt}\nGenerate unique sub-prompts. one per line "}
            ])
            mprint(approaches.message.content)
            return [approach.strip() for approach in approaches.message.content.split('\n') if approach.strip()]
        except Exception as e:
            logger.error(f"Error generating thought approaches: {e}")
            return ["a", "b", "c", "d"]

    def process_sub_prompt(self, sub_prompt: str, color: str) -> Dict[str, Any]:
        """Process individual sub-prompt with concise system instruction."""
        thread_id = str(uuid.uuid4())
        thread_logger = colorlog.getLogger(thread_id)
        thread_logger.setLevel(logging.INFO)
        
        try:
            response = ollama.chat(model=self.base_model, messages=[
                {'role': 'system', 'content': SUB_PROMPT_SYSTEM_INSTRUCTION},
                {'role': 'user', 'content': sub_prompt}
            ])
            
            mprint(f"{color}{sub_prompt}"+ "\n\n" + response.message.content+color)
            return {
                'thread_id': thread_id,
                'sub_prompt': sub_prompt,
                'response': response.message.content
            }
        except Exception as e:
            thread_logger.error(f"Error processing sub-prompt: {e}")
            return {
                'thread_id': thread_id,
                'sub_prompt': sub_prompt,
                'error': str(e)
            }

    def combine_responses(self, responses: List[Dict[str, Any]], main_prompt: str) -> Dict[str, Any]:
        formatted_responses = []
        for i, resp in enumerate(responses, 1):
            formatted_responses.append(f"Perspective {i}: {resp.get('sub_prompt', 'N/A')}\nResponse: {resp.get('response', 'No response')}")
        
        combination_prompt = f"Main Prompt: {main_prompt}\n\nResponses:\n" + "\n".join(formatted_responses) + "\n\nSynthesize these responses into a comprehensive summary. do not simply summarize, tie a narrative in. use bold and italics and whatever. be concise."
        
        try:
            combined_response = ollama.chat(model=self.base_model, messages=[
                {'role': 'system', 'content': COMBINER_SYSTEM_PROMPT},
                {'role': 'user', 'content': combination_prompt}
            ])
            
            logger.info("\033[37m[COMBINER] Comprehensive synthesis complete\033[0m")
            return {
                'main_prompt': main_prompt,
                'combined_response': combined_response.message.content,
                'original_responses': responses
            }
        except Exception as e:
            logger.error(f"Response combination error: {e}")
            return {'error': str(e)}


    def process_main_prompt(self, main_prompt: str):
        """Main orchestration method for prompt processing."""
        # Generate thought approach sub-prompts
        print(generate_system_prompts(main_prompt))
        sub_prompts = self.generate_thought_approaches(main_prompt)
        
        # Thread colors
        colors = [
            '\033[31m',  # Red
            '\033[91m',  # Bright Red
            '\033[33m',  # Yellow
            '\033[93m',  # Bright Yellow
            '\033[34m',  # Blue
            '\033[94m',  # Bright Blue
            '\033[35m',  # Magenta
            '\033[95m',  # Bright Magenta
            '\033[36m',  # Cyan
            '\033[96m'   # Bright Cyan
        ]
        
        # Process sub-prompts concurrently
        threads = []
        results = []
        result_queue = queue.Queue()
        
        for i, sub_prompt in enumerate(sub_prompts):
            if i >= len(colors):
                colors.append('\033[37m')  # White
            thread = threading.Thread(
                target=lambda q, sp, c: q.put(self.process_sub_prompt(sp, c)),
                args=(result_queue, sub_prompt, colors[i])
            )
            thread.start()
            threads.append(thread)
        
        # Collect results
        for thread in threads:
            thread.join()
        
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Combine responses
        mprint("# Final response is getting evaluated")
        combined_result = self.combine_responses(results, main_prompt)
        
        return results, combined_result

def main():
    processor = OllamaPromptProcessor()
    while True:
        console.clear()
        main_prompt = input("Enter a problem that can be solved via planning: ")
        mprint(f"# How do we {main_prompt}?")
        results, combined_output = processor.process_main_prompt(main_prompt)
        
        # Final output processing
        logger.info("\033[37m[FINAL RESULT] Comprehensive Analysis Complete\033[0m")
        mprint(combined_output['combined_response'])
        while True:
            choice = input("Would you like to enter a new question? [y/n]:")
            if choice.lower() == 'y':
                break

if __name__ == "__main__":
    main()
