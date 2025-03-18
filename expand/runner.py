#!/usr/bin/env python
"""
Alternative main.py that takes command line arguments instead of using a UI.
Dumps JSON files regularly to track progress.
"""

import os
import sys
import json
import time
import uuid
import logging
import argparse
import colorlog
from prompt_processor import GeminiPromptProcessor
from prompt_graph import PromptGraph

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

def dump_json(processor, iteration, output_dir):
    """Dump the current state of the graph to a JSON file."""
    try:
        # Create a filename with timestamp and iteration
        timestamp = int(time.time())
        filename = f"expand_{timestamp}_{iteration}.json"
        filepath = os.path.join(output_dir, "json", filename)
        
        # Make sure the json directory exists
        os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
        
        # Use the graph's save_to_json method
        processor.prompt_graph.save_to_json(filepath)
        
        # Also save a consistent filename that always has the latest data
        latest_filepath = os.path.join(output_dir, "expand.json")
        processor.prompt_graph.save_to_json(latest_filepath)
        
        logger.info(f"Saved graph to {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error dumping JSON: {e}")
        return None

# Create a subclass of GeminiPromptProcessor that dumps JSON after each prompt
class DumpingPromptProcessor(GeminiPromptProcessor):
    def __init__(self, api_key, output_dir):
        super().__init__(api_key)
        self.output_dir = output_dir
        self.json_iteration = 0
    
    def process_sub_prompt(self, task_info):
        # Call the original method
        result = super().process_sub_prompt(task_info)
        
        # Dump JSON after processing
        self.json_iteration += 1
        dump_json(self, self.json_iteration, self.output_dir)
        
        return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Expand prompt processor without UI.')
    parser.add_argument('--cwd', required=True, help='Working directory for output')
    parser.add_argument('--prompt', required=True, help='Prompt to process')
    parser.add_argument('--api-key', required=True, help='API Key for Google Generative AI')
    
    return parser.parse_args()

def main():
    """Main function to run the prompt processor with command line arguments."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Change to the specified working directory
    output_dir = args.cwd
    os.chdir(output_dir)
    logger.info(f"Changed working directory to: {output_dir}")
    
    # Create directories for output
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    
    # Initialize our custom processor that automatically dumps JSON
    processor = DumpingPromptProcessor(args.api_key, output_dir)
    
    try:
        # Initial dump
        dump_json(processor, 0, output_dir)
        
        # Process the prompt
        logger.info(f"Processing prompt: {args.prompt}")
        results = processor.process_main_prompt(args.prompt)
        
        # Final dump
        dump_json(processor, 999, output_dir)  # Use a high number for the final dump
        
        logger.info("Processing completed")
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()