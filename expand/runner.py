#!/usr/bin/env python
"""
Alternative main.py that takes command line arguments instead of using a UI.
Dumps JSON files regularly to track progress.
"""

import os
import sys
import time
import uuid
import logging
import argparse
import colorlog
import numpy as np
from prompt_processor import GeminiPromptProcessor
from prompt_graph import PromptGraph

# Try importing orjson, fall back to standard json
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    import json
    USE_ORJSON = False
    logging.warning("orjson not found, falling back to standard json library. Consider installing orjson for better performance.")

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

def convert_numpy(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle other numpy scalar types
        return obj.item()
    return obj

def dump_json(processor, iteration, output_dir):
    """
    Dump the current state of the graph to a JSON file with improved reliability.
    
    Args:
        processor: The processor with the prompt_graph
        iteration: Current iteration number 
        output_dir: Directory to save files
        
    Returns:
        Path to the created file or None on failure
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create a filename with timestamp and iteration
            timestamp = int(time.time())
            filename = f"expand_{timestamp}_{iteration}.json"
            json_dir = os.path.join(output_dir, "json")
            filepath = os.path.join(json_dir, filename)
            
            # Make sure the json directory exists
            os.makedirs(json_dir, exist_ok=True)
            
            # Log current graph stats before saving
            node_count = len(processor.prompt_graph.graph.nodes())
            edge_count = len(processor.prompt_graph.graph.edges())
            logger.info(f"Dumping graph with {node_count} nodes and {edge_count} edges to {filepath}")
            
            # Use the graph's save_to_json method (now with numpy handling)
            success = processor.prompt_graph.save_to_json(filepath)
            
            # Also save a consistent filename that always has the latest data
            latest_filepath = os.path.join(output_dir, "expand.json")
            success2 = processor.prompt_graph.save_to_json(latest_filepath)
            
            if success and success2:
                # Verify the files were created and have content
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    file_size_kb = os.path.getsize(filepath) / 1024
                    logger.info(f"Successfully saved graph to {filepath} ({file_size_kb:.2f} KB)")
                    
                    # Also verify latest file
                    latest_size_kb = os.path.getsize(latest_filepath) / 1024
                    logger.info(f"Also saved to {latest_filepath} ({latest_size_kb:.2f} KB)")
                    
                    return filepath
                else:
                    logger.error(f"Failed to create file {filepath} or file is empty")
                    retry_count += 1
            else:
                logger.error(f"Failed to save one or both JSON files (attempt {retry_count+1}/{max_retries})")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
                
        except Exception as e:
            logger.error(f"Error dumping JSON (attempt {retry_count+1}/{max_retries}): {e}", exc_info=True)
            retry_count += 1
            time.sleep(2)  # Wait before retrying
    
    logger.critical(f"Failed to dump JSON after {max_retries} attempts")
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

def verify_session_dir(output_dir):
    """Verify the session directory is properly set up."""
    try:
        # Test write permissions by creating a test file
        test_file = os.path.join(output_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("Testing write permissions")
        
        # Cleanup test file
        os.remove(test_file)
        
        # Create json directory
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        
        # Test json directory write
        test_json = os.path.join(json_dir, "test.json")
        if USE_ORJSON:
            with open(test_json, 'wb') as f:
                f.write(orjson.dumps({"test": "data"}))
        else:
            with open(test_json, 'w') as f:
                json.dump({"test": "data"}, f)
                
        # Cleanup test json
        os.remove(test_json)
        
        logger.info(f"Successfully verified write permissions in {output_dir}")
        return True
    except Exception as e:
        logger.critical(f"Failed to verify session directory: {e}", exc_info=True)
        return False

def main():
    """Main function to run the prompt processor with command line arguments."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Change to the specified working directory
    output_dir = args.cwd
    try:
        os.chdir(output_dir)
        logger.info(f"Changed working directory to: {output_dir}")
    except Exception as e:
        logger.critical(f"Failed to change to directory {output_dir}: {e}")
        sys.exit(1)
    
    # Verify session directory setup
    if not verify_session_dir(output_dir):
        logger.critical("Session directory verification failed. Exiting.")
        sys.exit(1)
    
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
        
        # Attempt emergency dump
        try:
            emergency_dump = os.path.join(output_dir, "emergency_dump.json")
            logger.info(f"Attempting emergency dump to {emergency_dump}")
            processor.prompt_graph.save_to_json(emergency_dump)
            logger.info(f"Emergency dump successful")
        except Exception as dump_error:
            logger.critical(f"Emergency dump failed: {dump_error}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()