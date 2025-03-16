"""
Main entry point for the Expand project.
"""

import os
import logging
import colorlog
from rich.console import Console
from rich.markdown import Markdown
from prompt_processor import GeminiPromptProcessor

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

# Rich console for improved output formatting
console = Console()

def mprint(string):
    """Print markdown-formatted text with clear separation."""
    print("\n\n")
    console.print(Markdown(string))
    print("\n\n")

def main():
    """Main function to run the prompt processor."""
    # In a production environment, get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCz5mTD7c7-mhWPDvyxlTXlb9JFWTzbvuY")
    processor = GeminiPromptProcessor(api_key)
    
    try:
        while True:
            console.clear()
            main_prompt = input("Enter a problem that can be solved via planning: ")
            if not main_prompt.strip():
                continue
                
            mprint(f"# {main_prompt}?")
            results = processor.process_main_prompt(main_prompt)
            
            # Show graph info
            logger.info("\033[37m[GRAPH] Graph structure saved and visualized\033[0m")
            
            # Ask if user wants to continue
            while True:
                choice = input("Would you like to enter a new question? [y/n]: ")
                if choice.lower() in ['y', 'n']:
                    break
            
            if choice.lower() == 'n':
                break
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()