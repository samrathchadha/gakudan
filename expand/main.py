"""
Main entry point for the Expand project.
Uses database for all storage - no file operations.
"""

import os
import logging
import colorlog
from rich.console import Console
from rich.markdown import Markdown
from prompt_processor import GeminiPromptProcessor
import sys

# Add parent directory to path to import database modules
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import database modules
import db
from db_manager import db_manager

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

class DatabaseGeminiPromptProcessor(GeminiPromptProcessor):
    """Extended prompt processor that uses database for all storage."""
    def __init__(self, api_key, session_id=None):
        super().__init__(api_key)
        self.session_id = session_id
        
        # Initialize database
        db.initialize_db()
        
        logger.info("Using database for all storage operations")
        
    def save_results(self, main_prompt, results):
        """Save results to database instead of files."""
        if not self.session_id:
            # Create a new session if none exists
            self.session_id, client_id = db_manager.create_session(main_prompt, "local_run")
            logger.info(f"Created new session {self.session_id} for database storage")
        
        # Format the data for database
        data = self.prompt_graph.format_for_database()
        
        # Save contract data
        db_manager.save_contract_data(self.session_id, data)
        
        # Update session status
        db_manager.update_session_status(self.session_id, "completed")
        
        logger.info(f"All results saved to database for session {self.session_id}")
        return self.session_id
        
    def process_main_prompt(self, main_prompt):
        """Override to ensure database saving."""
        # Process normally
        results = super().process_main_prompt(main_prompt)
        
        # Save to database
        session_id = self.save_results(main_prompt, results)
        
        # Return results with session ID
        return {
            'session_id': session_id,
            'results': results
        }

def main():
    """Main function to run the prompt processor."""
    # In a production environment, get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyB6z0QzdzpPdsBDxP9956mNtwsyqKVhfsk")
    processor = DatabaseGeminiPromptProcessor(api_key)
    
    try:
        while True:
            console.clear()
            main_prompt = input("Enter a problem that can be solved via planning: ")
            if not main_prompt.strip():
                continue
                
            mprint(f"# {main_prompt}?")
            
            # Process with database storage
            result_data = processor.process_main_prompt(main_prompt)
            session_id = result_data['session_id']
            
            # Show graph info
            logger.info(f"\033[37m[GRAPH] Graph structure saved to database with session ID: {session_id}\033[0m")
            
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
    # Initialize database
    db.initialize_db()
    main()
