"""
Rate limiter implementation for API calls.
Ensures that we don't exceed API rate limits when making requests.
"""

import threading
import time
import logging

class RateLimiter:
    """
    Implements rate limiting for API calls with a sliding window approach.
    Ensures no more than max_requests are made within the time_window.
    """
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        """
        Initialize rate limiter with configurable parameters.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = []
        self.lock = threading.Lock()
        logging.info(f"Rate limiter initialized: {max_requests} requests per {time_window} seconds")
    
    def wait_if_needed(self) -> float:
        """
        Check if rate limit is reached and wait if necessary.
        
        Returns:
            float: Time waited in seconds (0 if no waiting was needed)
        """
        current_time = time.time()
        wait_time = 0
        
        with self.lock:
            # Remove timestamps older than time_window
            self.request_timestamps = [
                t for t in self.request_timestamps 
                if current_time - t < self.time_window
            ]
            
            # If we're at the limit, calculate wait time
            if len(self.request_timestamps) >= self.max_requests:
                oldest_timestamp = min(self.request_timestamps)
                wait_time = self.time_window - (current_time - oldest_timestamp) + 0.1  # Small buffer
        
        # Wait outside the lock if needed
        if wait_time > 0:
            logging.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # Recursive call to check again after waiting
            return wait_time + self.wait_if_needed()
        
        # Add current timestamp and proceed
        with self.lock:
            self.request_timestamps.append(time.time())
        
        return wait_time
    
    def __call__(self, func):
        """
        Decorator for rate-limiting functions.
        
        Args:
            func: The function to rate-limit
            
        Returns:
            Wrapped function that respects rate limits
        """
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper