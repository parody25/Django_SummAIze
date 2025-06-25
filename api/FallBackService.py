import time
import functools
import threading

class TimeoutException(Exception):
    pass

FALLBACK_CONFIG = {
    "RETRIES": 3,
    "TIMEOUT_SECONDS": 30,
    "RETRY_DELAY_SECONDS": 1,
}

def timeout(seconds):
    """Decorator to timeout a function after X seconds."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds.")]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds.")
            elif isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

class FallBackService:
    def __init__(self, config=None):
        config = config or FALLBACK_CONFIG
        self.retries = config["RETRIES"]
        self.timeout_seconds = config["TIMEOUT_SECONDS"]
        self.retry_delay = config["RETRY_DELAY_SECONDS"]

    def execute(self, func, *args, **kwargs):
        """Executes a function with retry and timeout fallback."""
        for attempt in range(1, self.retries + 1):
            try:
                timed_func = timeout(self.timeout_seconds)(func)
                return timed_func(*args, **kwargs)
            except Exception as e:
                print(f"[FallbackService] Attempt {attempt} failed: {str(e)}")
                if attempt == self.retries:
                    raise e
                time.sleep(self.retry_delay)
