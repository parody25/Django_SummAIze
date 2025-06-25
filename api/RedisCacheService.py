import hashlib
from django.core.cache import cache
from typing import Any, Optional, Callable, Tuple, Dict
import json
import functools
from threading import Lock
from django.conf import settings

class RedisCacheService:
    VERSION = "v1"
    _cache_hits = 0
    _cache_misses = 0
    _lock = Lock()

    @staticmethod
    def generate_cache_key(prefix: str, *args: Any, debug: bool = False) -> str:
        """
        This function Generates a consistent cache key with versioning and debug support.
            
        """
        arg_string = ":".join(str(arg) for arg in args)
        
        if debug or getattr(settings, 'DEBUG', False):
            # Debug mode - use sanitized readable keys
            sanitized = "".join(c for c in arg_string if c.isalnum() or c in (':', '-', '_'))
            key = f"{RedisCacheService.VERSION}:{prefix}:{sanitized}"[:250]
        else:
            # Production mode - use hash
            arg_hash = hashlib.md5(arg_string.encode()).hexdigest()
            key = f"{RedisCacheService.VERSION}:{prefix}:{arg_hash}"
        
        if len(key) > 1000:
            raise ValueError("Cache key too long")
        return key
    
    @staticmethod
    def get_cached_data(key: str) -> Any:
        """
        This function Retrieves data from cache with metrics tracking.
        """
        data = cache.get(key)
        if data is not None:
            RedisCacheService._cache_hits += 1
        else:
            RedisCacheService._cache_misses += 1
        return data
    
    @staticmethod
    def set_cached_data(key: str, data: Any, timeout: int = None) -> None:
        """
        This function Stores data in cache with validation and default timeout of 10 mins.
        """
        if timeout is None:
            timeout = getattr(settings, 'CACHE_DEFAULT_TIMEOUT', 600)
        
        try:
            json.dumps(data)  # Validate serialization
            cache.set(key, data, timeout=timeout)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Data not cacheable: {str(e)}")
    
    @staticmethod
    def get_or_set(key: str, default_func: Callable[[], Any], timeout: int = None) -> Any:
        """
        Atomic get-or-set operation to prevent cache stampede.
        """
        with RedisCacheService._lock:
            data = RedisCacheService.get_cached_data(key)
            if data is not None:
                return data
            
            data = default_func()
            RedisCacheService.set_cached_data(key, data, timeout=timeout)
            return data
    
    @staticmethod
    def cached_function(
        prefix: str,
        timeout: int = None,
        key_args: Optional[Tuple[int]] = None,
        should_cache: Callable[[Any], bool] = lambda result: True,
        debug: bool = False
    ) -> Callable:
        """
        Enhanced decorator with metrics and debug support.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_args = args
                if key_args is not None:
                    cache_args = tuple(args[i] for i in key_args if i < len(args))
                
                cache_key = RedisCacheService.generate_cache_key(
                    prefix, *cache_args, debug=debug
                )
                
                return RedisCacheService.get_or_set(
                    cache_key,
                    lambda: func(*args, **kwargs) if should_cache(func(*args, **kwargs)) else None,
                    timeout=timeout
                )
            return wrapper
        return decorator
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """
        Returns cache performance metrics.
        """
        total = RedisCacheService._cache_hits + RedisCacheService._cache_misses
        return {
            'hits': RedisCacheService._cache_hits,
            'misses': RedisCacheService._cache_misses,
            'ratio': RedisCacheService._cache_hits / total if total > 0 else 0,
            'version': RedisCacheService.VERSION
        }
    
    @staticmethod
    def delete_cache_key(key: str) -> None:
        """Deletes a specific key from cache."""
        cache.delete(key)
    
    @staticmethod
    def clear_cache_pattern(pattern: str) -> None:
        """Clears keys matching pattern (use sparingly)."""
        keys = cache.keys(f"{RedisCacheService.VERSION}:{pattern}")
        if keys:
            cache.delete_many(keys)