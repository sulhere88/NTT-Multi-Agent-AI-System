"""
Shared Resources untuk Multi-Agent AI System
Singleton instances untuk efisiensi resource
"""

import asyncio
import logging
import hashlib
import os
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import CONFIG

class SingletonMeta(type):
    """Metaclass untuk singleton pattern"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class GlobalEncoder(metaclass=SingletonMeta):
    """Global singleton encoder untuk semua agent"""
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self._cache = {}  # Cache untuk hasil encoding
            self._initialized = True
            logging.info("Global encoder initialized")
    
    def encode_content(self, content: str, use_cache: bool = True):
        """Encode text content dengan caching"""
        if use_cache:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self._cache:
                return self._cache[content_hash]
        
        embedding = self.sentence_transformer.encode(content)
        
        if use_cache:
            # Batasi cache size
            if len(self._cache) > 1000:
                # Hapus 20% cache terlama
                oldest_keys = list(self._cache.keys())[:200]
                for key in oldest_keys:
                    del self._cache[key]
            
            self._cache[content_hash] = embedding
        
        return embedding
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity dengan caching"""
        import numpy as np
        
        embedding1 = self.encode_content(content1)
        embedding2 = self.encode_content(content2)
        
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistik cache"""
        return {
            "cache_size": len(self._cache),
            "cache_limit": 1000
        }

class OpenAIClientManager(metaclass=SingletonMeta):
    """Manager untuk OpenAI client dengan retry dan rate limiting"""
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            # Prefer OpenRouter if OPENROUTER_API_KEY is provided, else fallback to OpenAI
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if openrouter_key:
                default_headers = {}
                referer = os.getenv("OPENROUTER_SITE_URL", "").strip()
                app_title = os.getenv("OPENROUTER_APP_NAME", "NTT Multi-Agent AI System").strip()
                if referer:
                    default_headers["HTTP-Referer"] = referer
                if app_title:
                    default_headers["X-Title"] = app_title

                self.client = openai.OpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers=default_headers or None
                )
                logging.info("OpenAI client manager initialized with OpenRouter endpoint")
            else:
                self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
                logging.info("OpenAI client manager initialized with OpenAI endpoint")
            self._semaphore = asyncio.Semaphore(10)  # Max 10 concurrent calls
            self._request_count = 0
            self._cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
            self._analysis_cache_hits = 0
            self._analysis_cache_misses = 0
            self._initialized = True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def create_completion(
        self, 
        model: str,
        messages: list,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        use_cache: bool = True,
        stream: bool = False
    ):
        """Create completion dengan retry dan rate limiting"""
        
        # Untuk streaming, tidak bisa cache
        if stream:
            return await self._create_streaming_completion(
                model, messages, temperature, max_tokens, response_format
            )
        
        # Cache key berdasarkan parameter
        if use_cache:
            cache_key = hashlib.md5(
                f"{model}{str(messages)}{temperature}{max_tokens}{response_format}".encode()
            ).hexdigest()
            
            if cache_key in self._cache:
                self._cache_hits += 1
                logging.debug(f"Cache hit for completion request")
                return self._cache[cache_key]
            else:
                self._cache_misses += 1
        
        # Acquire rate limiter permission
        await rate_limiter.acquire()
        
        async with self._semaphore:
            self._request_count += 1
            start_time = asyncio.get_event_loop().time()
            
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
                
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    **kwargs
                )
                
                # Record success metrics
                end_time = asyncio.get_event_loop().time()
                response_time_ms = (end_time - start_time) * 1000
                rate_limiter.record_success(response_time_ms)
                
                if use_cache:
                    # Batasi cache size
                    if len(self._cache) > 500:
                        oldest_keys = list(self._cache.keys())[:100]
                        for key in oldest_keys:
                            del self._cache[key]
                    
                    self._cache[cache_key] = response
                
                logging.debug(f"OpenAI API call successful (total: {self._request_count}, {response_time_ms:.1f}ms)")
                return response
                
            except Exception as e:
                # Record error metrics
                error_type = type(e).__name__
                rate_limiter.record_error(error_type)
                logging.error(f"OpenAI API call failed: {str(e)}")
                raise
    
    async def _create_streaming_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None
    ):
        """Create streaming completion"""
        # Acquire rate limiter permission
        await rate_limiter.acquire()
        
        async with self._semaphore:
            self._request_count += 1
            start_time = asyncio.get_event_loop().time()
            
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True
                }
                
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if response_format:
                    kwargs["response_format"] = response_format
                
                # Streaming call
                stream = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    **kwargs
                )
                
                # Record success (initial stream start)
                end_time = asyncio.get_event_loop().time()
                response_time_ms = (end_time - start_time) * 1000
                rate_limiter.record_success(response_time_ms)
                
                logging.debug(f"OpenAI streaming call started (total: {self._request_count}, {response_time_ms:.1f}ms)")
                return stream
                
            except Exception as e:
                # Record error metrics
                error_type = type(e).__name__
                rate_limiter.record_error(error_type)
                logging.error(f"OpenAI streaming call failed: {str(e)}")
                raise
    
    async def stream_completion_text(self, stream) -> str:
        """Collect full text from streaming response"""
        full_content = ""
        
        try:
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    # Yield untuk real-time display
                    yield content
                    
        except Exception as e:
            logging.error(f"Error processing stream: {str(e)}")
            if not full_content:
                raise
        
        return full_content
    
    async def create_cached_analysis(
        self,
        analysis_type: str,
        content: str,
        system_prompt: str,
        temperature: float = 0.3,
        ttl_hours: int = 24
    ):
        """Create analysis dengan caching yang lebih agresif untuk pattern recognition"""
        
        # Enhanced cache key dengan analysis type
        cache_key = hashlib.md5(
            f"{analysis_type}:{content}:{system_prompt}:{temperature}".encode()
        ).hexdigest()
        
        # Check TTL cache
        if hasattr(self, '_analysis_cache'):
            if cache_key in self._analysis_cache:
                cached_item = self._analysis_cache[cache_key]
                import time
                if time.time() - cached_item['timestamp'] < ttl_hours * 3600:
                    self._analysis_cache_hits += 1
                    logging.debug(f"Analysis cache hit for {analysis_type}")
                    return cached_item['result']
                else:
                    # Expired cache entry
                    del self._analysis_cache[cache_key]
                    self._analysis_cache_misses += 1
            else:
                self._analysis_cache_misses += 1
        else:
            self._analysis_cache = {}
            self._analysis_cache_misses += 1
        
        # Generate new analysis
        response = await self.create_completion(
            model=CONFIG.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            use_cache=True
        )
        
        result = response.choices[0].message.content
        
        # Cache dengan TTL
        import time
        self._analysis_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'analysis_type': analysis_type
        }
        
        # Limit cache size
        if len(self._analysis_cache) > 200:
            # Remove oldest 50 items
            sorted_items = sorted(
                self._analysis_cache.items(), 
                key=lambda x: x[1]['timestamp']
            )
            for old_key, _ in sorted_items[:50]:
                del self._analysis_cache[old_key]
        
        logging.debug(f"Analysis cached for {analysis_type}")
        return result
    
    async def create_batch_completions(
        self,
        batch_requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Any]:
        """Create multiple completions in batches untuk efisiensi"""
        
        if not batch_requests:
            return []
        
        logging.info(f"Processing {len(batch_requests)} requests in batches (max concurrent: {max_concurrent})")
        
        # Group requests into batches
        batches = []
        for i in range(0, len(batch_requests), max_concurrent):
            batch = batch_requests[i:i + max_concurrent]
            batches.append(batch)
        
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            logging.debug(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} requests)")
            
            # Create tasks for this batch
            tasks = []
            for request in batch:
                if request.get('use_analysis_cache', False):
                    # Use analysis cache for pattern recognition
                    task = self.create_cached_analysis(
                        analysis_type=request.get('analysis_type', 'batch_analysis'),
                        content=request['user_prompt'],
                        system_prompt=request['system_prompt'],
                        temperature=request.get('temperature', 0.3),
                        ttl_hours=request.get('ttl_hours', 6)
                    )
                else:
                    # Regular completion
                    task = self.create_completion(
                        model=request.get('model', CONFIG.model_name),
                        messages=[
                            {"role": "system", "content": request['system_prompt']},
                            {"role": "user", "content": request['user_prompt']}
                        ],
                        temperature=request.get('temperature', 0.3),
                        max_tokens=request.get('max_tokens'),
                        response_format=request.get('response_format'),
                        use_cache=request.get('use_cache', True)
                    )
                tasks.append(task)
            
            # Execute batch in parallel
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logging.error(f"Batch request {batch_idx * max_concurrent + i} failed: {str(result)}")
                        all_results.append({"error": str(result)})
                    else:
                        # Extract content based on result type
                        if hasattr(result, 'choices'):
                            # OpenAI response object
                            content = result.choices[0].message.content
                        else:
                            # String result from cached analysis
                            content = result
                        
                        all_results.append({"content": content, "success": True})
                
            except Exception as e:
                logging.error(f"Batch {batch_idx + 1} failed: {str(e)}")
                # Add error results for all requests in this batch
                for _ in batch:
                    all_results.append({"error": str(e)})
            
            # Small delay between batches to avoid overwhelming API
            if batch_idx < len(batches) - 1:
                await asyncio.sleep(0.5)
        
        logging.info(f"Batch processing completed: {len(all_results)} results")
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistik penggunaan"""
        analysis_cache_size = len(getattr(self, '_analysis_cache', {}))
        rate_limiter_stats = rate_limiter.get_stats()
        
        # Calculate accurate cache hit rates
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / max(total_cache_requests, 1)
        
        total_analysis_requests = self._analysis_cache_hits + self._analysis_cache_misses
        analysis_cache_hit_rate = self._analysis_cache_hits / max(total_analysis_requests, 1)
        
        return {
            "total_requests": self._request_count,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "analysis_cache_size": analysis_cache_size,
            "analysis_cache_hits": self._analysis_cache_hits,
            "analysis_cache_misses": self._analysis_cache_misses,
            "analysis_cache_hit_rate": analysis_cache_hit_rate,
            "active_semaphore_permits": self._semaphore._value,
            "rate_limiter": rate_limiter_stats
        }

class AdaptiveRateLimiter:
    """Adaptive rate limiter dengan backpressure dinamis"""
    
    def __init__(self, base_max_calls: int = 60, time_window: int = 60):
        # Use profile settings if available
        try:
            from core.env_profiles import get_rate_limit_config
            rate_config = get_rate_limit_config()
            self.base_max_calls = rate_config.base_max_calls
            self.time_window = rate_config.time_window
            self.adaptive_enabled = rate_config.adaptive_enabled
            self.error_threshold = rate_config.error_threshold
        except ImportError:
            # Fallback to defaults
            self.base_max_calls = base_max_calls
            self.time_window = time_window
            self.adaptive_enabled = True
            self.error_threshold = 0.1
        self.calls = []
        
        # Adaptive parameters
        self.current_max_calls = self.base_max_calls
        self.error_count = 0
        self.success_count = 0
        self.response_times = []
        self.last_adjustment = 0
        
        # Thresholds
        self.rtt_threshold = 5000   # 5 second response time
        self.adjustment_interval = 30  # 30 seconds between adjustments
    
    async def acquire(self):
        """Acquire permission dengan adaptive backpressure"""
        import time
        
        now = time.time()
        
        # Periodic adjustment berdasarkan metrics
        if now - self.last_adjustment > self.adjustment_interval:
            await self._adjust_limits()
            self.last_adjustment = now
        
        # Hapus calls yang sudah expired
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.current_max_calls:
            # Adaptive wait time berdasarkan error rate
            oldest_call = min(self.calls)
            base_wait = self.time_window - (now - oldest_call)
            
            # Increase wait time jika banyak error
            error_rate = self.error_count / max(self.error_count + self.success_count, 1)
            if error_rate > self.error_threshold:
                base_wait *= (1 + error_rate * 2)  # Up to 3x wait time
            
            if base_wait > 0:
                logging.warning(f"Adaptive rate limit: waiting {base_wait:.2f}s (limit: {self.current_max_calls})")
                await asyncio.sleep(base_wait)
        
        self.calls.append(now)
    
    async def _adjust_limits(self):
        """Adjust rate limits berdasarkan performance metrics"""
        if self.error_count + self.success_count < 10:
            return  # Not enough data
        
        error_rate = self.error_count / (self.error_count + self.success_count)
        avg_rtt = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        old_limit = self.current_max_calls
        
        # Decrease limit jika terlalu banyak error atau RTT tinggi
        if error_rate > self.error_threshold:
            self.current_max_calls = max(10, int(self.current_max_calls * 0.8))
            logging.warning(f"Rate limit decreased due to error rate: {error_rate:.2%}")
        
        elif avg_rtt > self.rtt_threshold:
            self.current_max_calls = max(10, int(self.current_max_calls * 0.9))
            logging.warning(f"Rate limit decreased due to high RTT: {avg_rtt:.1f}ms")
        
        # Increase limit jika performance bagus
        elif error_rate < 0.05 and avg_rtt < 2000:  # < 5% error, < 2s RTT
            self.current_max_calls = min(self.base_max_calls * 2, int(self.current_max_calls * 1.1))
            logging.info(f"Rate limit increased due to good performance")
        
        if old_limit != self.current_max_calls:
            logging.info(f"Adaptive rate limit: {old_limit} → {self.current_max_calls}")
        
        # Reset counters
        self.error_count = 0
        self.success_count = 0
        self.response_times = self.response_times[-50:]  # Keep last 50
    
    def record_success(self, response_time_ms: float):
        """Record successful API call"""
        self.success_count += 1
        self.response_times.append(response_time_ms)
    
    def record_error(self, error_type: str):
        """Record failed API call"""
        self.error_count += 1
        logging.debug(f"Rate limiter recorded error: {error_type}")
    
    def get_stats(self):
        """Get rate limiter statistics"""
        total_calls = self.error_count + self.success_count
        error_rate = self.error_count / max(total_calls, 1)
        avg_rtt = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "current_limit": self.current_max_calls,
            "base_limit": self.base_max_calls,
            "active_calls": len(self.calls),
            "error_rate": error_rate,
            "avg_response_time_ms": avg_rtt,
            "total_calls": total_calls
        }

class RateLimiter(AdaptiveRateLimiter):
    """Backward compatibility alias"""
    pass

# Global instances
global_encoder = GlobalEncoder()
openai_manager = OpenAIClientManager()
rate_limiter = RateLimiter()

# Secure logging configuration dengan PII redaction
class SecureFormatter(logging.Formatter):
    """Custom formatter yang redact PII dari log messages"""
    
    def format(self, record):
        # Import here to avoid circular imports
        from core.security_utils import redact_log_message
        
        # Format message normal
        formatted = super().format(record)
        
        # Redact PII
        return redact_log_message(formatted)

# Setup secure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)

# Apply secure formatter to all handlers
secure_formatter = SecureFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logging.root.handlers:
    handler.setFormatter(secure_formatter)
