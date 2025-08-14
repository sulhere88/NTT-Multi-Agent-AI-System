"""
Metrics HTTP Server untuk Multi-Agent AI System
Provides /metrics dan /healthz endpoints
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
from config import CONFIG

class MetricsCollector:
    """Collect dan aggregate metrics dari berbagai komponen"""
    
    def __init__(self):
        self.metrics = {
            "system_start_time": datetime.now().isoformat(),
            "total_requests": 0,
            "agent_stats": {},
            "memory_stats": {},
            "validation_stats": {},
            "openai_stats": {},
            "error_counts": {},
            "performance_metrics": {
                "avg_response_time": 0.0,
                "p95_response_time": 0.0,
                "cache_hit_rate": 0.0
            }
        }
        self.request_times = []
        self.last_update = datetime.now()
        
        # Metrics persistence
        self.metrics_history = []
        self.persistence_interval = 300  # 5 minutes
        self.max_history_entries = 288   # 24 hours of 5-minute intervals
        self.last_persistence = datetime.now()
    
    def update_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Update metrics untuk agent tertentu"""
        self.metrics["agent_stats"][agent_id] = {
            **metrics,
            "last_updated": datetime.now().isoformat()
        }
        self.last_update = datetime.now()
    
    def update_memory_metrics(self, memory_stats: Dict[str, Any]):
        """Update memory system metrics"""
        self.metrics["memory_stats"] = {
            **memory_stats,
            "last_updated": datetime.now().isoformat()
        }
        self.last_update = datetime.now()
    
    def update_openai_metrics(self, openai_stats: Dict[str, Any]):
        """Update OpenAI API metrics"""
        self.metrics["openai_stats"] = {
            **openai_stats,
            "last_updated": datetime.now().isoformat()
        }
        
        # Calculate cache hit rate
        if "cache_size" in openai_stats and "total_requests" in openai_stats:
            total_requests = openai_stats["total_requests"]
            if total_requests > 0:
                # Estimasi hit rate berdasarkan cache size vs requests
                estimated_hits = min(openai_stats.get("cache_size", 0), total_requests)
                self.metrics["performance_metrics"]["cache_hit_rate"] = estimated_hits / total_requests
        
        self.last_update = datetime.now()
    
    def record_request_time(self, response_time_ms: float):
        """Record response time untuk performance metrics"""
        self.request_times.append(response_time_ms)
        
        # Keep only last 1000 requests
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        # Update performance metrics
        if self.request_times:
            self.metrics["performance_metrics"]["avg_response_time"] = sum(self.request_times) / len(self.request_times)
            sorted_times = sorted(self.request_times)
            p95_index = int(len(sorted_times) * 0.95)
            self.metrics["performance_metrics"]["p95_response_time"] = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
    
    def increment_error_count(self, error_type: str):
        """Increment error counter"""
        if error_type not in self.metrics["error_counts"]:
            self.metrics["error_counts"][error_type] = 0
        self.metrics["error_counts"][error_type] += 1
        self.last_update = datetime.now()
        
        # Check if we need to persist metrics
        if (datetime.now() - self.last_persistence).total_seconds() >= self.persistence_interval:
            self._persist_metrics()
    
    def _persist_metrics(self):
        """Persist current metrics to history"""
        current_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "metrics": dict(self.metrics),  # Deep copy
            "request_times_count": len(self.request_times),
            "performance": dict(self.metrics["performance_metrics"])
        }
        
        self.metrics_history.append(current_snapshot)
        
        # Trim history if too long
        if len(self.metrics_history) > self.max_history_entries:
            self.metrics_history = self.metrics_history[-self.max_history_entries:]
        
        self.last_persistence = datetime.now()
        logging.debug(f"Metrics persisted: {len(self.metrics_history)} entries in history")
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history untuk specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_history = []
        for entry in self.metrics_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= cutoff_time:
                filtered_history.append(entry)
        
        return filtered_history
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-style metrics"""
        prometheus_metrics = []
        
        # System uptime
        start_time = datetime.fromisoformat(self.metrics["system_start_time"])
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        prometheus_metrics.append(f"multiagent_system_uptime_seconds {uptime_seconds}")
        
        # Total requests
        prometheus_metrics.append(f"multiagent_total_requests {self.metrics['total_requests']}")
        
        # Agent metrics
        for agent_id, stats in self.metrics["agent_stats"].items():
            for metric_name, value in stats.items():
                if isinstance(value, (int, float)):
                    prometheus_metrics.append(f'multiagent_agent_metric{{agent_id="{agent_id}",metric="{metric_name}"}} {value}')
        
        # Memory metrics
        for metric_name, value in self.metrics["memory_stats"].items():
            if isinstance(value, (int, float)):
                prometheus_metrics.append(f"multiagent_memory_{metric_name} {value}")
        
        # OpenAI metrics
        for metric_name, value in self.metrics["openai_stats"].items():
            if isinstance(value, (int, float)):
                prometheus_metrics.append(f"multiagent_openai_{metric_name} {value}")
        
        # Performance metrics
        for metric_name, value in self.metrics["performance_metrics"].items():
            prometheus_metrics.append(f"multiagent_performance_{metric_name} {value}")
        
        # Error counts
        for error_type, count in self.metrics["error_counts"].items():
            prometheus_metrics.append(f'multiagent_errors_total{{error_type="{error_type}"}} {count}')
        
        return "\n".join(prometheus_metrics)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Generate health check status"""
        current_time = datetime.now()
        
        # Check if system is responsive (last update < 5 minutes ago)
        time_since_update = (current_time - self.last_update).total_seconds()
        is_healthy = time_since_update < 300  # 5 minutes
        
        # Check error rate (< 5% errors in last 100 requests)
        total_errors = sum(self.metrics["error_counts"].values())
        error_rate = total_errors / max(self.metrics["total_requests"], 1)
        error_healthy = error_rate < 0.05
        
        # Check memory usage (if available)
        memory_healthy = True
        if "total_memories" in self.metrics["memory_stats"]:
            total_memories = self.metrics["memory_stats"]["total_memories"]
            max_memories = CONFIG.max_episodic_memories + CONFIG.max_semantic_memories
            memory_usage = total_memories / max_memories if max_memories > 0 else 0
            memory_healthy = memory_usage < 0.9  # < 90% usage
        
        overall_healthy = is_healthy and error_healthy and memory_healthy
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": current_time.isoformat(),
            "uptime_seconds": (current_time - datetime.fromisoformat(self.metrics["system_start_time"])).total_seconds(),
            "checks": {
                "system_responsive": is_healthy,
                "low_error_rate": error_healthy,
                "memory_usage_ok": memory_healthy
            },
            "metrics_summary": {
                "total_requests": self.metrics["total_requests"],
                "total_errors": total_errors,
                "error_rate": error_rate,
                "avg_response_time_ms": self.metrics["performance_metrics"]["avg_response_time"],
                "cache_hit_rate": self.metrics["performance_metrics"]["cache_hit_rate"]
            }
        }

# Global metrics collector
metrics_collector = MetricsCollector()

# FastAPI app
app = FastAPI(title="Multi-Agent AI Metrics", version="1.0.0")

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = metrics_collector.get_health_status()
        
        if health_status["status"] == "healthy":
            return JSONResponse(content=health_status, status_code=200)
        else:
            return JSONResponse(content=health_status, status_code=503)
    
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    try:
        # Update metrics dari berbagai komponen
        await _update_live_metrics()
        
        prometheus_metrics = metrics_collector.get_prometheus_metrics()
        return PlainTextResponse(content=prometheus_metrics, media_type="text/plain")
    
    except Exception as e:
        logging.error(f"Metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@app.get("/metrics/json")
async def get_metrics_json():
    """JSON format metrics endpoint"""
    try:
        await _update_live_metrics()
        return JSONResponse(content=metrics_collector.metrics)
    
    except Exception as e:
        logging.error(f"JSON metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSON metrics collection failed: {str(e)}")

@app.get("/metrics/agents")
async def get_agent_metrics():
    """Agent-specific metrics"""
    try:
        await _update_live_metrics()
        return JSONResponse(content=metrics_collector.metrics.get("agent_stats", {}))
    
    except Exception as e:
        logging.error(f"Agent metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Performance metrics endpoint"""
    try:
        await _update_live_metrics()
        return JSONResponse(content=metrics_collector.metrics.get("performance_metrics", {}))
    
    except Exception as e:
        logging.error(f"Performance metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/history")
async def get_metrics_history(hours: int = 1):
    """Metrics history endpoint"""
    try:
        if hours < 1 or hours > 24:
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 24")
        
        history = metrics_collector.get_metrics_history(hours)
        return JSONResponse(content={
            "history": history,
            "entries_count": len(history),
            "time_range_hours": hours
        })
    
    except Exception as e:
        logging.error(f"Metrics history collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/cache")
async def get_cache_metrics():
    """Detailed cache metrics endpoint"""
    try:
        await _update_live_metrics()
        openai_stats = metrics_collector.metrics.get("openai_stats", {})
        
        cache_metrics = {
            "completion_cache": {
                "hits": openai_stats.get("cache_hits", 0),
                "misses": openai_stats.get("cache_misses", 0),
                "hit_rate": openai_stats.get("cache_hit_rate", 0.0),
                "size": openai_stats.get("cache_size", 0)
            },
            "analysis_cache": {
                "hits": openai_stats.get("analysis_cache_hits", 0),
                "misses": openai_stats.get("analysis_cache_misses", 0),
                "hit_rate": openai_stats.get("analysis_cache_hit_rate", 0.0),
                "size": openai_stats.get("analysis_cache_size", 0)
            }
        }
        
        return JSONResponse(content=cache_metrics)
    
    except Exception as e:
        logging.error(f"Cache metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _update_live_metrics():
    """Update metrics dari berbagai komponen sistem"""
    try:
        # Update OpenAI metrics
        from core.shared_resources import openai_manager
        openai_stats = openai_manager.get_stats()
        metrics_collector.update_openai_metrics(openai_stats)
        
        # Update memory metrics (jika tersedia)
        try:
            # Ini akan diupdate ketika memory system dipanggil
            pass
        except:
            pass
        
        # Update total requests
        metrics_collector.metrics["total_requests"] += 1
        
    except Exception as e:
        logging.error(f"Error updating live metrics: {str(e)}")
        metrics_collector.increment_error_count("metrics_update_error")

class MetricsServer:
    """HTTP server untuk metrics endpoints"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.server = None
    
    async def start(self):
        """Start metrics server"""
        try:
            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False  # Reduce noise
            )
            
            self.server = uvicorn.Server(config)
            logging.info(f"Starting metrics server on {self.host}:{self.port}")
            
            # Run server in background task
            await self.server.serve()
            
        except Exception as e:
            logging.error(f"Failed to start metrics server: {str(e)}")
            raise
    
    async def stop(self):
        """Stop metrics server"""
        if self.server:
            self.server.should_exit = True
            logging.info("Metrics server stopped")

# Global server instance
metrics_server = MetricsServer()

async def start_metrics_server():
    """Start metrics server as background task"""
    try:
        await metrics_server.start()
    except Exception as e:
        logging.error(f"Metrics server startup failed: {str(e)}")

def update_agent_metrics(agent_id: str, metrics: Dict[str, Any]):
    """Helper function untuk update agent metrics"""
    metrics_collector.update_agent_metrics(agent_id, metrics)

def update_memory_metrics(memory_stats: Dict[str, Any]):
    """Helper function untuk update memory metrics"""
    metrics_collector.update_memory_metrics(memory_stats)

def record_request_time(response_time_ms: float):
    """Helper function untuk record request time"""
    metrics_collector.record_request_time(response_time_ms)

def increment_error_count(error_type: str):
    """Helper function untuk increment error count"""
    metrics_collector.increment_error_count(error_type)

# Test function
if __name__ == "__main__":
    # Test metrics collection
    metrics_collector.update_agent_metrics("test_agent", {"tasks_completed": 5, "success_rate": 0.95})
    metrics_collector.record_request_time(150.5)
    metrics_collector.increment_error_count("validation_error")
    
    print("Prometheus Metrics:")
    print(metrics_collector.get_prometheus_metrics())
    
    print("\nHealth Status:")
    print(json.dumps(metrics_collector.get_health_status(), indent=2))
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8001)
