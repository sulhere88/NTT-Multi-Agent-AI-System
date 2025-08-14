"""
Environment Profiles untuk Multi-Agent AI System
Mendukung dev/staging/prod dan policy biaya vs kualitas
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from config import CONFIG

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"

class CostPolicy(Enum):
    """Cost optimization policies"""
    COST_OPTIMIZED = "cost_optimized"    # Minimize costs
    BALANCED = "balanced"                # Balance cost vs quality
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality

@dataclass
class ModelConfig:
    """Configuration untuk model selection berdasarkan policy"""
    primary_model: str
    fallback_model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int

@dataclass
class ValidationConfig:
    """Configuration untuk validation berdasarkan policy"""
    min_validators: int
    max_validators: int
    consensus_threshold: float
    enable_cross_validation: bool
    batch_size: int

@dataclass
class MemoryConfig:
    """Configuration untuk memory management"""
    max_episodic_memories: int
    max_semantic_memories: int
    consolidation_threshold: int
    ttl_episodic_days: int
    ttl_semantic_days: int
    eviction_batch_size: int

@dataclass
class RateLimitConfig:
    """Configuration untuk rate limiting"""
    base_max_calls: int
    time_window: int
    max_concurrent: int
    adaptive_enabled: bool
    error_threshold: float

@dataclass
class EnvironmentProfile:
    """Complete environment profile"""
    environment: Environment
    cost_policy: CostPolicy
    model_config: ModelConfig
    validation_config: ValidationConfig
    memory_config: MemoryConfig
    rate_limit_config: RateLimitConfig
    features: Dict[str, bool]
    logging_level: str
    metrics_enabled: bool
    debug_mode: bool

class ProfileManager:
    """Manager untuk environment profiles"""
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
        self.current_profile = self._detect_environment()
        
        # Apply current profile
        self._apply_profile(self.current_profile)
        
        logging.info(f"Environment profile loaded: {self.current_profile.environment.value} ({self.current_profile.cost_policy.value})")
    
    def _initialize_profiles(self) -> Dict[str, EnvironmentProfile]:
        """Initialize predefined profiles"""
        profiles = {}
        
        # Development Profile - Cost Optimized
        profiles["dev_cost"] = EnvironmentProfile(
            environment=Environment.DEVELOPMENT,
            cost_policy=CostPolicy.COST_OPTIMIZED,
            model_config=ModelConfig(
                primary_model="gpt-3.5-turbo",
                fallback_model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=1000,
                timeout_seconds=30
            ),
            validation_config=ValidationConfig(
                min_validators=1,
                max_validators=2,
                consensus_threshold=0.6,
                enable_cross_validation=False,
                batch_size=3
            ),
            memory_config=MemoryConfig(
                max_episodic_memories=100,
                max_semantic_memories=50,
                consolidation_threshold=10,
                ttl_episodic_days=7,
                ttl_semantic_days=30,
                eviction_batch_size=10
            ),
            rate_limit_config=RateLimitConfig(
                base_max_calls=30,
                time_window=60,
                max_concurrent=3,
                adaptive_enabled=True,
                error_threshold=0.15
            ),
            features={
                "streaming": False,
                "batch_processing": False,
                "advanced_caching": True,
                "pii_redaction": False,
                "audit_logging": False
            },
            logging_level="DEBUG",
            metrics_enabled=True,
            debug_mode=True
        )
        
        # Development Profile - Balanced
        profiles["dev_balanced"] = EnvironmentProfile(
            environment=Environment.DEVELOPMENT,
            cost_policy=CostPolicy.BALANCED,
            model_config=ModelConfig(
                primary_model="gpt-4o-mini",
                fallback_model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=2000,
                timeout_seconds=45
            ),
            validation_config=ValidationConfig(
                min_validators=2,
                max_validators=3,
                consensus_threshold=0.7,
                enable_cross_validation=True,
                batch_size=5
            ),
            memory_config=MemoryConfig(
                max_episodic_memories=200,
                max_semantic_memories=100,
                consolidation_threshold=15,
                ttl_episodic_days=14,
                ttl_semantic_days=60,
                eviction_batch_size=20
            ),
            rate_limit_config=RateLimitConfig(
                base_max_calls=45,
                time_window=60,
                max_concurrent=5,
                adaptive_enabled=True,
                error_threshold=0.12
            ),
            features={
                "streaming": True,
                "batch_processing": True,
                "advanced_caching": True,
                "pii_redaction": True,
                "audit_logging": True
            },
            logging_level="INFO",
            metrics_enabled=True,
            debug_mode=True
        )
        
        # Staging Profile - Quality Optimized
        profiles["staging_quality"] = EnvironmentProfile(
            environment=Environment.STAGING,
            cost_policy=CostPolicy.QUALITY_OPTIMIZED,
            model_config=ModelConfig(
                primary_model="gpt-4o",
                fallback_model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=4000,
                timeout_seconds=60
            ),
            validation_config=ValidationConfig(
                min_validators=3,
                max_validators=5,
                consensus_threshold=0.8,
                enable_cross_validation=True,
                batch_size=8
            ),
            memory_config=MemoryConfig(
                max_episodic_memories=500,
                max_semantic_memories=250,
                consolidation_threshold=25,
                ttl_episodic_days=30,
                ttl_semantic_days=90,
                eviction_batch_size=50
            ),
            rate_limit_config=RateLimitConfig(
                base_max_calls=60,
                time_window=60,
                max_concurrent=8,
                adaptive_enabled=True,
                error_threshold=0.08
            ),
            features={
                "streaming": True,
                "batch_processing": True,
                "advanced_caching": True,
                "pii_redaction": True,
                "audit_logging": True
            },
            logging_level="INFO",
            metrics_enabled=True,
            debug_mode=False
        )
        
        # Production Profile - Balanced (default for prod)
        profiles["prod_balanced"] = EnvironmentProfile(
            environment=Environment.PRODUCTION,
            cost_policy=CostPolicy.BALANCED,
            model_config=ModelConfig(
                primary_model="gpt-4o-mini",
                fallback_model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=3000,
                timeout_seconds=45
            ),
            validation_config=ValidationConfig(
                min_validators=2,
                max_validators=4,
                consensus_threshold=0.75,
                enable_cross_validation=True,
                batch_size=6
            ),
            memory_config=MemoryConfig(
                max_episodic_memories=1000,
                max_semantic_memories=500,
                consolidation_threshold=50,
                ttl_episodic_days=30,
                ttl_semantic_days=90,
                eviction_batch_size=100
            ),
            rate_limit_config=RateLimitConfig(
                base_max_calls=60,
                time_window=60,
                max_concurrent=10,
                adaptive_enabled=True,
                error_threshold=0.05
            ),
            features={
                "streaming": True,
                "batch_processing": True,
                "advanced_caching": True,
                "pii_redaction": True,
                "audit_logging": True
            },
            logging_level="WARNING",
            metrics_enabled=True,
            debug_mode=False
        )
        
        # Production Profile - Quality Optimized
        profiles["prod_quality"] = EnvironmentProfile(
            environment=Environment.PRODUCTION,
            cost_policy=CostPolicy.QUALITY_OPTIMIZED,
            model_config=ModelConfig(
                primary_model="gpt-4o",
                fallback_model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=4000,
                timeout_seconds=60
            ),
            validation_config=ValidationConfig(
                min_validators=3,
                max_validators=6,
                consensus_threshold=0.85,
                enable_cross_validation=True,
                batch_size=10
            ),
            memory_config=MemoryConfig(
                max_episodic_memories=2000,
                max_semantic_memories=1000,
                consolidation_threshold=100,
                ttl_episodic_days=60,
                ttl_semantic_days=180,
                eviction_batch_size=200
            ),
            rate_limit_config=RateLimitConfig(
                base_max_calls=100,
                time_window=60,
                max_concurrent=15,
                adaptive_enabled=True,
                error_threshold=0.03
            ),
            features={
                "streaming": True,
                "batch_processing": True,
                "advanced_caching": True,
                "pii_redaction": True,
                "audit_logging": True
            },
            logging_level="ERROR",
            metrics_enabled=True,
            debug_mode=False
        )
        
        return profiles
    
    def _detect_environment(self) -> EnvironmentProfile:
        """Detect environment berdasarkan env vars atau config"""
        
        # Check environment variables
        env = os.getenv("MULTIAGENT_ENV", "dev").lower()
        policy = os.getenv("MULTIAGENT_POLICY", "balanced").lower()
        
        # Map to profile key
        profile_key = f"{env}_{policy}"
        
        # Fallback logic
        if profile_key not in self.profiles:
            if env == "dev":
                profile_key = "dev_balanced"
            elif env == "staging":
                profile_key = "staging_quality"
            elif env == "prod":
                profile_key = "prod_balanced"
            else:
                profile_key = "dev_balanced"  # Ultimate fallback
        
        return self.profiles[profile_key]
    
    def _apply_profile(self, profile: EnvironmentProfile):
        """Apply profile settings ke system components"""
        
        # Update CONFIG dengan profile settings
        CONFIG.model_name = profile.model_config.primary_model
        CONFIG.max_episodic_memories = profile.memory_config.max_episodic_memories
        CONFIG.max_semantic_memories = profile.memory_config.max_semantic_memories
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, profile.logging_level))
        
        logging.info(f"Applied profile: {profile.environment.value} ({profile.cost_policy.value})")
        logging.info(f"Model: {profile.model_config.primary_model}")
        logging.info(f"Validation: {profile.validation_config.min_validators}-{profile.validation_config.max_validators} validators")
        logging.info(f"Memory: {profile.memory_config.max_episodic_memories}E/{profile.memory_config.max_semantic_memories}S")
        logging.info(f"Features: {[k for k, v in profile.features.items() if v]}")
    
    def switch_profile(self, profile_key: str) -> bool:
        """Switch ke profile lain"""
        if profile_key not in self.profiles:
            logging.error(f"Profile not found: {profile_key}")
            return False
        
        self.current_profile = self.profiles[profile_key]
        self._apply_profile(self.current_profile)
        
        logging.info(f"Switched to profile: {profile_key}")
        return True
    
    def get_current_profile(self) -> EnvironmentProfile:
        """Get current active profile"""
        return self.current_profile
    
    def list_profiles(self) -> Dict[str, str]:
        """List available profiles"""
        return {
            key: f"{profile.environment.value} ({profile.cost_policy.value})"
            for key, profile in self.profiles.items()
        }
    
    def export_profile(self, profile_key: str = None) -> Dict[str, Any]:
        """Export profile configuration"""
        profile = self.profiles.get(profile_key, self.current_profile)
        return asdict(profile)
    
    def create_custom_profile(
        self, 
        name: str, 
        base_profile: str,
        overrides: Dict[str, Any]
    ) -> bool:
        """Create custom profile berdasarkan base profile"""
        
        if base_profile not in self.profiles:
            logging.error(f"Base profile not found: {base_profile}")
            return False
        
        # Clone base profile
        base = self.profiles[base_profile]
        custom_data = asdict(base)
        
        # Apply overrides
        def update_nested_dict(d, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in d:
                    update_nested_dict(d[key], value)
                else:
                    d[key] = value
        
        update_nested_dict(custom_data, overrides)
        
        # Create new profile
        try:
            # Reconstruct nested objects
            custom_data['environment'] = Environment(custom_data['environment'])
            custom_data['cost_policy'] = CostPolicy(custom_data['cost_policy'])
            custom_data['model_config'] = ModelConfig(**custom_data['model_config'])
            custom_data['validation_config'] = ValidationConfig(**custom_data['validation_config'])
            custom_data['memory_config'] = MemoryConfig(**custom_data['memory_config'])
            custom_data['rate_limit_config'] = RateLimitConfig(**custom_data['rate_limit_config'])
            
            custom_profile = EnvironmentProfile(**custom_data)
            self.profiles[name] = custom_profile
            
            logging.info(f"Created custom profile: {name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create custom profile: {str(e)}")
            return False
    
    def get_policy_recommendations(self, task_complexity: str, budget_priority: str) -> str:
        """Get policy recommendation berdasarkan task dan budget"""
        
        recommendations = {
            ("simple", "high"): "dev_cost",
            ("simple", "medium"): "dev_balanced", 
            ("simple", "low"): "staging_quality",
            ("medium", "high"): "dev_balanced",
            ("medium", "medium"): "staging_quality",
            ("medium", "low"): "prod_quality",
            ("complex", "high"): "staging_quality",
            ("complex", "medium"): "prod_balanced",
            ("complex", "low"): "prod_quality"
        }
        
        return recommendations.get((task_complexity, budget_priority), "dev_balanced")

# Global profile manager
profile_manager = ProfileManager()

def get_current_profile() -> EnvironmentProfile:
    """Get current active profile"""
    return profile_manager.get_current_profile()

def is_feature_enabled(feature_name: str) -> bool:
    """Check if feature is enabled in current profile"""
    return profile_manager.current_profile.features.get(feature_name, False)

def get_model_config() -> ModelConfig:
    """Get current model configuration"""
    return profile_manager.current_profile.model_config

def get_validation_config() -> ValidationConfig:
    """Get current validation configuration"""
    return profile_manager.current_profile.validation_config

def get_memory_config() -> MemoryConfig:
    """Get current memory configuration"""
    return profile_manager.current_profile.memory_config

def get_rate_limit_config() -> RateLimitConfig:
    """Get current rate limit configuration"""
    return profile_manager.current_profile.rate_limit_config

# Test function
if __name__ == "__main__":
    print("Available profiles:")
    for key, desc in profile_manager.list_profiles().items():
        print(f"  {key}: {desc}")
    
    print(f"\nCurrent profile: {profile_manager.current_profile.environment.value} ({profile_manager.current_profile.cost_policy.value})")
    print(f"Model: {profile_manager.current_profile.model_config.primary_model}")
    print(f"Features enabled: {[k for k, v in profile_manager.current_profile.features.items() if v]}")
