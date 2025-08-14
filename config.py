"""
Configuration settings for Multi-Agent AI System
Based on NTT's Multi-Agent AI Technology
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    max_episodic_memories: int = 100
    max_semantic_memories: int = 50
    memory_decay_rate: float = 0.1
    collaboration_threshold: float = 0.7
    expertise_level: float = 0.8
    
@dataclass
class SystemConfig:
    """Global system configuration"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model_name: str = "gpt-4-turbo-preview"
    max_agents: int = 10
    meeting_duration: int = 300  # seconds
    consensus_threshold: float = 0.75
    knowledge_similarity_threshold: float = 0.8
    vector_db_path: str = "./chroma_db"
    
    # Meeting configurations
    team_meeting_frequency: int = 3  # every 3 tasks
    production_meeting_frequency: int = 5  # every 5 tasks
    
    # Quality assurance
    cross_check_rounds: int = 3
    validation_threshold: float = 0.85
    
# Global configuration instance
CONFIG = SystemConfig()

# Agent role templates
AGENT_ROLES = {
    "project_manager": {
        "description": "Coordinates overall project execution and ensures deliverables meet requirements",
        "expertise_areas": ["project_management", "coordination", "quality_assurance"],
        "collaboration_priority": 1.0
    },
    "business_analyst": {
        "description": "Analyzes business requirements and market conditions",
        "expertise_areas": ["business_analysis", "market_research", "requirements_gathering"],
        "collaboration_priority": 0.9
    },
    "creative_strategist": {
        "description": "Develops creative solutions and innovative approaches",
        "expertise_areas": ["creative_strategy", "innovation", "design_thinking"],
        "collaboration_priority": 0.8
    },
    "technical_specialist": {
        "description": "Provides technical expertise and implementation guidance",
        "expertise_areas": ["technical_implementation", "system_design", "feasibility_analysis"],
        "collaboration_priority": 0.8
    },
    "marketing_expert": {
        "description": "Develops marketing strategies and customer engagement approaches",
        "expertise_areas": ["marketing_strategy", "customer_engagement", "brand_development"],
        "collaboration_priority": 0.7
    },
    "quality_assurance": {
        "description": "Ensures output quality and validates solutions",
        "expertise_areas": ["quality_control", "validation", "testing"],
        "collaboration_priority": 0.9
    }
}

# Task complexity levels
TASK_COMPLEXITY = {
    "simple": {
        "min_agents": 2,
        "max_agents": 3,
        "meeting_rounds": 1,
        "validation_rounds": 1
    },
    "moderate": {
        "min_agents": 3,
        "max_agents": 5,
        "meeting_rounds": 2,
        "validation_rounds": 2
    },
    "complex": {
        "min_agents": 5,
        "max_agents": 8,
        "meeting_rounds": 3,
        "validation_rounds": 3
    },
    "enterprise": {
        "min_agents": 6,
        "max_agents": 10,
        "meeting_rounds": 4,
        "validation_rounds": 4
    }
}
