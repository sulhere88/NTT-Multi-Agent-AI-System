"""
Advanced AI Agent Implementation with Reusability and Knowledge Accumulation
Based on NTT's Multi-Agent AI Technology
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
from config import CONFIG, AGENT_ROLES
from core.memory import AdvancedMemorySystem, EpisodicMemory, SemanticMemory
from core.communication import CommunicationProtocol, Message, MessageType
from core.validation import KnowledgeValidator, ValidationRequest, ValidationType
import pickle
import os

class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    COLLABORATING = "collaborating"
    LEARNING = "learning"
    VALIDATING = "validating"
    MEETING = "meeting"
    SUSPENDED = "suspended"
    ERROR = "error"

class AgentCapability(Enum):
    """Agent capabilities"""
    ANALYSIS = "analysis"
    CREATIVE_THINKING = "creative_thinking"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    PROJECT_MANAGEMENT = "project_management"
    QUALITY_ASSURANCE = "quality_assurance"
    RESEARCH = "research"
    COMMUNICATION = "communication"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"

@dataclass
class AgentProfile:
    """Agent profile and characteristics"""
    agent_id: str
    name: str
    role: str
    expertise_domains: List[str]
    capabilities: List[AgentCapability]
    personality_traits: Dict[str, float]  # e.g., {"collaborative": 0.8, "analytical": 0.9}
    experience_level: float  # 0.0 to 1.0
    collaboration_preference: float  # 0.0 (individual) to 1.0 (highly collaborative)
    learning_rate: float  # How quickly agent learns from experiences
    created_at: datetime
    last_active: datetime
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_active, str):
            self.last_active = datetime.fromisoformat(self.last_active)

@dataclass
class TaskExecution:
    """Record of task execution"""
    task_id: str
    agent_id: str
    task_description: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # pending, in_progress, completed, failed
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    collaborators: List[str]
    knowledge_used: List[str]  # Memory IDs used
    knowledge_gained: List[str]  # Memory IDs created
    performance_metrics: Dict[str, float]
    lessons_learned: List[str]
    
    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str) and self.end_time:
            self.end_time = datetime.fromisoformat(self.end_time)

class AdvancedAgent:
    """Advanced AI Agent with memory, learning, and collaboration capabilities"""
    
    def __init__(
        self, 
        profile: AgentProfile,
        communication_protocol: CommunicationProtocol,
        db_path: str = "./chroma_db"
    ):
        self.profile = profile
        self.state = AgentState.INITIALIZING
        
        # Core systems
        self.memory_system = AdvancedMemorySystem(profile.agent_id, db_path)
        self.communication = communication_protocol
        self.validator = KnowledgeValidator(profile.agent_id, profile.expertise_domains)
        self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
        
        # Task execution tracking
        self.task_history: Dict[str, TaskExecution] = {}
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # Learning and adaptation
        self.performance_metrics: Dict[str, List[float]] = {}
        self.collaboration_effectiveness: Dict[str, float] = {}  # agent_id -> effectiveness
        self.knowledge_application_success: Dict[str, int] = {}
        
        # Reusability features
        self.reusable_patterns: Dict[str, Dict[str, Any]] = {}
        self.template_solutions: Dict[str, Dict[str, Any]] = {}
        self.best_practices: List[Dict[str, Any]] = []
        
        self.state = AgentState.IDLE
        
        # Register message handlers
        self._register_message_handlers()
    
    def _register_message_handlers(self):
        """Register handlers for different message types"""
        handlers = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
            MessageType.KNOWLEDGE_SHARE: self._handle_knowledge_share,
            MessageType.VALIDATION_REQUEST: self._handle_validation_request,
            MessageType.EXPERT_CONSULTATION: self._handle_expert_consultation
        }
        
        for msg_type, handler in handlers.items():
            self.communication.register_message_handler(msg_type, handler)
    
    async def execute_task(
        self, 
        task_description: str, 
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None,
        collaboration_allowed: bool = True
    ) -> Dict[str, Any]:
        """Execute a task using accumulated knowledge and collaboration"""
        
        task_id = str(uuid.uuid4())
        self.state = AgentState.ACTIVE
        
        # Create task execution record
        task_execution = TaskExecution(
            task_id=task_id,
            agent_id=self.profile.agent_id,
            task_description=task_description,
            start_time=datetime.now(),
            end_time=None,
            status="in_progress",
            input_data=input_data,
            output_data={},
            collaborators=[],
            knowledge_used=[],
            knowledge_gained=[],
            performance_metrics={},
            lessons_learned=[]
        )
        
        self.active_tasks[task_id] = task_execution
        
        try:
            # Phase 1: Knowledge retrieval and preparation
            relevant_knowledge = await self._retrieve_relevant_knowledge(
                task_description, input_data, context or {}
            )
            
            # Phase 2: Check for reusable patterns
            reusable_solution = await self._find_reusable_solution(
                task_description, input_data
            )
            
            # Phase 3: Determine if collaboration is needed
            collaboration_needed = await self._assess_collaboration_need(
                task_description, input_data, relevant_knowledge
            )
            
            # Phase 4: Execute task (with or without collaboration)
            if collaboration_needed and collaboration_allowed:
                output = await self._execute_collaborative_task(
                    task_execution, relevant_knowledge, reusable_solution
                )
            else:
                output = await self._execute_individual_task(
                    task_execution, relevant_knowledge, reusable_solution
                )
            
            # Phase 5: Post-processing and learning
            await self._post_process_task(task_execution, output)
            
            task_execution.end_time = datetime.now()
            task_execution.status = "completed"
            task_execution.output_data = output
            
            # Move to history
            self.task_history[task_id] = task_execution
            del self.active_tasks[task_id]
            
            self.state = AgentState.IDLE
            self.profile.last_active = datetime.now()
            
            return output
            
        except Exception as e:
            task_execution.status = "failed"
            task_execution.output_data = {"error": str(e)}
            task_execution.end_time = datetime.now()
            
            self.task_history[task_id] = task_execution
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            self.state = AgentState.ERROR
            raise e
    
    async def _retrieve_relevant_knowledge(
        self, 
        task_description: str, 
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve relevant knowledge from memory system"""
        
        # Combine task description with context for search
        search_query = f"{task_description} {' '.join(str(v) for v in context.values())}"
        
        # Retrieve episodic memories
        episodic_memories = self.memory_system.retrieve_episodic_memories(
            search_query, limit=10, similarity_threshold=0.6
        )
        
        # Retrieve semantic memories
        semantic_memories = self.memory_system.retrieve_semantic_memories(
            search_query, limit=5, similarity_threshold=0.7
        )
        
        return {
            "episodic_memories": episodic_memories,
            "semantic_memories": semantic_memories,
            "search_query": search_query
        }
    
    async def _find_reusable_solution(
        self, 
        task_description: str, 
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find reusable patterns or template solutions"""
        
        # Check template solutions
        for template_id, template in self.template_solutions.items():
            similarity = await self._calculate_task_similarity(
                task_description, template["task_description"]
            )
            
            if similarity > 0.8:  # High similarity threshold
                return {
                    "type": "template",
                    "template_id": template_id,
                    "template": template,
                    "similarity": similarity
                }
        
        # Check reusable patterns
        for pattern_id, pattern in self.reusable_patterns.items():
            pattern_match = await self._check_pattern_match(
                task_description, input_data, pattern
            )
            
            if pattern_match["confidence"] > 0.7:
                return {
                    "type": "pattern",
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                    "match_confidence": pattern_match["confidence"],
                    "adaptations_needed": pattern_match["adaptations"]
                }
        
        return None
    
    async def _calculate_task_similarity(self, task1: str, task2: str) -> float:
        """Calculate similarity between two task descriptions"""
        
        # Use memory system's encoder for consistency
        similarity = self.memory_system.encoder.calculate_similarity(task1, task2)
        return similarity
    
    async def _check_pattern_match(
        self, 
        task_description: str, 
        input_data: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if current task matches a reusable pattern"""
        
        # Simplified pattern matching - in real implementation, use more sophisticated matching
        task_similarity = await self._calculate_task_similarity(
            task_description, pattern.get("task_pattern", "")
        )
        
        # Check input data structure similarity
        input_similarity = self._calculate_data_structure_similarity(
            input_data, pattern.get("input_pattern", {})
        )
        
        overall_confidence = (task_similarity + input_similarity) / 2
        
        adaptations = []
        if task_similarity < 0.9:
            adaptations.append("task_description_adaptation")
        if input_similarity < 0.9:
            adaptations.append("input_data_adaptation")
        
        return {
            "confidence": overall_confidence,
            "adaptations": adaptations,
            "task_similarity": task_similarity,
            "input_similarity": input_similarity
        }
    
    def _calculate_data_structure_similarity(
        self, 
        data1: Dict[str, Any], 
        data2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between data structures"""
        
        if not data1 and not data2:
            return 1.0
        if not data1 or not data2:
            return 0.0
        
        # Compare keys
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        if not keys1 and not keys2:
            return 1.0
        
        common_keys = keys1.intersection(keys2)
        all_keys = keys1.union(keys2)
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 0.0
        
        return key_similarity
    
    async def _assess_collaboration_need(
        self, 
        task_description: str, 
        input_data: Dict[str, Any],
        relevant_knowledge: Dict[str, Any]
    ) -> bool:
        """Assess whether collaboration is needed for this task"""
        
        # Check agent's collaboration preference
        if self.profile.collaboration_preference < 0.3:
            return False
        
        # Use AI to assess complexity and collaboration need
        system_prompt = f"""
        You are {self.profile.name} with expertise in: {', '.join(self.profile.expertise_domains)}.
        Assess whether collaboration is needed for this task.
        
        Consider:
        1. Task complexity
        2. Your expertise coverage
        3. Available knowledge
        4. Potential benefits of collaboration
        5. Time constraints
        
        Return true if collaboration would significantly improve outcomes.
        """
        
        user_prompt = f"""
        Task: {task_description}
        Input Data: {json.dumps(input_data, indent=2)}
        Available Knowledge: {len(relevant_knowledge.get('episodic_memories', []))} episodic memories, {len(relevant_knowledge.get('semantic_memories', []))} semantic memories
        
        Should I collaborate on this task? Return JSON with "collaborate": boolean and "reasoning": string.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("collaborate", False)
            
        except Exception as e:
            # Fallback to simple heuristic
            return len(relevant_knowledge.get('semantic_memories', [])) < 2
    
    async def _execute_collaborative_task(
        self, 
        task_execution: TaskExecution,
        relevant_knowledge: Dict[str, Any],
        reusable_solution: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute task with collaboration"""
        
        self.state = AgentState.COLLABORATING
        
        # Find suitable collaborators
        collaborators = await self._find_collaborators(
            task_execution.task_description, task_execution.input_data
        )
        
        if not collaborators:
            # Fallback to individual execution
            return await self._execute_individual_task(
                task_execution, relevant_knowledge, reusable_solution
            )
        
        task_execution.collaborators = collaborators
        
        # Start collaboration conversation
        conversation_id = await self.communication.start_conversation(
            participants=[self.profile.agent_id] + collaborators,
            topic=f"Collaborative task: {task_execution.task_description}",
            context={
                "task_id": task_execution.task_id,
                "task_description": task_execution.task_description,
                "input_data": task_execution.input_data,
                "reusable_solution": reusable_solution
            },
            agent_capabilities={
                agent_id: ["collaboration", "problem_solving"] 
                for agent_id in [self.profile.agent_id] + collaborators
            }
        )
        
        # Execute collaborative solution development
        collaborative_output = await self._develop_collaborative_solution(
            task_execution, conversation_id, relevant_knowledge, reusable_solution
        )
        
        # Record collaboration metrics
        await self._record_collaboration_metrics(
            task_execution.task_id, collaborators, collaborative_output
        )
        
        return collaborative_output
    
    async def _execute_individual_task(
        self, 
        task_execution: TaskExecution,
        relevant_knowledge: Dict[str, Any],
        reusable_solution: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute task individually using available knowledge and patterns"""
        
        # Prepare context from relevant knowledge
        knowledge_context = self._prepare_knowledge_context(relevant_knowledge)
        
        # Apply reusable solution if available
        if reusable_solution:
            return await self._apply_reusable_solution(
                task_execution, reusable_solution, knowledge_context
            )
        
        # Generate solution from scratch
        return await self._generate_original_solution(
            task_execution, knowledge_context
        )
    
    def _prepare_knowledge_context(self, relevant_knowledge: Dict[str, Any]) -> str:
        """Prepare knowledge context for task execution"""
        
        context_parts = []
        
        # Add episodic memories
        episodic_memories = relevant_knowledge.get("episodic_memories", [])
        if episodic_memories:
            context_parts.append("Relevant Past Experiences:")
            for memory in episodic_memories[:5]:  # Limit to top 5
                context_parts.append(f"- {memory.content}")
        
        # Add semantic memories
        semantic_memories = relevant_knowledge.get("semantic_memories", [])
        if semantic_memories:
            context_parts.append("Relevant Knowledge:")
            for memory in semantic_memories[:3]:  # Limit to top 3
                context_parts.append(f"- {memory.content}")
        
        return "\n".join(context_parts)
    
    async def _apply_reusable_solution(
        self, 
        task_execution: TaskExecution,
        reusable_solution: Dict[str, Any],
        knowledge_context: str
    ) -> Dict[str, Any]:
        """Apply and adapt a reusable solution"""
        
        system_prompt = f"""
        You are {self.profile.name} with expertise in: {', '.join(self.profile.expertise_domains)}.
        Apply and adapt the provided reusable solution to the current task.
        
        Capabilities: {', '.join([cap.value for cap in self.profile.capabilities])}
        """
        
        solution_context = ""
        if reusable_solution["type"] == "template":
            solution_context = f"Template Solution: {json.dumps(reusable_solution['template'], indent=2)}"
        elif reusable_solution["type"] == "pattern":
            solution_context = f"Pattern: {json.dumps(reusable_solution['pattern'], indent=2)}"
        
        user_prompt = f"""
        Task: {task_execution.task_description}
        Input: {json.dumps(task_execution.input_data, indent=2)}
        
        {solution_context}
        
        Knowledge Context:
        {knowledge_context}
        
        Adapt and apply the reusable solution to complete this task. Return the solution in JSON format.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            solution = json.loads(response.choices[0].message.content)
            solution["reused_solution"] = True
            solution["reuse_type"] = reusable_solution["type"]
            
            return solution
            
        except Exception as e:
            # Fallback to original solution generation
            return await self._generate_original_solution(task_execution, knowledge_context)
    
    async def _generate_original_solution(
        self, 
        task_execution: TaskExecution,
        knowledge_context: str
    ) -> Dict[str, Any]:
        """Generate original solution for the task"""
        
        system_prompt = f"""
        You are {self.profile.name}, an AI agent with the following profile:
        - Role: {self.profile.role}
        - Expertise: {', '.join(self.profile.expertise_domains)}
        - Capabilities: {', '.join([cap.value for cap in self.profile.capabilities])}
        - Experience Level: {self.profile.experience_level}
        
        Generate a comprehensive solution for the given task using your expertise and available knowledge.
        """
        
        user_prompt = f"""
        Task: {task_execution.task_description}
        Input Data: {json.dumps(task_execution.input_data, indent=2)}
        
        Available Knowledge:
        {knowledge_context}
        
        Provide a detailed solution in JSON format with the following structure:
        {{
            "solution_summary": "Brief summary of the solution",
            "detailed_solution": "Comprehensive solution details",
            "methodology": "Approach and methods used",
            "key_insights": ["insight1", "insight2"],
            "recommendations": ["rec1", "rec2"],
            "potential_issues": ["issue1", "issue2"],
            "success_metrics": ["metric1", "metric2"],
            "implementation_steps": ["step1", "step2"]
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                response_format={"type": "json_object"}
            )
            
            solution = json.loads(response.choices[0].message.content)
            solution["reused_solution"] = False
            solution["agent_id"] = self.profile.agent_id
            
            return solution
            
        except Exception as e:
            return {
                "solution_summary": f"Error generating solution: {str(e)}",
                "detailed_solution": "Unable to generate solution due to error",
                "error": str(e),
                "reused_solution": False,
                "agent_id": self.profile.agent_id
            }
    
    async def _find_collaborators(
        self, 
        task_description: str, 
        input_data: Dict[str, Any]
    ) -> List[str]:
        """Find suitable collaborators for the task"""
        
        from core.shared_resources import global_encoder
        
        # Analyze task description to find relevant expertise
        task_embedding = global_encoder.encode_content(task_description.lower())
        
        collaborators = []
        
        # Simulate available experts (in real implementation, this would query a registry)
        available_experts = {
            "business_strategy_expert": ["business_strategy", "market_analysis"],
            "marketing_expert": ["marketing_strategy", "customer_engagement"],
            "technical_architect": ["system_design", "technical_implementation"],
            "creative_designer": ["user_experience", "design_thinking"],
            "project_manager": ["project_management", "coordination"],
            "quality_assurance": ["quality_control", "validation"]
        }
        
        if len(available_experts) < 2:
            return []
        
        # Calculate relevance scores for each expert
        expert_scores = []
        
        for expert_id, expertise_domains in available_experts.items():
            if expert_id == self.profile.agent_id:
                continue  # Don't collaborate with self
            
            # Calculate expertise relevance
            expertise_text = " ".join(expertise_domains).lower()
            expertise_embedding = global_encoder.encode_content(expertise_text)
            
            import numpy as np
            relevance_score = np.dot(task_embedding, expertise_embedding) / (
                np.linalg.norm(task_embedding) * np.linalg.norm(expertise_embedding)
            )
            
            # Consider past collaboration effectiveness
            effectiveness_factor = self.collaboration_effectiveness.get(expert_id, 0.7)
            
            # Consider collaboration preference
            collab_factor = self.profile.collaboration_preference
            
            final_score = relevance_score * effectiveness_factor * collab_factor
            
            expert_scores.append((expert_id, final_score))
        
        # Sort by score and select top collaborators
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 2-3 collaborators based on score threshold
        min_score_threshold = 0.4
        max_collaborators = 3
        
        for expert_id, score in expert_scores[:max_collaborators]:
            if score >= min_score_threshold:
                collaborators.append(expert_id)
        
        # Ensure at least 1 collaborator if available and score is reasonable
        if len(collaborators) == 0 and expert_scores and expert_scores[0][1] > 0.2:
            collaborators.append(expert_scores[0][0])
        
        return collaborators[:3]  # Max 3 collaborators
    
    async def _develop_collaborative_solution(
        self, 
        task_execution: TaskExecution,
        conversation_id: str,
        relevant_knowledge: Dict[str, Any],
        reusable_solution: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Develop solution through collaboration"""
        
        # Get conversation results
        conversation = await self.communication.get_conversation(conversation_id)
        
        if not conversation:
            # Fallback to individual execution
            return await self._execute_individual_task(
                task_execution, relevant_knowledge, reusable_solution
            )
        
        # Synthesize collaborative insights
        collaborative_insights = []
        for message in conversation.messages:
            if message.sender_id != self.profile.agent_id:
                collaborative_insights.append({
                    "contributor": message.sender_id,
                    "insight": message.content
                })
        
        # Generate final solution incorporating collaboration
        system_prompt = f"""
        You are {self.profile.name} synthesizing insights from a collaborative discussion.
        Incorporate the collaborative insights to create a comprehensive solution.
        """
        
        insights_text = "\n".join([
            f"{insight['contributor']}: {insight['insight']}" 
            for insight in collaborative_insights[:5]
        ])
        
        user_prompt = f"""
        Task: {task_execution.task_description}
        Input: {json.dumps(task_execution.input_data, indent=2)}
        
        Collaborative Insights:
        {insights_text}
        
        Create a final solution that incorporates the collaborative insights. Return in JSON format.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            solution = json.loads(response.choices[0].message.content)
            solution["collaborative"] = True
            solution["collaborators"] = task_execution.collaborators
            solution["collaborative_insights"] = collaborative_insights
            
            return solution
            
        except Exception as e:
            # Fallback to individual solution
            return await self._execute_individual_task(
                task_execution, relevant_knowledge, reusable_solution
            )
    
    async def _post_process_task(self, task_execution: TaskExecution, output: Dict[str, Any]):
        """Post-process task execution for learning and improvement"""
        
        # Store episodic memory
        episodic_memory = EpisodicMemory(
            id=str(uuid.uuid4()),
            agent_id=self.profile.agent_id,
            timestamp=datetime.now(),
            event_type="task_completion",
            context={
                "task_id": task_execution.task_id,
                "collaborators": task_execution.collaborators,
                "success": task_execution.status == "completed"
            },
            participants=task_execution.collaborators,
            content=f"Completed task: {task_execution.task_description}. Output: {str(output)[:200]}",
            emotional_valence=0.8 if task_execution.status == "completed" else -0.3,
            importance_score=0.7,
            related_memories=[]
        )
        
        memory_id = self.memory_system.store_episodic_memory(episodic_memory)
        task_execution.knowledge_gained.append(memory_id)
        
        # Extract lessons learned
        lessons = await self._extract_lessons_learned(task_execution, output)
        task_execution.lessons_learned = lessons
        
        # Update reusable patterns if successful
        if task_execution.status == "completed" and self._is_pattern_worthy(output):
            await self._create_reusable_pattern(task_execution, output)
        
        # Update performance metrics
        await self._update_performance_metrics(task_execution, output)
    
    async def _extract_lessons_learned(
        self, 
        task_execution: TaskExecution, 
        output: Dict[str, Any]
    ) -> List[str]:
        """Extract lessons learned from task execution"""
        
        system_prompt = f"""
        As {self.profile.name}, reflect on the task execution and extract key lessons learned.
        Focus on insights that could improve future performance.
        """
        
        user_prompt = f"""
        Task: {task_execution.task_description}
        Status: {task_execution.status}
        Collaborators: {', '.join(task_execution.collaborators) if task_execution.collaborators else 'None'}
        Duration: {(task_execution.end_time - task_execution.start_time).total_seconds() if task_execution.end_time else 'Unknown'} seconds
        Output Quality: {'High' if task_execution.status == 'completed' else 'Low'}
        
        Extract 3-5 key lessons learned from this task execution.
        Return as JSON array of strings.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("lessons", [])
            
        except Exception as e:
            return [f"Task execution experience: {task_execution.status}"]
    
    def _is_pattern_worthy(self, output: Dict[str, Any]) -> bool:
        """Determine if output is worthy of creating a reusable pattern"""
        
        # Simple heuristics - in real implementation, use more sophisticated criteria
        if output.get("reused_solution"):
            return False  # Don't create patterns from reused solutions
        
        if "error" in output:
            return False  # Don't create patterns from failed executions
        
        # Check if output has comprehensive structure
        required_fields = ["solution_summary", "detailed_solution", "methodology"]
        return all(field in output for field in required_fields)
    
    async def _create_reusable_pattern(self, task_execution: TaskExecution, output: Dict[str, Any]):
        """Create a reusable pattern from successful task execution"""
        
        pattern_id = str(uuid.uuid4())
        
        pattern = {
            "pattern_id": pattern_id,
            "task_pattern": task_execution.task_description,
            "input_pattern": {key: type(value).__name__ for key, value in task_execution.input_data.items()},
            "solution_template": {
                key: value for key, value in output.items()
                if key not in ["agent_id", "reused_solution"]
            },
            "success_metrics": output.get("success_metrics", []),
            "methodology": output.get("methodology", ""),
            "created_from_task": task_execution.task_id,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "success_rate": 1.0  # Start with 100% since it's created from successful execution
        }
        
        self.reusable_patterns[pattern_id] = pattern
        
        # Store as semantic memory
        semantic_memory = SemanticMemory(
            id=str(uuid.uuid4()),
            agent_id=self.profile.agent_id,
            concept=f"Reusable pattern: {task_execution.task_description[:50]}",
            knowledge_type="pattern",
            content=f"Pattern for {task_execution.task_description}: {output.get('methodology', '')}",
            confidence_level=0.9,
            source_episodes=[task_execution.task_id],
            validation_count=1,
            last_updated=datetime.now(),
            expertise_domain=self.profile.expertise_domains[0] if self.profile.expertise_domains else "general"
        )
        
        self.memory_system.store_semantic_memory(semantic_memory)
    
    async def _update_performance_metrics(self, task_execution: TaskExecution, output: Dict[str, Any]):
        """Update performance metrics based on task execution"""
        
        # Calculate execution time
        if task_execution.end_time:
            execution_time = (task_execution.end_time - task_execution.start_time).total_seconds()
            
            if "execution_time" not in self.performance_metrics:
                self.performance_metrics["execution_time"] = []
            self.performance_metrics["execution_time"].append(execution_time)
        
        # Calculate success rate
        success = 1.0 if task_execution.status == "completed" else 0.0
        if "success_rate" not in self.performance_metrics:
            self.performance_metrics["success_rate"] = []
        self.performance_metrics["success_rate"].append(success)
        
        # Update collaboration effectiveness
        if task_execution.collaborators:
            collaboration_success = 1.0 if task_execution.status == "completed" else 0.0
            for collaborator in task_execution.collaborators:
                if collaborator not in self.collaboration_effectiveness:
                    self.collaboration_effectiveness[collaborator] = 0.5
                
                # Update using exponential moving average
                alpha = 0.3
                self.collaboration_effectiveness[collaborator] = (
                    alpha * collaboration_success + 
                    (1 - alpha) * self.collaboration_effectiveness[collaborator]
                )
    
    async def _record_collaboration_metrics(
        self, 
        task_id: str, 
        collaborators: List[str], 
        output: Dict[str, Any]
    ):
        """Record metrics about collaboration effectiveness"""
        
        collaboration_record = {
            "task_id": task_id,
            "collaborators": collaborators,
            "timestamp": datetime.now().isoformat(),
            "success": "error" not in output,
            "output_quality": self._assess_output_quality(output),
            "collaboration_value": self._assess_collaboration_value(output)
        }
        
        self.collaboration_history.append(collaboration_record)
    
    def _assess_output_quality(self, output: Dict[str, Any]) -> float:
        """Assess quality of output (0.0 to 1.0)"""
        
        if "error" in output:
            return 0.1
        
        quality_score = 0.5  # Base score
        
        # Check completeness
        if "detailed_solution" in output and len(output["detailed_solution"]) > 100:
            quality_score += 0.2
        
        if "recommendations" in output and output["recommendations"]:
            quality_score += 0.1
        
        if "implementation_steps" in output and output["implementation_steps"]:
            quality_score += 0.1
        
        if "key_insights" in output and output["key_insights"]:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _assess_collaboration_value(self, output: Dict[str, Any]) -> float:
        """Assess value added by collaboration"""
        
        if not output.get("collaborative", False):
            return 0.0
        
        value_score = 0.3  # Base value for collaboration
        
        # Check for collaborative insights
        if "collaborative_insights" in output and output["collaborative_insights"]:
            value_score += 0.3
        
        # Check for diverse perspectives
        if len(output.get("collaborators", [])) > 1:
            value_score += 0.2
        
        # Check output comprehensiveness
        if self._assess_output_quality(output) > 0.8:
            value_score += 0.2
        
        return min(1.0, value_score)
    
    # Message Handlers
    async def _handle_task_request(self, message: Message):
        """Handle incoming task request"""
        
        task_data = message.context.get("task_data", {})
        
        # Execute the requested task
        try:
            result = await self.execute_task(
                task_description=message.content,
                input_data=task_data,
                context=message.context
            )
            
            # Send response
            response_message = Message(
                id=str(uuid.uuid4()),
                sender_id=self.profile.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content=json.dumps(result),
                context={"task_result": result, "original_task_id": message.id},
                timestamp=datetime.now(),
                priority=message.priority,
                requires_response=False,
                parent_message_id=message.id
            )
            
            await self.communication.send_message(response_message)
            
        except Exception as e:
            # Send error response
            error_response = Message(
                id=str(uuid.uuid4()),
                sender_id=self.profile.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content=f"Task execution failed: {str(e)}",
                context={"error": str(e), "original_task_id": message.id},
                timestamp=datetime.now(),
                priority=message.priority,
                requires_response=False,
                parent_message_id=message.id
            )
            
            await self.communication.send_message(error_response)
    
    async def _handle_collaboration_request(self, message: Message):
        """Handle collaboration request from another agent"""
        
        # Assess collaboration willingness
        willingness = await self._assess_collaboration_willingness(message)
        
        response_content = {
            "willing_to_collaborate": willingness > 0.5,
            "willingness_score": willingness,
            "available_expertise": self.profile.expertise_domains,
            "current_load": len(self.active_tasks)
        }
        
        response = Message(
            id=str(uuid.uuid4()),
            sender_id=self.profile.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.COLLABORATION_RESPONSE,
            content=json.dumps(response_content),
            context={"collaboration_assessment": response_content},
            timestamp=datetime.now(),
            priority=message.priority,
            requires_response=False,
            parent_message_id=message.id
        )
        
        await self.communication.send_message(response)
    
    async def _assess_collaboration_willingness(self, message: Message) -> float:
        """Assess willingness to collaborate based on various factors"""
        
        base_willingness = self.profile.collaboration_preference
        
        # Adjust based on current workload
        workload_factor = max(0.1, 1.0 - (len(self.active_tasks) * 0.2))
        
        # Adjust based on past collaboration effectiveness with sender
        sender_effectiveness = self.collaboration_effectiveness.get(message.sender_id, 0.5)
        
        # Adjust based on expertise relevance
        expertise_relevance = 0.5  # Placeholder - would calculate based on task content
        
        willingness = base_willingness * workload_factor * sender_effectiveness * expertise_relevance
        
        return min(1.0, willingness)
    
    async def _handle_knowledge_share(self, message: Message):
        """Handle knowledge sharing from another agent"""
        
        # Store shared knowledge as episodic memory
        episodic_memory = EpisodicMemory(
            id=str(uuid.uuid4()),
            agent_id=self.profile.agent_id,
            timestamp=datetime.now(),
            event_type="knowledge_share",
            context={"shared_by": message.sender_id},
            participants=[message.sender_id],
            content=f"Knowledge shared by {message.sender_id}: {message.content}",
            emotional_valence=0.5,
            importance_score=0.6,
            related_memories=[]
        )
        
        self.memory_system.store_episodic_memory(episodic_memory)
        
        # Send acknowledgment
        ack_message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.profile.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.COLLABORATION_RESPONSE,
            content="Knowledge received and stored",
            context={"knowledge_stored": True},
            timestamp=datetime.now(),
            priority=1,
            requires_response=False,
            parent_message_id=message.id
        )
        
        await self.communication.send_message(ack_message)
    
    async def _handle_validation_request(self, message: Message):
        """Handle validation request from another agent"""
        
        validation_data = message.context.get("validation_request", {})
        
        # Create validation request
        request = ValidationRequest(
            id=str(uuid.uuid4()),
            requester_id=message.sender_id,
            content=message.content,
            validation_types=[ValidationType.EXPERTISE_ALIGNMENT, ValidationType.FACTUAL_ACCURACY],
            context=message.context,
            priority=message.priority,
            deadline=None,
            required_validators=[self.profile.agent_id],
            minimum_validators=1,
            created_at=datetime.now()
        )
        
        # Perform validation
        validation_response = await self.validator.validate_knowledge(
            request, ValidationType.EXPERTISE_ALIGNMENT
        )
        
        # Send validation response
        response = Message(
            id=str(uuid.uuid4()),
            sender_id=self.profile.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.VALIDATION_RESPONSE,
            content=json.dumps(asdict(validation_response)),
            context={"validation_result": asdict(validation_response)},
            timestamp=datetime.now(),
            priority=message.priority,
            requires_response=False,
            parent_message_id=message.id
        )
        
        await self.communication.send_message(response)
    
    async def _handle_expert_consultation(self, message: Message):
        """Handle expert consultation request"""
        
        # Check if this agent has relevant expertise
        expertise_relevance = self.validator._calculate_expertise_relevance(
            message.content, message.context
        )
        
        if expertise_relevance < 0.5:
            # Not qualified to provide expert consultation
            response_content = {
                "can_provide_consultation": False,
                "expertise_relevance": expertise_relevance,
                "reason": "Insufficient expertise in requested domain"
            }
        else:
            # Provide expert consultation
            consultation = await self._provide_expert_consultation(message)
            response_content = {
                "can_provide_consultation": True,
                "expertise_relevance": expertise_relevance,
                "consultation": consultation
            }
        
        response = Message(
            id=str(uuid.uuid4()),
            sender_id=self.profile.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.EXPERT_CONSULTATION,
            content=json.dumps(response_content),
            context={"consultation_result": response_content},
            timestamp=datetime.now(),
            priority=message.priority,
            requires_response=False,
            parent_message_id=message.id
        )
        
        await self.communication.send_message(response)
    
    async def _provide_expert_consultation(self, message: Message) -> Dict[str, Any]:
        """Provide expert consultation based on agent's expertise"""
        
        system_prompt = f"""
        You are {self.profile.name}, an expert consultant with the following qualifications:
        - Expertise: {', '.join(self.profile.expertise_domains)}
        - Experience Level: {self.profile.experience_level}
        - Capabilities: {', '.join([cap.value for cap in self.profile.capabilities])}
        
        Provide expert consultation on the requested topic.
        """
        
        user_prompt = f"""
        Consultation Request: {message.content}
        Context: {json.dumps(message.context, indent=2)}
        
        Provide expert consultation including:
        1. Analysis of the situation
        2. Expert recommendations
        3. Potential risks and considerations
        4. Best practices
        5. Implementation guidance
        
        Return as JSON format.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            consultation = json.loads(response.choices[0].message.content)
            consultation["consultant"] = self.profile.agent_id
            consultation["expertise_domains"] = self.profile.expertise_domains
            
            return consultation
            
        except Exception as e:
            return {
                "error": str(e),
                "consultant": self.profile.agent_id,
                "status": "consultation_failed"
            }
    
    # Utility methods
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        
        total_tasks = len(self.task_history)
        successful_tasks = len([t for t in self.task_history.values() if t.status == "completed"])
        
        avg_execution_time = 0.0
        if "execution_time" in self.performance_metrics:
            times = self.performance_metrics["execution_time"]
            avg_execution_time = sum(times) / len(times) if times else 0.0
        
        success_rate = 0.0
        if "success_rate" in self.performance_metrics:
            rates = self.performance_metrics["success_rate"]
            success_rate = sum(rates) / len(rates) if rates else 0.0
        
        return {
            "agent_id": self.profile.agent_id,
            "agent_name": self.profile.name,
            "current_state": self.state.value,
            "total_tasks_completed": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_execution_time_seconds": avg_execution_time,
            "active_tasks": len(self.active_tasks),
            "reusable_patterns_created": len(self.reusable_patterns),
            "template_solutions": len(self.template_solutions),
            "collaboration_partners": len(self.collaboration_effectiveness),
            "memory_statistics": self.memory_system.get_memory_statistics(),
            "expertise_domains": self.profile.expertise_domains,
            "experience_level": self.profile.experience_level,
            "collaboration_preference": self.profile.collaboration_preference
        }
    
    def save_agent_state(self, filepath: str):
        """Save agent state to file for persistence"""
        
        state_data = {
            "profile": asdict(self.profile),
            "task_history": {k: asdict(v) for k, v in self.task_history.items()},
            "performance_metrics": self.performance_metrics,
            "collaboration_effectiveness": self.collaboration_effectiveness,
            "reusable_patterns": self.reusable_patterns,
            "template_solutions": self.template_solutions,
            "collaboration_history": self.collaboration_history,
            "best_practices": self.best_practices,
            "memory_export": self.memory_system.export_memories()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
    
    def load_agent_state(self, filepath: str):
        """Load agent state from file"""
        
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        # Restore profile
        self.profile = AgentProfile(**state_data["profile"])
        
        # Restore task history
        self.task_history = {
            k: TaskExecution(**v) for k, v in state_data.get("task_history", {}).items()
        }
        
        # Restore other state
        self.performance_metrics = state_data.get("performance_metrics", {})
        self.collaboration_effectiveness = state_data.get("collaboration_effectiveness", {})
        self.reusable_patterns = state_data.get("reusable_patterns", {})
        self.template_solutions = state_data.get("template_solutions", {})
        self.collaboration_history = state_data.get("collaboration_history", [])
        self.best_practices = state_data.get("best_practices", [])
        
        # Restore memory system
        if "memory_export" in state_data:
            self.memory_system.import_memories(state_data["memory_export"])
