"""
Advanced Task Orchestration and Collaborative Planning System
Implements the central orchestration system as described in NTT's Multi-Agent AI Technology
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
from config import CONFIG, TASK_COMPLEXITY
from core.agent import AdvancedAgent, AgentProfile, AgentCapability, TaskExecution
from core.expert_agents import ExpertAgentRegistry, ExpertDomain
from core.communication import CommunicationProtocol, Message, MessageType
from core.meetings import MeetingOrchestrator, MeetingType
from core.validation import CrossValidationOrchestrator, ValidationRequest, ValidationType
import networkx as nx

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    CRITICAL = 7
    EMERGENCY = 10

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CollaborationType(Enum):
    """Types of collaboration patterns"""
    SEQUENTIAL = "sequential"  # One agent after another
    PARALLEL = "parallel"     # Multiple agents simultaneously
    HIERARCHICAL = "hierarchical"  # Lead agent with sub-agents
    PEER_TO_PEER = "peer_to_peer"  # Equal collaboration
    EXPERT_CONSULTATION = "expert_consultation"  # Consulting experts
    CONSENSUS_BUILDING = "consensus_building"  # Building agreement

@dataclass
class TaskDefinition:
    """Comprehensive task definition"""
    id: str
    title: str
    description: str
    objectives: List[str]
    deliverables: List[str]
    constraints: Dict[str, Any]
    priority: TaskPriority
    complexity: str  # simple, moderate, complex, enterprise
    estimated_duration: timedelta
    deadline: Optional[datetime]
    required_expertise: List[ExpertDomain]
    required_capabilities: List[AgentCapability]
    success_criteria: List[str]
    context: Dict[str, Any]
    dependencies: List[str]  # Other task IDs
    created_at: datetime
    created_by: str
    
    def __post_init__(self):
        if isinstance(self.estimated_duration, str):
            # Parse duration string if needed
            pass
        if isinstance(self.deadline, str) and self.deadline:
            self.deadline = datetime.fromisoformat(self.deadline)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

@dataclass
class CollaborationPlan:
    """Plan for multi-agent collaboration"""
    id: str
    task_id: str
    collaboration_type: CollaborationType
    participating_agents: List[str]
    agent_roles: Dict[str, str]  # agent_id -> role in collaboration
    workflow_steps: List[Dict[str, Any]]
    communication_protocol: Dict[str, Any]
    coordination_meetings: List[str]  # Meeting IDs
    validation_checkpoints: List[Dict[str, Any]]
    integration_strategy: Dict[str, Any]
    estimated_timeline: Dict[str, datetime]
    risk_mitigation: List[Dict[str, Any]]
    created_at: datetime
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

@dataclass
class TaskExecution:
    """Extended task execution with orchestration details"""
    id: str
    task_definition: TaskDefinition
    collaboration_plan: Optional[CollaborationPlan]
    status: TaskStatus
    assigned_agents: List[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    progress_percentage: float
    current_phase: str
    intermediate_outputs: Dict[str, Any]
    validation_results: List[Dict[str, Any]]
    meetings_conducted: List[str]
    issues_encountered: List[Dict[str, Any]]
    lessons_learned: List[str]
    final_output: Optional[Dict[str, Any]]
    quality_score: Optional[float]
    stakeholder_feedback: List[Dict[str, Any]]
    
    def __post_init__(self):
        if isinstance(self.start_time, str) and self.start_time:
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str) and self.end_time:
            self.end_time = datetime.fromisoformat(self.end_time)

class TaskOrchestrator:
    """Central orchestrator for complex multi-agent tasks"""
    
    def __init__(
        self, 
        expert_registry: ExpertAgentRegistry,
        communication_protocol: CommunicationProtocol,
        meeting_orchestrator: MeetingOrchestrator,
        validation_orchestrator: CrossValidationOrchestrator
    ):
        self.expert_registry = expert_registry
        self.communication = communication_protocol
        self.meeting_orchestrator = meeting_orchestrator
        self.validation_orchestrator = validation_orchestrator
        self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
        
        # Task management
        self.pending_tasks: Dict[str, TaskDefinition] = {}
        self.active_executions: Dict[str, TaskExecution] = {}
        self.completed_tasks: Dict[str, TaskExecution] = {}
        self.task_dependencies = nx.DiGraph()
        
        # Collaboration patterns
        self.collaboration_templates: Dict[str, Dict[str, Any]] = {}
        self.successful_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.execution_metrics: Dict[str, List[float]] = {}
        self.collaboration_effectiveness: Dict[str, float] = {}
        
        self._initialize_collaboration_templates()
    
    def _initialize_collaboration_templates(self):
        """Initialize collaboration pattern templates"""
        
        self.collaboration_templates = {
            "strategic_planning": {
                "collaboration_type": CollaborationType.HIERARCHICAL,
                "lead_expertise": [ExpertDomain.BUSINESS_STRATEGY],
                "supporting_expertise": [ExpertDomain.MARKETING, ExpertDomain.TECHNICAL_ARCHITECTURE],
                "workflow_pattern": "analysis -> strategy -> validation -> integration",
                "meeting_cadence": "kickoff -> midpoint -> review -> final",
                "validation_points": ["strategic_analysis", "recommendations", "implementation_plan"]
            },
            "product_development": {
                "collaboration_type": CollaborationType.PEER_TO_PEER,
                "required_expertise": [
                    ExpertDomain.CREATIVE_DESIGN, 
                    ExpertDomain.TECHNICAL_ARCHITECTURE,
                    ExpertDomain.MARKETING,
                    ExpertDomain.PROJECT_MANAGEMENT
                ],
                "workflow_pattern": "ideation -> design -> technical -> marketing -> integration",
                "meeting_cadence": "daily_standups -> weekly_reviews -> milestone_reviews",
                "validation_points": ["concept_validation", "technical_feasibility", "market_validation"]
            },
            "crisis_response": {
                "collaboration_type": CollaborationType.PARALLEL,
                "required_expertise": [
                    ExpertDomain.RISK_MANAGEMENT,
                    ExpertDomain.OPERATIONS,
                    ExpertDomain.CUSTOMER_EXPERIENCE
                ],
                "workflow_pattern": "assessment -> parallel_response -> coordination -> resolution",
                "meeting_cadence": "emergency_briefing -> hourly_updates -> resolution_review",
                "validation_points": ["impact_assessment", "response_effectiveness", "resolution_quality"]
            }
        }
    
    async def orchestrate_task(
        self, 
        task_definition: TaskDefinition,
        collaboration_preferences: Dict[str, Any] = None
    ) -> str:
        """Orchestrate execution of a complex task"""
        
        # Store task definition
        self.pending_tasks[task_definition.id] = task_definition
        
        # Analyze task requirements
        task_analysis = await self._analyze_task_requirements(task_definition)
        
        # Design collaboration plan
        collaboration_plan = await self._design_collaboration_plan(
            task_definition, task_analysis, collaboration_preferences or {}
        )
        
        # Select and assign agents
        assigned_agents = await self._select_and_assign_agents(
            task_definition, collaboration_plan
        )
        
        # Create task execution
        task_execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_definition=task_definition,
            collaboration_plan=collaboration_plan,
            status=TaskStatus.PLANNING,
            assigned_agents=assigned_agents,
            start_time=None,
            end_time=None,
            progress_percentage=0.0,
            current_phase="planning",
            intermediate_outputs={},
            validation_results=[],
            meetings_conducted=[],
            issues_encountered=[],
            lessons_learned=[],
            final_output=None,
            quality_score=None,
            stakeholder_feedback=[]
        )
        
        # Move to active executions
        self.active_executions[task_execution.id] = task_execution
        del self.pending_tasks[task_definition.id]
        
        # Start execution asynchronously
        asyncio.create_task(self._execute_task(task_execution))
        
        return task_execution.id
    
    async def _analyze_task_requirements(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Analyze task requirements and complexity"""
        
        system_prompt = """
        You are an expert task analyst for multi-agent AI systems.
        Analyze the task requirements and provide detailed assessment.
        
        Consider:
        1. Task complexity and scope
        2. Required expertise and capabilities
        3. Collaboration requirements
        4. Risk factors and challenges
        5. Success factors and dependencies
        """
        
        user_prompt = f"""
        Task: {task_definition.title}
        Description: {task_definition.description}
        Objectives: {', '.join(task_definition.objectives)}
        Deliverables: {', '.join(task_definition.deliverables)}
        Constraints: {json.dumps(task_definition.constraints, indent=2)}
        Priority: {task_definition.priority.name}
        Complexity: {task_definition.complexity}
        Required Expertise: {', '.join([e.value for e in task_definition.required_expertise])}
        
        Provide comprehensive task analysis in JSON format:
        {{
            "complexity_assessment": "...",
            "collaboration_requirements": "...",
            "risk_factors": [...],
            "success_factors": [...],
            "resource_requirements": "...",
            "timeline_assessment": "...",
            "quality_requirements": "..."
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
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "complexity_assessment": f"Analysis error: {str(e)}",
                "collaboration_requirements": "Unknown",
                "risk_factors": ["Analysis failure"],
                "success_factors": ["Proper execution"],
                "resource_requirements": "Unknown",
                "timeline_assessment": "Unknown",
                "quality_requirements": "Standard"
            }
    
    async def _design_collaboration_plan(
        self, 
        task_definition: TaskDefinition,
        task_analysis: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> CollaborationPlan:
        """Design optimal collaboration plan for the task"""
        
        # Select collaboration pattern
        collaboration_type = await self._select_collaboration_pattern(
            task_definition, task_analysis, preferences
        )
        
        # Design workflow steps
        workflow_steps = await self._design_workflow_steps(
            task_definition, collaboration_type, task_analysis
        )
        
        # Plan coordination meetings
        meeting_schedule = await self._plan_coordination_meetings(
            task_definition, collaboration_type, workflow_steps
        )
        
        # Define validation checkpoints
        validation_checkpoints = self._define_validation_checkpoints(
            task_definition, workflow_steps
        )
        
        # Create integration strategy
        integration_strategy = await self._create_integration_strategy(
            task_definition, collaboration_type, workflow_steps
        )
        
        collaboration_plan = CollaborationPlan(
            id=str(uuid.uuid4()),
            task_id=task_definition.id,
            collaboration_type=collaboration_type,
            participating_agents=[],  # Will be filled during agent selection
            agent_roles={},
            workflow_steps=workflow_steps,
            communication_protocol={
                "update_frequency": "daily",
                "escalation_rules": ["blocked_for_24h", "quality_issues", "resource_conflicts"],
                "documentation_requirements": ["progress_reports", "decision_logs", "issue_tracking"]
            },
            coordination_meetings=meeting_schedule,
            validation_checkpoints=validation_checkpoints,
            integration_strategy=integration_strategy,
            estimated_timeline=self._estimate_timeline(workflow_steps, task_definition),
            risk_mitigation=self._identify_risk_mitigation(task_analysis),
            created_at=datetime.now()
        )
        
        return collaboration_plan
    
    async def _select_collaboration_pattern(
        self, 
        task_definition: TaskDefinition,
        task_analysis: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> CollaborationType:
        """Select optimal collaboration pattern"""
        
        # Check for template matches
        for template_name, template in self.collaboration_templates.items():
            if self._matches_template(task_definition, template):
                return template["collaboration_type"]
        
        # Analyze task characteristics to determine pattern
        complexity = task_definition.complexity
        expertise_count = len(task_definition.required_expertise)
        has_dependencies = len(task_definition.dependencies) > 0
        
        # Decision logic
        if complexity == "enterprise" and expertise_count > 4:
            return CollaborationType.HIERARCHICAL
        elif expertise_count > 3 and not has_dependencies:
            return CollaborationType.PARALLEL
        elif task_definition.priority == TaskPriority.CRITICAL:
            return CollaborationType.EXPERT_CONSULTATION
        elif "consensus" in task_definition.description.lower():
            return CollaborationType.CONSENSUS_BUILDING
        else:
            return CollaborationType.PEER_TO_PEER
    
    def _matches_template(self, task_definition: TaskDefinition, template: Dict[str, Any]) -> bool:
        """Check if task matches a collaboration template"""
        
        # Simple keyword matching - in real implementation, use more sophisticated matching
        task_text = f"{task_definition.title} {task_definition.description}".lower()
        
        template_keywords = {
            "strategic_planning": ["strategy", "planning", "business", "strategic"],
            "product_development": ["product", "development", "design", "feature"],
            "crisis_response": ["crisis", "emergency", "urgent", "critical"]
        }
        
        for template_name, keywords in template_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                return True
        
        return False
    
    async def _design_workflow_steps(
        self, 
        task_definition: TaskDefinition,
        collaboration_type: CollaborationType,
        task_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Design detailed workflow steps"""
        
        system_prompt = f"""
        Design workflow steps for a {collaboration_type.value} collaboration pattern.
        Create detailed, actionable steps that agents can execute.
        
        Consider:
        1. Logical sequence of activities
        2. Dependencies between steps
        3. Collaboration touchpoints
        4. Quality checkpoints
        5. Integration opportunities
        """
        
        user_prompt = f"""
        Task: {task_definition.title}
        Collaboration Type: {collaboration_type.value}
        Objectives: {', '.join(task_definition.objectives)}
        Deliverables: {', '.join(task_definition.deliverables)}
        Task Analysis: {json.dumps(task_analysis, indent=2)}
        
        Design workflow steps in JSON format:
        [
            {{
                "step_id": "step_1",
                "name": "Step Name",
                "description": "Detailed description",
                "assigned_roles": ["role1", "role2"],
                "inputs": ["input1", "input2"],
                "outputs": ["output1", "output2"],
                "duration_hours": 8,
                "dependencies": ["previous_step_id"],
                "collaboration_required": true,
                "validation_required": true
            }}
        ]
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
            return result.get("workflow_steps", [])
            
        except Exception as e:
            # Fallback to basic workflow
            return [
                {
                    "step_id": "analysis",
                    "name": "Requirements Analysis",
                    "description": "Analyze task requirements and constraints",
                    "assigned_roles": ["analyst"],
                    "inputs": ["task_definition"],
                    "outputs": ["analysis_report"],
                    "duration_hours": 4,
                    "dependencies": [],
                    "collaboration_required": False,
                    "validation_required": True
                },
                {
                    "step_id": "solution_design",
                    "name": "Solution Design",
                    "description": "Design comprehensive solution approach",
                    "assigned_roles": ["expert"],
                    "inputs": ["analysis_report"],
                    "outputs": ["solution_design"],
                    "duration_hours": 8,
                    "dependencies": ["analysis"],
                    "collaboration_required": True,
                    "validation_required": True
                },
                {
                    "step_id": "implementation",
                    "name": "Solution Implementation",
                    "description": "Implement the designed solution",
                    "assigned_roles": ["implementer"],
                    "inputs": ["solution_design"],
                    "outputs": ["implementation_result"],
                    "duration_hours": 12,
                    "dependencies": ["solution_design"],
                    "collaboration_required": False,
                    "validation_required": True
                }
            ]
    
    async def _select_and_assign_agents(
        self, 
        task_definition: TaskDefinition,
        collaboration_plan: CollaborationPlan
    ) -> List[str]:
        """Select and assign optimal agents for the task"""
        
        # Get expert team recommendation
        expert_team = self.expert_registry.recommend_expert_team(
            task_requirements=asdict(task_definition),
            required_domains=task_definition.required_expertise,
            team_size_limit=TASK_COMPLEXITY[task_definition.complexity]["max_agents"]
        )
        
        # Assign roles based on collaboration plan
        agent_assignments = []
        agent_roles = {}
        
        for i, agent in enumerate(expert_team):
            agent_assignments.append(agent.profile.agent_id)
            
            # Assign role based on expertise and workflow steps
            role = self._determine_agent_role(agent, collaboration_plan.workflow_steps)
            agent_roles[agent.profile.agent_id] = role
        
        # Update collaboration plan with agent assignments
        collaboration_plan.participating_agents = agent_assignments
        collaboration_plan.agent_roles = agent_roles
        
        return agent_assignments
    
    def _determine_agent_role(
        self, 
        agent: AdvancedAgent, 
        workflow_steps: List[Dict[str, Any]]
    ) -> str:
        """Determine agent's role in the collaboration"""
        
        # Map agent expertise to workflow roles
        expertise_to_role = {
            "business_strategy": "strategic_lead",
            "marketing": "marketing_specialist",
            "technical_architecture": "technical_lead",
            "creative_design": "creative_lead",
            "project_management": "project_coordinator",
            "quality_assurance": "quality_validator"
        }
        
        # Find best role match
        for domain in agent.profile.expertise_domains:
            for expertise, role in expertise_to_role.items():
                if expertise in domain:
                    return role
        
        return "contributor"  # Default role
    
    async def _execute_task(self, task_execution: TaskExecution):
        """Execute the orchestrated task"""
        
        try:
            task_execution.status = TaskStatus.IN_PROGRESS
            task_execution.start_time = datetime.now()
            
            # Execute workflow steps
            for step in task_execution.collaboration_plan.workflow_steps:
                await self._execute_workflow_step(task_execution, step)
                
                # Update progress
                completed_steps = len([s for s in task_execution.collaboration_plan.workflow_steps 
                                     if s.get("completed", False)])
                total_steps = len(task_execution.collaboration_plan.workflow_steps)
                task_execution.progress_percentage = (completed_steps / total_steps) * 100
            
            # Final integration and validation
            await self._perform_final_integration(task_execution)
            
            # Complete task
            task_execution.status = TaskStatus.COMPLETED
            task_execution.end_time = datetime.now()
            task_execution.progress_percentage = 100.0
            
            # Move to completed tasks
            self.completed_tasks[task_execution.id] = task_execution
            del self.active_executions[task_execution.id]
            
            # Record success metrics
            await self._record_execution_metrics(task_execution)
            
        except Exception as e:
            task_execution.status = TaskStatus.FAILED
            task_execution.issues_encountered.append({
                "type": "execution_error",
                "description": str(e),
                "timestamp": datetime.now().isoformat(),
                "severity": "critical"
            })
            
            # Move to completed tasks (as failed)
            self.completed_tasks[task_execution.id] = task_execution
            if task_execution.id in self.active_executions:
                del self.active_executions[task_execution.id]
    
    async def _execute_workflow_step(
        self, 
        task_execution: TaskExecution, 
        step: Dict[str, Any]
    ):
        """Execute a single workflow step"""
        
        step_id = step["step_id"]
        task_execution.current_phase = step["name"]
        
        # Check dependencies
        for dep_step_id in step.get("dependencies", []):
            if not self._is_step_completed(task_execution, dep_step_id):
                raise Exception(f"Dependency {dep_step_id} not completed for step {step_id}")
        
        # Get assigned agents for this step
        assigned_agents = self._get_step_agents(task_execution, step)
        
        if not assigned_agents:
            raise Exception(f"No agents assigned for step {step_id}")
        
        # Execute step based on collaboration requirements
        if step.get("collaboration_required", False):
            step_output = await self._execute_collaborative_step(
                task_execution, step, assigned_agents
            )
        else:
            step_output = await self._execute_individual_step(
                task_execution, step, assigned_agents[0]
            )
        
        # Store intermediate output
        task_execution.intermediate_outputs[step_id] = step_output
        
        # Perform validation if required
        if step.get("validation_required", False):
            validation_result = await self._validate_step_output(
                task_execution, step, step_output
            )
            task_execution.validation_results.append(validation_result)
            
            if validation_result["result"] == "rejected":
                raise Exception(f"Step {step_id} validation failed: {validation_result['reasoning']}")
        
        # Mark step as completed
        step["completed"] = True
        step["completion_time"] = datetime.now().isoformat()
    
    def _get_step_agents(
        self, 
        task_execution: TaskExecution, 
        step: Dict[str, Any]
    ) -> List[str]:
        """Get agents assigned to execute a specific step"""
        
        required_roles = step.get("assigned_roles", [])
        assigned_agents = []
        
        # Map roles to actual agents
        for agent_id, role in task_execution.collaboration_plan.agent_roles.items():
            if role in required_roles or not required_roles:
                assigned_agents.append(agent_id)
        
        return assigned_agents
    
    async def _execute_collaborative_step(
        self, 
        task_execution: TaskExecution,
        step: Dict[str, Any],
        assigned_agents: List[str]
    ) -> Dict[str, Any]:
        """Execute step requiring collaboration between multiple agents"""
        
        # Schedule coordination meeting if needed
        if len(assigned_agents) > 2:
            meeting_id = await self.meeting_orchestrator.schedule_team_meeting(
                participants=assigned_agents,
                context={
                    "task_id": task_execution.id,
                    "step_id": step["step_id"],
                    "step_description": step["description"]
                },
                agent_capabilities={}  # Would be populated from agent registry
            )
            
            # Conduct meeting
            meeting_minutes = await self.meeting_orchestrator.conduct_meeting(
                meeting_id, {}
            )
            
            task_execution.meetings_conducted.append(meeting_id)
        
        # Coordinate individual agent contributions
        agent_contributions = {}
        
        for agent_id in assigned_agents:
            # Get agent from registry
            agent = self._get_agent_by_id(agent_id)
            if agent:
                contribution = await self._get_agent_step_contribution(
                    agent, task_execution, step
                )
                agent_contributions[agent_id] = contribution
        
        # Synthesize collaborative output
        collaborative_output = await self._synthesize_collaborative_output(
            step, agent_contributions, task_execution
        )
        
        return collaborative_output
    
    async def _execute_individual_step(
        self, 
        task_execution: TaskExecution,
        step: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """Execute step by individual agent"""
        
        agent = self._get_agent_by_id(agent_id)
        if not agent:
            raise Exception(f"Agent {agent_id} not found")
        
        # Prepare step context
        step_context = {
            "task_definition": asdict(task_execution.task_definition),
            "step_description": step["description"],
            "step_inputs": step.get("inputs", []),
            "expected_outputs": step.get("outputs", []),
            "previous_outputs": task_execution.intermediate_outputs
        }
        
        # Execute step through agent
        step_output = await agent.execute_task(
            task_description=step["description"],
            input_data=step_context,
            context=step_context,
            collaboration_allowed=False
        )
        
        return step_output
    
    def _get_agent_by_id(self, agent_id: str) -> Optional[AdvancedAgent]:
        """Get agent instance by ID"""
        
        # Get from expert registry
        for expert in self.expert_registry.experts.values():
            if expert.profile.agent_id == agent_id:
                return expert
        
        return None
    
    async def _get_agent_step_contribution(
        self, 
        agent: AdvancedAgent,
        task_execution: TaskExecution,
        step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get individual agent's contribution to a collaborative step"""
        
        step_context = {
            "task_definition": asdict(task_execution.task_definition),
            "step_description": step["description"],
            "collaboration_context": "multi_agent_step",
            "other_agents": [aid for aid in task_execution.assigned_agents if aid != agent.profile.agent_id]
        }
        
        contribution = await agent.execute_task(
            task_description=f"Contribute to: {step['description']}",
            input_data=step_context,
            context=step_context,
            collaboration_allowed=True
        )
        
        return contribution
    
    async def _synthesize_collaborative_output(
        self, 
        step: Dict[str, Any],
        agent_contributions: Dict[str, Dict[str, Any]],
        task_execution: TaskExecution
    ) -> Dict[str, Any]:
        """Synthesize outputs from multiple agent contributions"""
        
        system_prompt = """
        Synthesize multiple agent contributions into a unified step output.
        Ensure consistency, completeness, and quality while preserving the best insights from each contribution.
        """
        
        contributions_summary = []
        for agent_id, contribution in agent_contributions.items():
            contributions_summary.append(
                f"Agent {agent_id}: {json.dumps(contribution, indent=2)[:500]}..."
            )
        
        user_prompt = f"""
        Step: {step['name']}
        Description: {step['description']}
        Expected Outputs: {', '.join(step.get('outputs', []))}
        
        Agent Contributions:
        {chr(10).join(contributions_summary)}
        
        Synthesize these contributions into a unified output that:
        1. Combines the best elements from each contribution
        2. Resolves any conflicts or inconsistencies
        3. Ensures completeness for the step requirements
        4. Maintains high quality standards
        
        Return as JSON format.
        """
        
        try:
            from core.shared_resources import openai_manager
            
            print(f"\n🔄 Synthesizing outputs from {len(agent_contributions)} agents...")
            
            # Use streaming for synthesis to show progress
            stream = await openai_manager.create_completion(
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                stream=True
            )
            
            # Collect streaming content dengan progress display
            content = ""
            print("📝 Synthesis: ", end="", flush=True)
            
            async for chunk in openai_manager.stream_completion_text(stream):
                content += chunk
                print(chunk, end="", flush=True)
            
            print()  # New line after streaming
            
            # Parse JSON result
            synthesized_output = json.loads(content)
            synthesized_output["synthesis_metadata"] = {
                "contributor_agents": list(agent_contributions.keys()),
                "synthesis_timestamp": datetime.now().isoformat(),
                "step_id": step["step_id"]
            }
            
            return synthesized_output
            
        except Exception as e:
            # Fallback: combine contributions directly
            return {
                "synthesis_error": str(e),
                "raw_contributions": agent_contributions,
                "step_id": step["step_id"]
            }
    
    async def _validate_step_output(
        self, 
        task_execution: TaskExecution,
        step: Dict[str, Any],
        step_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate output of a workflow step"""
        
        # Create validation request
        validation_request = ValidationRequest(
            id=str(uuid.uuid4()),
            requester_id="task_orchestrator",
            content=json.dumps(step_output),
            validation_types=[
                ValidationType.COMPLETENESS,
                ValidationType.LOGICAL_CONSISTENCY,
                ValidationType.RELEVANCE
            ],
            context={
                "task_id": task_execution.id,
                "step_id": step["step_id"],
                "step_description": step["description"],
                "expected_outputs": step.get("outputs", [])
            },
            priority=5,
            deadline=None,
            required_validators=[],
            minimum_validators=2,
            created_at=datetime.now()
        )
        
        # Perform cross-validation
        try:
            cross_validation_result = await self.validation_orchestrator.cross_validate(
                validation_request
            )
            
            return {
                "step_id": step["step_id"],
                "result": cross_validation_result.final_result.value,
                "confidence": cross_validation_result.confidence_score,
                "reasoning": cross_validation_result.synthesized_feedback,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step_id": step["step_id"],
                "result": "validation_error",
                "confidence": 0.0,
                "reasoning": f"Validation failed: {str(e)}",
                "validation_timestamp": datetime.now().isoformat()
            }
    
    def _is_step_completed(self, task_execution: TaskExecution, step_id: str) -> bool:
        """Check if a workflow step is completed"""
        
        for step in task_execution.collaboration_plan.workflow_steps:
            if step["step_id"] == step_id:
                return step.get("completed", False)
        
        return False
    
    async def _perform_final_integration(self, task_execution: TaskExecution):
        """Perform final integration of all step outputs"""
        
        # Collect all intermediate outputs
        all_outputs = task_execution.intermediate_outputs
        
        # Apply integration strategy
        integration_strategy = task_execution.collaboration_plan.integration_strategy
        
        # Generate final integrated output
        final_output = await self._integrate_outputs(
            all_outputs, integration_strategy, task_execution
        )
        
        # Perform final quality assessment
        quality_score = await self._assess_final_quality(final_output, task_execution)
        
        task_execution.final_output = final_output
        task_execution.quality_score = quality_score
    
    async def _integrate_outputs(
        self, 
        outputs: Dict[str, Any],
        integration_strategy: Dict[str, Any],
        task_execution: TaskExecution
    ) -> Dict[str, Any]:
        """Integrate all step outputs into final deliverable"""
        
        system_prompt = """
        Integrate multiple workflow step outputs into a comprehensive final deliverable.
        Ensure the final output meets all task objectives and deliverable requirements.
        """
        
        user_prompt = f"""
        Task: {task_execution.task_definition.title}
        Objectives: {', '.join(task_execution.task_definition.objectives)}
        Deliverables: {', '.join(task_execution.task_definition.deliverables)}
        
        Step Outputs to Integrate:
        {json.dumps(outputs, indent=2)}
        
        Integration Strategy: {json.dumps(integration_strategy, indent=2)}
        
        Create a comprehensive final deliverable that:
        1. Meets all task objectives
        2. Includes all required deliverables
        3. Integrates insights from all workflow steps
        4. Provides actionable recommendations
        5. Maintains high quality standards
        
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
            
            integrated_output = json.loads(response.choices[0].message.content)
            integrated_output["integration_metadata"] = {
                "integrated_steps": list(outputs.keys()),
                "integration_timestamp": datetime.now().isoformat(),
                "task_id": task_execution.id
            }
            
            return integrated_output
            
        except Exception as e:
            return {
                "integration_error": str(e),
                "raw_outputs": outputs,
                "task_id": task_execution.id
            }
    
    async def _assess_final_quality(
        self, 
        final_output: Dict[str, Any],
        task_execution: TaskExecution
    ) -> float:
        """Assess quality of final output"""
        
        quality_factors = []
        
        # Check completeness
        required_deliverables = task_execution.task_definition.deliverables
        completeness_score = self._assess_completeness(final_output, required_deliverables)
        quality_factors.append(completeness_score)
        
        # Check objective achievement
        objectives = task_execution.task_definition.objectives
        objective_score = self._assess_objective_achievement(final_output, objectives)
        quality_factors.append(objective_score)
        
        # Check validation results
        validation_scores = [vr.get("confidence", 0.5) for vr in task_execution.validation_results]
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.5
        quality_factors.append(avg_validation_score)
        
        # Calculate overall quality score
        overall_quality = sum(quality_factors) / len(quality_factors)
        
        return overall_quality
    
    def _assess_completeness(
        self, 
        output: Dict[str, Any], 
        required_deliverables: List[str]
    ) -> float:
        """Assess completeness of output against required deliverables"""
        
        if not required_deliverables:
            return 1.0
        
        # Simple keyword-based assessment
        output_text = json.dumps(output).lower()
        
        found_deliverables = 0
        for deliverable in required_deliverables:
            if deliverable.lower() in output_text:
                found_deliverables += 1
        
        return found_deliverables / len(required_deliverables)
    
    def _assess_objective_achievement(
        self, 
        output: Dict[str, Any], 
        objectives: List[str]
    ) -> float:
        """Assess how well output achieves stated objectives"""
        
        if not objectives:
            return 1.0
        
        # Simple keyword-based assessment
        output_text = json.dumps(output).lower()
        
        achieved_objectives = 0
        for objective in objectives:
            # Extract key terms from objective
            objective_words = objective.lower().split()
            key_words = [word for word in objective_words if len(word) > 4]
            
            # Check if key words appear in output
            if any(word in output_text for word in key_words):
                achieved_objectives += 1
        
        return achieved_objectives / len(objectives)
    
    # Utility methods for collaboration plan creation
    async def _plan_coordination_meetings(
        self, 
        task_definition: TaskDefinition,
        collaboration_type: CollaborationType,
        workflow_steps: List[Dict[str, Any]]
    ) -> List[str]:
        """Plan coordination meetings for the task"""
        
        meetings = []
        
        # Kickoff meeting
        if len(workflow_steps) > 2:
            meetings.append("kickoff_meeting")
        
        # Milestone meetings
        milestone_steps = [step for step in workflow_steps if step.get("collaboration_required")]
        for step in milestone_steps:
            meetings.append(f"milestone_meeting_{step['step_id']}")
        
        # Final review meeting
        meetings.append("final_review_meeting")
        
        return meetings
    
    def _define_validation_checkpoints(
        self, 
        task_definition: TaskDefinition,
        workflow_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Define validation checkpoints throughout the workflow"""
        
        checkpoints = []
        
        for step in workflow_steps:
            if step.get("validation_required"):
                checkpoints.append({
                    "checkpoint_id": f"validation_{step['step_id']}",
                    "step_id": step["step_id"],
                    "validation_types": ["completeness", "quality", "relevance"],
                    "required_validators": 2,
                    "validation_criteria": step.get("outputs", [])
                })
        
        return checkpoints
    
    async def _create_integration_strategy(
        self, 
        task_definition: TaskDefinition,
        collaboration_type: CollaborationType,
        workflow_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create strategy for integrating workflow outputs"""
        
        return {
            "integration_approach": "hierarchical_synthesis",
            "integration_steps": [
                "collect_all_outputs",
                "identify_key_themes",
                "resolve_conflicts",
                "synthesize_comprehensive_solution",
                "validate_integration"
            ],
            "quality_gates": ["consistency_check", "completeness_check", "objective_alignment"],
            "integration_criteria": task_definition.success_criteria
        }
    
    def _estimate_timeline(
        self, 
        workflow_steps: List[Dict[str, Any]],
        task_definition: TaskDefinition
    ) -> Dict[str, datetime]:
        """Estimate timeline for workflow execution"""
        
        timeline = {}
        current_time = datetime.now()
        
        for step in workflow_steps:
            step_duration = timedelta(hours=step.get("duration_hours", 4))
            timeline[step["step_id"]] = current_time + step_duration
            current_time += step_duration
        
        timeline["estimated_completion"] = current_time
        
        return timeline
    
    def _identify_risk_mitigation(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify risk mitigation strategies"""
        
        risk_factors = task_analysis.get("risk_factors", [])
        mitigation_strategies = []
        
        for risk in risk_factors:
            mitigation_strategies.append({
                "risk": risk,
                "mitigation_strategy": f"Monitor and mitigate: {risk}",
                "contingency_plan": "Escalate to project manager",
                "monitoring_frequency": "daily"
            })
        
        return mitigation_strategies
    
    async def _record_execution_metrics(self, task_execution: TaskExecution):
        """Record metrics for task execution performance"""
        
        if task_execution.start_time and task_execution.end_time:
            execution_time = (task_execution.end_time - task_execution.start_time).total_seconds()
            
            if "execution_time" not in self.execution_metrics:
                self.execution_metrics["execution_time"] = []
            self.execution_metrics["execution_time"].append(execution_time)
        
        # Record quality metrics
        if task_execution.quality_score:
            if "quality_score" not in self.execution_metrics:
                self.execution_metrics["quality_score"] = []
            self.execution_metrics["quality_score"].append(task_execution.quality_score)
        
        # Record collaboration effectiveness
        if len(task_execution.assigned_agents) > 1:
            collaboration_key = "_".join(sorted(task_execution.assigned_agents))
            success_score = 1.0 if task_execution.status == TaskStatus.COMPLETED else 0.0
            
            if collaboration_key not in self.collaboration_effectiveness:
                self.collaboration_effectiveness[collaboration_key] = []
            self.collaboration_effectiveness[collaboration_key].append(success_score)
    
    # Public interface methods
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics"""
        
        total_tasks = len(self.completed_tasks)
        successful_tasks = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.COMPLETED])
        
        avg_execution_time = 0.0
        if "execution_time" in self.execution_metrics:
            times = self.execution_metrics["execution_time"]
            avg_execution_time = sum(times) / len(times) if times else 0.0
        
        avg_quality_score = 0.0
        if "quality_score" in self.execution_metrics:
            scores = self.execution_metrics["quality_score"]
            avg_quality_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "total_tasks_orchestrated": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "active_tasks": len(self.active_executions),
            "pending_tasks": len(self.pending_tasks),
            "average_execution_time_seconds": avg_execution_time,
            "average_quality_score": avg_quality_score,
            "collaboration_patterns_used": len(self.collaboration_templates),
            "expert_agents_available": len(self.expert_registry.experts)
        }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get information about currently active tasks"""
        
        active_tasks = []
        for task_execution in self.active_executions.values():
            active_tasks.append({
                "task_id": task_execution.id,
                "title": task_execution.task_definition.title,
                "status": task_execution.status.value,
                "progress_percentage": task_execution.progress_percentage,
                "current_phase": task_execution.current_phase,
                "assigned_agents": task_execution.assigned_agents,
                "start_time": task_execution.start_time.isoformat() if task_execution.start_time else None,
                "estimated_completion": task_execution.collaboration_plan.estimated_timeline.get("estimated_completion").isoformat() if task_execution.collaboration_plan.estimated_timeline.get("estimated_completion") else None
            })
        
        return active_tasks
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific task"""
        
        # Check active executions
        if task_id in self.active_executions:
            task_execution = self.active_executions[task_id]
            return {
                "task_id": task_id,
                "status": task_execution.status.value,
                "progress": task_execution.progress_percentage,
                "current_phase": task_execution.current_phase,
                "intermediate_outputs": task_execution.intermediate_outputs,
                "issues": task_execution.issues_encountered,
                "validation_results": task_execution.validation_results
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task_execution = self.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task_execution.status.value,
                "progress": 100.0,
                "final_output": task_execution.final_output,
                "quality_score": task_execution.quality_score,
                "execution_time": (task_execution.end_time - task_execution.start_time).total_seconds() if task_execution.start_time and task_execution.end_time else None
            }
        
        return {"error": f"Task {task_id} not found"}
