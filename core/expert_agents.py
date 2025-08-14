"""
Specialized Expert Agents for Multi-Agent AI System
Implements domain-specific expert agents as described in NTT's research
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
from config import CONFIG, AGENT_ROLES
from core.agent import AdvancedAgent, AgentProfile, AgentCapability, AgentState
from core.communication import CommunicationProtocol, Message, MessageType
from core.memory import EpisodicMemory, SemanticMemory
import numpy as np

class ExpertiseLevel(Enum):
    """Levels of expertise"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"

class ExpertDomain(Enum):
    """Expert domains"""
    BUSINESS_STRATEGY = "business_strategy"
    MARKETING = "marketing"
    TECHNICAL_ARCHITECTURE = "technical_architecture"
    CREATIVE_DESIGN = "creative_design"
    PROJECT_MANAGEMENT = "project_management"
    QUALITY_ASSURANCE = "quality_assurance"
    RESEARCH_ANALYSIS = "research_analysis"
    FINANCIAL_PLANNING = "financial_planning"
    CUSTOMER_EXPERIENCE = "customer_experience"
    OPERATIONS = "operations"
    INNOVATION = "innovation"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class ExpertKnowledge:
    """Specialized knowledge held by expert agents"""
    id: str
    domain: ExpertDomain
    knowledge_type: str  # methodology, best_practice, case_study, framework, tool
    title: str
    description: str
    content: str
    expertise_level_required: ExpertiseLevel
    applicability_contexts: List[str]
    success_metrics: List[str]
    limitations: List[str]
    related_knowledge: List[str]
    created_at: datetime
    last_updated: datetime
    usage_count: int
    effectiveness_score: float
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)

class BusinessStrategyExpert(AdvancedAgent):
    """Expert agent specializing in business strategy and planning"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        profile = AgentProfile(
            agent_id="business_strategy_expert",
            name="Strategic Advisor",
            role="business_strategist",
            expertise_domains=["business_strategy", "market_analysis", "competitive_intelligence", "strategic_planning"],
            capabilities=[
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.RESEARCH,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            ],
            personality_traits={
                "analytical": 0.95,
                "strategic": 0.9,
                "collaborative": 0.8,
                "innovative": 0.7,
                "detail_oriented": 0.85
            },
            experience_level=0.9,
            collaboration_preference=0.8,
            learning_rate=0.3,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        super().__init__(profile, communication_protocol)
        self.expert_knowledge = self._initialize_business_knowledge()
        
    def _initialize_business_knowledge(self) -> Dict[str, ExpertKnowledge]:
        """Initialize specialized business strategy knowledge"""
        
        knowledge_base = {}
        
        # Strategic Planning Framework
        strategic_framework = ExpertKnowledge(
            id=str(uuid.uuid4()),
            domain=ExpertDomain.BUSINESS_STRATEGY,
            knowledge_type="framework",
            title="Strategic Planning Framework",
            description="Comprehensive framework for business strategic planning",
            content="""
            Strategic Planning Process:
            1. Environmental Analysis (PESTLE, Porter's Five Forces)
            2. Internal Analysis (SWOT, Core Competencies)
            3. Strategic Options Generation
            4. Strategy Evaluation and Selection
            5. Implementation Planning
            6. Performance Monitoring and Control
            
            Key Considerations:
            - Market positioning and competitive advantage
            - Resource allocation and capability building
            - Risk assessment and mitigation
            - Stakeholder alignment and communication
            """,
            expertise_level_required=ExpertiseLevel.ADVANCED,
            applicability_contexts=["strategic_planning", "business_development", "market_entry"],
            success_metrics=["strategic_clarity", "stakeholder_alignment", "implementation_feasibility"],
            limitations=["requires_market_data", "context_dependent"],
            related_knowledge=[],
            created_at=datetime.now(),
            last_updated=datetime.now(),
            usage_count=0,
            effectiveness_score=0.85
        )
        
        knowledge_base[strategic_framework.id] = strategic_framework
        
        # Market Analysis Methodology
        market_analysis = ExpertKnowledge(
            id=str(uuid.uuid4()),
            domain=ExpertDomain.BUSINESS_STRATEGY,
            knowledge_type="methodology",
            title="Market Analysis Methodology",
            description="Systematic approach to market research and analysis",
            content="""
            Market Analysis Components:
            1. Market Size and Growth Analysis
            2. Customer Segmentation and Needs Analysis
            3. Competitive Landscape Assessment
            4. Industry Trends and Drivers
            5. Regulatory and Environmental Factors
            6. Market Entry Barriers and Opportunities
            
            Tools and Techniques:
            - TAM/SAM/SOM analysis
            - Customer journey mapping
            - Competitive positioning maps
            - Trend analysis and forecasting
            """,
            expertise_level_required=ExpertiseLevel.INTERMEDIATE,
            applicability_contexts=["market_research", "product_launch", "expansion_planning"],
            success_metrics=["market_understanding", "opportunity_identification", "risk_assessment"],
            limitations=["data_availability", "market_volatility"],
            related_knowledge=[strategic_framework.id],
            created_at=datetime.now(),
            last_updated=datetime.now(),
            usage_count=0,
            effectiveness_score=0.8
        )
        
        knowledge_base[market_analysis.id] = market_analysis
        
        return knowledge_base
    
    async def provide_strategic_consultation(
        self, 
        business_context: Dict[str, Any],
        strategic_challenge: str,
        constraints: List[str] = None
    ) -> Dict[str, Any]:
        """Provide specialized strategic consultation"""
        
        # Retrieve relevant expert knowledge
        relevant_knowledge = self._find_relevant_expert_knowledge(
            strategic_challenge, business_context
        )
        
        # Generate strategic analysis
        strategic_analysis = await self._generate_strategic_analysis(
            business_context, strategic_challenge, relevant_knowledge
        )
        
        # Develop strategic recommendations
        recommendations = await self._develop_strategic_recommendations(
            strategic_analysis, constraints or []
        )
        
        # Create implementation roadmap
        roadmap = await self._create_implementation_roadmap(recommendations)
        
        return {
            "strategic_analysis": strategic_analysis,
            "recommendations": recommendations,
            "implementation_roadmap": roadmap,
            "expert_knowledge_used": [k.title for k in relevant_knowledge],
            "consultation_confidence": self._calculate_consultation_confidence(relevant_knowledge),
            "next_steps": self._suggest_next_steps(recommendations)
        }
    
    def _find_relevant_expert_knowledge(
        self, 
        challenge: str, 
        context: Dict[str, Any]
    ) -> List[ExpertKnowledge]:
        """Find relevant expert knowledge for the strategic challenge"""
        
        relevant_knowledge = []
        
        for knowledge in self.expert_knowledge.values():
            # Check applicability contexts
            context_match = any(
                ctx.lower() in challenge.lower() or 
                ctx.lower() in str(context).lower()
                for ctx in knowledge.applicability_contexts
            )
            
            # Check domain relevance
            domain_relevance = (
                knowledge.domain.value in challenge.lower() or
                any(domain in challenge.lower() for domain in ["strategy", "business", "market"])
            )
            
            if context_match or domain_relevance:
                relevant_knowledge.append(knowledge)
        
        # Sort by effectiveness score
        relevant_knowledge.sort(key=lambda k: k.effectiveness_score, reverse=True)
        
        return relevant_knowledge[:3]  # Return top 3 most relevant
    
    async def _generate_strategic_analysis(
        self, 
        business_context: Dict[str, Any],
        strategic_challenge: str,
        relevant_knowledge: List[ExpertKnowledge]
    ) -> Dict[str, Any]:
        """Generate comprehensive strategic analysis"""
        
        knowledge_context = "\n".join([
            f"Knowledge: {k.title}\nContent: {k.content[:300]}..."
            for k in relevant_knowledge
        ])
        
        system_prompt = f"""
        You are {self.profile.name}, a senior business strategy expert with deep expertise in:
        {', '.join(self.profile.expertise_domains)}
        
        Provide comprehensive strategic analysis using established frameworks and methodologies.
        """
        
        user_prompt = f"""
        Business Context: {json.dumps(business_context, indent=2)}
        Strategic Challenge: {strategic_challenge}
        
        Expert Knowledge Available:
        {knowledge_context}
        
        Provide strategic analysis including:
        1. Situation Assessment
        2. Key Strategic Issues
        3. Opportunities and Threats
        4. Competitive Positioning
        5. Strategic Options
        6. Risk Assessment
        
        Return as detailed JSON analysis.
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
                "situation_assessment": f"Analysis error: {str(e)}",
                "key_issues": [],
                "opportunities": [],
                "threats": [],
                "competitive_position": "Unknown",
                "strategic_options": [],
                "risk_assessment": []
            }

class MarketingExpert(AdvancedAgent):
    """Expert agent specializing in marketing strategy and execution"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        profile = AgentProfile(
            agent_id="marketing_expert",
            name="Marketing Strategist",
            role="marketing_specialist",
            expertise_domains=["marketing_strategy", "brand_management", "customer_engagement", "digital_marketing"],
            capabilities=[
                AgentCapability.CREATIVE_THINKING,
                AgentCapability.ANALYSIS,
                AgentCapability.COMMUNICATION,
                AgentCapability.RESEARCH
            ],
            personality_traits={
                "creative": 0.9,
                "analytical": 0.8,
                "customer_focused": 0.95,
                "collaborative": 0.85,
                "innovative": 0.9
            },
            experience_level=0.85,
            collaboration_preference=0.9,
            learning_rate=0.4,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        super().__init__(profile, communication_protocol)
        self.expert_knowledge = self._initialize_marketing_knowledge()
    
    def _initialize_marketing_knowledge(self) -> Dict[str, ExpertKnowledge]:
        """Initialize specialized marketing knowledge"""
        
        knowledge_base = {}
        
        # Customer Journey Mapping
        customer_journey = ExpertKnowledge(
            id=str(uuid.uuid4()),
            domain=ExpertDomain.MARKETING,
            knowledge_type="methodology",
            title="Customer Journey Mapping",
            description="Comprehensive approach to understanding and optimizing customer experience",
            content="""
            Customer Journey Stages:
            1. Awareness - Customer discovers need/problem
            2. Consideration - Research and evaluate options
            3. Decision - Make purchase decision
            4. Onboarding - Initial product/service experience
            5. Engagement - Ongoing usage and interaction
            6. Advocacy - Recommend to others
            
            Mapping Process:
            - Define customer personas
            - Identify touchpoints and channels
            - Map emotions and pain points
            - Identify optimization opportunities
            - Design intervention strategies
            """,
            expertise_level_required=ExpertiseLevel.INTERMEDIATE,
            applicability_contexts=["customer_experience", "marketing_strategy", "product_development"],
            success_metrics=["customer_satisfaction", "conversion_rates", "retention_rates"],
            limitations=["requires_customer_data", "personas_accuracy"],
            related_knowledge=[],
            created_at=datetime.now(),
            last_updated=datetime.now(),
            usage_count=0,
            effectiveness_score=0.88
        )
        
        knowledge_base[customer_journey.id] = customer_journey
        
        # Brand Positioning Framework
        brand_positioning = ExpertKnowledge(
            id=str(uuid.uuid4()),
            domain=ExpertDomain.MARKETING,
            knowledge_type="framework",
            title="Brand Positioning Framework",
            description="Strategic framework for developing compelling brand positioning",
            content="""
            Brand Positioning Elements:
            1. Target Audience Definition
            2. Competitive Frame of Reference
            3. Point of Difference (POD)
            4. Reason to Believe (RTB)
            5. Brand Personality and Values
            
            Positioning Statement Template:
            "For [target audience] who [need/occasion], [brand] is the [competitive frame] 
            that [point of difference] because [reason to believe]."
            
            Validation Criteria:
            - Relevance to target audience
            - Differentiation from competitors
            - Credibility and deliverability
            - Consistency across touchpoints
            """,
            expertise_level_required=ExpertiseLevel.ADVANCED,
            applicability_contexts=["brand_development", "marketing_strategy", "product_launch"],
            success_metrics=["brand_awareness", "brand_preference", "market_differentiation"],
            limitations=["market_research_required", "competitive_response"],
            related_knowledge=[customer_journey.id],
            created_at=datetime.now(),
            last_updated=datetime.now(),
            usage_count=0,
            effectiveness_score=0.82
        )
        
        knowledge_base[brand_positioning.id] = brand_positioning
        
        return knowledge_base
    
    async def develop_marketing_strategy(
        self, 
        product_context: Dict[str, Any],
        target_market: Dict[str, Any],
        business_objectives: List[str]
    ) -> Dict[str, Any]:
        """Develop comprehensive marketing strategy"""
        
        # Find relevant marketing knowledge
        relevant_knowledge = self._find_relevant_marketing_knowledge(
            product_context, target_market, business_objectives
        )
        
        # Generate marketing analysis
        market_analysis = await self._generate_marketing_analysis(
            product_context, target_market, relevant_knowledge
        )
        
        # Develop positioning strategy
        positioning = await self._develop_positioning_strategy(
            product_context, target_market, market_analysis
        )
        
        # Create marketing mix strategy
        marketing_mix = await self._create_marketing_mix_strategy(
            positioning, business_objectives
        )
        
        # Design campaign concepts
        campaign_concepts = await self._design_campaign_concepts(
            positioning, marketing_mix, target_market
        )
        
        return {
            "market_analysis": market_analysis,
            "positioning_strategy": positioning,
            "marketing_mix": marketing_mix,
            "campaign_concepts": campaign_concepts,
            "success_metrics": self._define_marketing_metrics(business_objectives),
            "implementation_timeline": self._create_marketing_timeline(campaign_concepts),
            "budget_recommendations": self._suggest_budget_allocation(marketing_mix)
        }

class TechnicalArchitectExpert(AdvancedAgent):
    """Expert agent specializing in technical architecture and implementation"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        profile = AgentProfile(
            agent_id="technical_architect_expert",
            name="Technical Architect",
            role="technical_specialist",
            expertise_domains=["system_architecture", "software_design", "technology_strategy", "implementation_planning"],
            capabilities=[
                AgentCapability.TECHNICAL_IMPLEMENTATION,
                AgentCapability.ANALYSIS,
                AgentCapability.PROBLEM_SOLVING,
                AgentCapability.DECISION_MAKING
            ],
            personality_traits={
                "analytical": 0.95,
                "systematic": 0.9,
                "detail_oriented": 0.95,
                "innovative": 0.8,
                "collaborative": 0.7
            },
            experience_level=0.92,
            collaboration_preference=0.75,
            learning_rate=0.35,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        super().__init__(profile, communication_protocol)
        self.expert_knowledge = self._initialize_technical_knowledge()
    
    def _initialize_technical_knowledge(self) -> Dict[str, ExpertKnowledge]:
        """Initialize specialized technical knowledge"""
        
        knowledge_base = {}
        
        # System Architecture Patterns
        architecture_patterns = ExpertKnowledge(
            id=str(uuid.uuid4()),
            domain=ExpertDomain.TECHNICAL_ARCHITECTURE,
            knowledge_type="framework",
            title="System Architecture Patterns",
            description="Common architectural patterns and their applications",
            content="""
            Key Architecture Patterns:
            1. Microservices Architecture
               - Decomposed services with clear boundaries
               - Independent deployment and scaling
               - Technology diversity and team autonomy
            
            2. Event-Driven Architecture
               - Asynchronous communication via events
               - Loose coupling and high scalability
               - Real-time processing capabilities
            
            3. Layered Architecture
               - Clear separation of concerns
               - Maintainable and testable code
               - Technology stack flexibility
            
            4. CQRS (Command Query Responsibility Segregation)
               - Separate read and write operations
               - Optimized performance for different operations
               - Complex domain modeling support
            
            Selection Criteria:
            - System complexity and scale requirements
            - Team expertise and organizational structure
            - Performance and availability requirements
            - Technology constraints and preferences
            """,
            expertise_level_required=ExpertiseLevel.ADVANCED,
            applicability_contexts=["system_design", "scalability_planning", "technology_selection"],
            success_metrics=["system_performance", "maintainability", "scalability"],
            limitations=["complexity_overhead", "team_expertise_required"],
            related_knowledge=[],
            created_at=datetime.now(),
            last_updated=datetime.now(),
            usage_count=0,
            effectiveness_score=0.9
        )
        
        knowledge_base[architecture_patterns.id] = architecture_patterns
        
        return knowledge_base
    
    async def design_system_architecture(
        self, 
        requirements: Dict[str, Any],
        constraints: Dict[str, Any],
        quality_attributes: List[str]
    ) -> Dict[str, Any]:
        """Design comprehensive system architecture"""
        
        # Analyze requirements and constraints
        architecture_analysis = await self._analyze_architecture_requirements(
            requirements, constraints, quality_attributes
        )
        
        # Select appropriate patterns
        selected_patterns = await self._select_architecture_patterns(
            architecture_analysis, quality_attributes
        )
        
        # Design system components
        system_design = await self._design_system_components(
            selected_patterns, requirements
        )
        
        # Create implementation plan
        implementation_plan = await self._create_implementation_plan(
            system_design, constraints
        )
        
        return {
            "architecture_analysis": architecture_analysis,
            "selected_patterns": selected_patterns,
            "system_design": system_design,
            "implementation_plan": implementation_plan,
            "quality_assurance": self._define_quality_measures(quality_attributes),
            "risk_assessment": self._assess_technical_risks(system_design),
            "technology_recommendations": self._recommend_technologies(system_design)
        }

class CreativeDesignExpert(AdvancedAgent):
    """Expert agent specializing in creative design and user experience"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        profile = AgentProfile(
            agent_id="creative_design_expert",
            name="Creative Designer",
            role="creative_strategist",
            expertise_domains=["user_experience", "visual_design", "design_thinking", "creative_strategy"],
            capabilities=[
                AgentCapability.CREATIVE_THINKING,
                AgentCapability.PROBLEM_SOLVING,
                AgentCapability.COMMUNICATION,
                AgentCapability.ANALYSIS
            ],
            personality_traits={
                "creative": 0.98,
                "empathetic": 0.9,
                "innovative": 0.95,
                "collaborative": 0.88,
                "detail_oriented": 0.8
            },
            experience_level=0.87,
            collaboration_preference=0.9,
            learning_rate=0.45,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        super().__init__(profile, communication_protocol)
        self.expert_knowledge = self._initialize_design_knowledge()

class ProjectManagementExpert(AdvancedAgent):
    """Expert agent specializing in project management and coordination"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        profile = AgentProfile(
            agent_id="project_management_expert",
            name="Project Manager",
            role="project_manager",
            expertise_domains=["project_management", "resource_planning", "risk_management", "stakeholder_management"],
            capabilities=[
                AgentCapability.PROJECT_MANAGEMENT,
                AgentCapability.COMMUNICATION,
                AgentCapability.DECISION_MAKING,
                AgentCapability.PROBLEM_SOLVING
            ],
            personality_traits={
                "organized": 0.95,
                "collaborative": 0.92,
                "detail_oriented": 0.9,
                "diplomatic": 0.88,
                "results_oriented": 0.93
            },
            experience_level=0.88,
            collaboration_preference=0.95,
            learning_rate=0.3,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        super().__init__(profile, communication_protocol)
        self.expert_knowledge = self._initialize_pm_knowledge()

class QualityAssuranceExpert(AdvancedAgent):
    """Expert agent specializing in quality assurance and validation"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        profile = AgentProfile(
            agent_id="quality_assurance_expert",
            name="Quality Assurance Specialist",
            role="quality_assurance",
            expertise_domains=["quality_control", "testing_strategies", "process_improvement", "standards_compliance"],
            capabilities=[
                AgentCapability.QUALITY_ASSURANCE,
                AgentCapability.ANALYSIS,
                AgentCapability.PROBLEM_SOLVING,
                AgentCapability.TECHNICAL_IMPLEMENTATION
            ],
            personality_traits={
                "meticulous": 0.98,
                "analytical": 0.95,
                "systematic": 0.93,
                "collaborative": 0.8,
                "perfectionist": 0.9
            },
            experience_level=0.9,
            collaboration_preference=0.8,
            learning_rate=0.25,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        super().__init__(profile, communication_protocol)
        self.expert_knowledge = self._initialize_qa_knowledge()

class ExpertAgentRegistry:
    """Registry and factory for expert agents"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        self.communication_protocol = communication_protocol
        self.experts: Dict[str, AdvancedAgent] = {}
        self.expertise_map: Dict[ExpertDomain, List[str]] = {}
        
    def initialize_expert_agents(self) -> Dict[str, AdvancedAgent]:
        """Initialize all expert agents"""
        
        experts = {
            "business_strategy": BusinessStrategyExpert(self.communication_protocol),
            "marketing": MarketingExpert(self.communication_protocol),
            "technical_architecture": TechnicalArchitectExpert(self.communication_protocol),
            "creative_design": CreativeDesignExpert(self.communication_protocol),
            "project_management": ProjectManagementExpert(self.communication_protocol),
            "quality_assurance": QualityAssuranceExpert(self.communication_protocol)
        }
        
        self.experts = experts
        self._build_expertise_map()
        
        return experts
    
    def _build_expertise_map(self):
        """Build mapping of expertise domains to expert agents"""
        
        for expert_id, expert in self.experts.items():
            for domain_str in expert.profile.expertise_domains:
                # Map string domains to ExpertDomain enum
                for domain_enum in ExpertDomain:
                    if domain_enum.value in domain_str or domain_str in domain_enum.value:
                        if domain_enum not in self.expertise_map:
                            self.expertise_map[domain_enum] = []
                        self.expertise_map[domain_enum].append(expert_id)
    
    def find_experts_by_domain(self, domain: ExpertDomain) -> List[AdvancedAgent]:
        """Find expert agents by domain"""
        
        expert_ids = self.expertise_map.get(domain, [])
        return [self.experts[expert_id] for expert_id in expert_ids if expert_id in self.experts]
    
    def find_experts_by_capability(self, capability: AgentCapability) -> List[AdvancedAgent]:
        """Find expert agents by capability"""
        
        matching_experts = []
        for expert in self.experts.values():
            if capability in expert.profile.capabilities:
                matching_experts.append(expert)
        
        return matching_experts
    
    def recommend_expert_team(
        self, 
        task_requirements: Dict[str, Any],
        required_domains: List[ExpertDomain],
        team_size_limit: int = 5
    ) -> List[AdvancedAgent]:
        """Recommend optimal expert team for a task"""
        
        recommended_experts = []
        
        # Add experts for required domains
        for domain in required_domains:
            domain_experts = self.find_experts_by_domain(domain)
            if domain_experts and len(recommended_experts) < team_size_limit:
                # Select best expert for domain based on experience
                best_expert = max(domain_experts, key=lambda e: e.profile.experience_level)
                if best_expert not in recommended_experts:
                    recommended_experts.append(best_expert)
        
        # Fill remaining slots with complementary expertise
        while len(recommended_experts) < team_size_limit:
            # Find expert with highest collaboration preference not yet selected
            available_experts = [
                expert for expert in self.experts.values()
                if expert not in recommended_experts
            ]
            
            if not available_experts:
                break
            
            best_collaborator = max(
                available_experts, 
                key=lambda e: e.profile.collaboration_preference
            )
            recommended_experts.append(best_collaborator)
        
        return recommended_experts
    
    async def coordinate_expert_consultation(
        self, 
        consultation_request: str,
        context: Dict[str, Any],
        required_expertise: List[ExpertDomain]
    ) -> Dict[str, Any]:
        """Coordinate consultation across multiple expert agents"""
        
        # Find relevant experts
        expert_team = []
        for domain in required_expertise:
            domain_experts = self.find_experts_by_domain(domain)
            expert_team.extend(domain_experts)
        
        # Remove duplicates
        expert_team = list(set(expert_team))
        
        if not expert_team:
            return {
                "error": "No experts available for required domains",
                "required_expertise": [d.value for d in required_expertise]
            }
        
        # Collect consultations from each expert
        consultations = {}
        consultation_tasks = []
        
        for expert in expert_team:
            task = self._get_expert_consultation(expert, consultation_request, context)
            consultation_tasks.append((expert.profile.agent_id, task))
        
        # Execute consultations in parallel
        for expert_id, task in consultation_tasks:
            try:
                consultation_result = await task
                consultations[expert_id] = consultation_result
            except Exception as e:
                consultations[expert_id] = {"error": str(e)}
        
        # Synthesize expert opinions
        synthesized_response = await self._synthesize_expert_consultations(
            consultation_request, consultations, context
        )
        
        return {
            "consultation_request": consultation_request,
            "expert_consultations": consultations,
            "synthesized_response": synthesized_response,
            "participating_experts": [expert.profile.name for expert in expert_team],
            "consultation_confidence": self._calculate_consultation_confidence(consultations)
        }
    
    async def _get_expert_consultation(
        self, 
        expert: AdvancedAgent, 
        request: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get consultation from a specific expert"""
        
        if hasattr(expert, 'provide_strategic_consultation'):
            return await expert.provide_strategic_consultation(context, request)
        elif hasattr(expert, 'develop_marketing_strategy'):
            return await expert.develop_marketing_strategy(context, {}, [request])
        else:
            # Generic consultation
            consultation_message = Message(
                id=str(uuid.uuid4()),
                sender_id="expert_registry",
                receiver_id=expert.profile.agent_id,
                message_type=MessageType.EXPERT_CONSULTATION,
                content=request,
                context=context,
                timestamp=datetime.now(),
                priority=5,
                requires_response=True
            )
            
            return await expert._provide_expert_consultation(consultation_message)
    
    async def _synthesize_expert_consultations(
        self, 
        request: str,
        consultations: Dict[str, Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize multiple expert consultations into unified response"""
        
        client = openai.OpenAI(api_key=CONFIG.openai_api_key)
        
        system_prompt = """
        You are a senior consultant synthesizing insights from multiple domain experts.
        Create a unified, coherent response that integrates the diverse expert perspectives.
        
        Focus on:
        1. Common themes and agreements
        2. Complementary insights
        3. Conflicting viewpoints and resolutions
        4. Comprehensive recommendations
        5. Implementation priorities
        """
        
        # Prepare expert insights summary
        expert_insights = []
        for expert_id, consultation in consultations.items():
            if "error" not in consultation:
                expert_insights.append(f"{expert_id}: {json.dumps(consultation, indent=2)[:500]}...")
        
        user_prompt = f"""
        Consultation Request: {request}
        Context: {json.dumps(context, indent=2)}
        
        Expert Consultations:
        {chr(10).join(expert_insights)}
        
        Synthesize these expert perspectives into a unified response with:
        1. Executive Summary
        2. Key Insights and Recommendations
        3. Implementation Approach
        4. Success Metrics
        5. Risk Considerations
        
        Return as JSON format.
        """
        
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "executive_summary": "Error synthesizing expert consultations",
                "error": str(e),
                "expert_count": len(consultations)
            }
    
    def _calculate_consultation_confidence(self, consultations: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall confidence in consultation results"""
        
        successful_consultations = [
            c for c in consultations.values() 
            if "error" not in c
        ]
        
        if not successful_consultations:
            return 0.0
        
        # Base confidence on number of successful consultations
        base_confidence = len(successful_consultations) / len(consultations)
        
        # Boost confidence if multiple experts agree
        if len(successful_consultations) > 1:
            base_confidence += 0.2
        
        return min(1.0, base_confidence)
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get statistics about expert agents"""
        
        total_experts = len(self.experts)
        domain_coverage = len(self.expertise_map)
        
        # Calculate average experience level
        avg_experience = sum(
            expert.profile.experience_level 
            for expert in self.experts.values()
        ) / total_experts if total_experts > 0 else 0.0
        
        # Calculate collaboration readiness
        avg_collaboration = sum(
            expert.profile.collaboration_preference 
            for expert in self.experts.values()
        ) / total_experts if total_experts > 0 else 0.0
        
        return {
            "total_experts": total_experts,
            "domain_coverage": domain_coverage,
            "average_experience_level": avg_experience,
            "average_collaboration_preference": avg_collaboration,
            "expertise_domains": list(self.expertise_map.keys()),
            "expert_details": {
                expert_id: {
                    "name": expert.profile.name,
                    "role": expert.profile.role,
                    "expertise_domains": expert.profile.expertise_domains,
                    "experience_level": expert.profile.experience_level,
                    "current_state": expert.state.value
                }
                for expert_id, expert in self.experts.items()
            }
        }
