"""
Advanced Meeting Orchestration System for Multi-Agent AI
Implements team meetings and production meetings as described in NTT's research
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
from config import CONFIG
from core.communication import Message, MessageType, CommunicationStatus

class MeetingType(Enum):
    """Types of meetings in the multi-agent system"""
    TEAM_MEETING = "team_meeting"
    PRODUCTION_MEETING = "production_meeting"
    EXPERT_CONSULTATION = "expert_consultation"
    CRISIS_MEETING = "crisis_meeting"
    PLANNING_SESSION = "planning_session"
    REVIEW_SESSION = "review_session"
    BRAINSTORMING = "brainstorming"
    CONSENSUS_BUILDING = "consensus_building"

class MeetingStatus(Enum):
    """Status of meetings"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"

@dataclass
class MeetingAgenda:
    """Meeting agenda item"""
    id: str
    title: str
    description: str
    presenter: str
    duration_minutes: int
    required_participants: List[str]
    optional_participants: List[str]
    preparation_materials: List[str]
    expected_outcomes: List[str]
    priority: int  # 1-10

@dataclass
class MeetingDecision:
    """Decision made during a meeting"""
    id: str
    decision_text: str
    rationale: str
    supporting_agents: List[str]
    opposing_agents: List[str]
    abstaining_agents: List[str]
    confidence_level: float
    implementation_deadline: Optional[datetime]
    follow_up_actions: List[str]
    timestamp: datetime

@dataclass
class MeetingMinutes:
    """Meeting minutes and outcomes"""
    meeting_id: str
    key_discussions: List[str]
    decisions_made: List[MeetingDecision]
    action_items: List[Dict[str, Any]]
    unresolved_issues: List[str]
    next_meeting_topics: List[str]
    participant_contributions: Dict[str, List[str]]
    knowledge_shared: List[str]
    consensus_items: List[str]
    meeting_effectiveness_score: float

@dataclass
class Meeting:
    """Meeting representation"""
    id: str
    meeting_type: MeetingType
    title: str
    description: str
    organizer: str
    participants: List[str]
    required_participants: List[str]
    agenda: List[MeetingAgenda]
    scheduled_time: datetime
    duration_minutes: int
    status: MeetingStatus
    context: Dict[str, Any]
    preparation_materials: List[str]
    meeting_minutes: Optional[MeetingMinutes]
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        if isinstance(self.scheduled_time, str):
            self.scheduled_time = datetime.fromisoformat(self.scheduled_time)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)

class MeetingFacilitator:
    """AI facilitator for managing meeting flow and dynamics"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
    
    async def facilitate_meeting(
        self, 
        meeting: Meeting, 
        agent_capabilities: Dict[str, List[str]]
    ) -> MeetingMinutes:
        """Facilitate a complete meeting from start to finish"""
        
        meeting.status = MeetingStatus.IN_PROGRESS
        meeting.updated_at = datetime.now()
        
        # Initialize meeting minutes
        minutes = MeetingMinutes(
            meeting_id=meeting.id,
            key_discussions=[],
            decisions_made=[],
            action_items=[],
            unresolved_issues=[],
            next_meeting_topics=[],
            participant_contributions={agent: [] for agent in meeting.participants},
            knowledge_shared=[],
            consensus_items=[],
            meeting_effectiveness_score=0.0
        )
        
        # Opening phase
        await self._conduct_opening_phase(meeting, minutes)
        
        # Agenda execution
        for agenda_item in meeting.agenda:
            await self._execute_agenda_item(meeting, agenda_item, minutes, agent_capabilities)
        
        # Closing phase
        await self._conduct_closing_phase(meeting, minutes)
        
        # Calculate effectiveness score
        minutes.meeting_effectiveness_score = self._calculate_meeting_effectiveness(meeting, minutes)
        
        meeting.meeting_minutes = minutes
        meeting.status = MeetingStatus.COMPLETED
        meeting.updated_at = datetime.now()
        
        return minutes
    
    async def _conduct_opening_phase(self, meeting: Meeting, minutes: MeetingMinutes):
        """Conduct meeting opening phase"""
        
        opening_discussion = await self._generate_opening_discussion(meeting)
        minutes.key_discussions.append(f"Opening: {opening_discussion}")
        
        # Check participant readiness
        for participant in meeting.participants:
            readiness_check = await self._check_participant_readiness(participant, meeting)
            minutes.participant_contributions[participant].append(f"Readiness: {readiness_check}")
    
    async def _execute_agenda_item(
        self, 
        meeting: Meeting, 
        agenda_item: MeetingAgenda, 
        minutes: MeetingMinutes,
        agent_capabilities: Dict[str, List[str]]
    ):
        """Execute a single agenda item"""
        
        # Generate discussion for agenda item
        discussion = await self._facilitate_agenda_discussion(
            meeting, agenda_item, agent_capabilities
        )
        
        minutes.key_discussions.append(f"{agenda_item.title}: {discussion}")
        
        # Collect participant contributions
        for participant in agenda_item.required_participants + agenda_item.optional_participants:
            if participant in meeting.participants:
                contribution = await self._get_participant_contribution(
                    participant, agenda_item, agent_capabilities.get(participant, [])
                )
                minutes.participant_contributions[participant].append(contribution)
        
        # Check if decisions need to be made
        if any(outcome.startswith("Decision:") for outcome in agenda_item.expected_outcomes):
            decision = await self._facilitate_decision_making(meeting, agenda_item)
            if decision:
                minutes.decisions_made.append(decision)
        
        # Extract knowledge shared
        knowledge = await self._extract_shared_knowledge(discussion)
        minutes.knowledge_shared.extend(knowledge)
    
    async def _conduct_closing_phase(self, meeting: Meeting, minutes: MeetingMinutes):
        """Conduct meeting closing phase"""
        
        # Summarize decisions and action items
        summary = await self._generate_meeting_summary(meeting, minutes)
        minutes.key_discussions.append(f"Summary: {summary}")
        
        # Identify action items
        action_items = await self._identify_action_items(minutes)
        minutes.action_items.extend(action_items)
        
        # Check for unresolved issues
        unresolved = await self._identify_unresolved_issues(minutes)
        minutes.unresolved_issues.extend(unresolved)
        
        # Plan next meeting topics
        next_topics = await self._suggest_next_meeting_topics(meeting, minutes)
        minutes.next_meeting_topics.extend(next_topics)
    
    async def _generate_opening_discussion(self, meeting: Meeting) -> str:
        """Generate opening discussion for the meeting"""
        
        system_prompt = f"""
        You are facilitating a {meeting.meeting_type.value} for a multi-agent AI system.
        Generate an opening discussion that:
        1. Sets the meeting tone and objectives
        2. Reviews the agenda briefly
        3. Establishes ground rules for collaboration
        4. Motivates participation
        """
        
        user_prompt = f"""
        Meeting: {meeting.title}
        Description: {meeting.description}
        Participants: {', '.join(meeting.participants)}
        Agenda Items: {', '.join([item.title for item in meeting.agenda])}
        
        Generate an opening discussion for this meeting.
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
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Welcome to {meeting.title}. Let's collaborate effectively to achieve our objectives."
    
    async def _check_participant_readiness(self, participant: str, meeting: Meeting) -> str:
        """Check if participant is ready for the meeting"""
        
        # Simulate readiness check - in real implementation, this would query the agent
        readiness_factors = [
            "Has reviewed preparation materials",
            "Understands meeting objectives", 
            "Ready to contribute expertise",
            "Available for full duration"
        ]
        
        return f"Ready - {', '.join(readiness_factors[:2])}"
    
    async def _facilitate_agenda_discussion(
        self, 
        meeting: Meeting, 
        agenda_item: MeetingAgenda,
        agent_capabilities: Dict[str, List[str]]
    ) -> str:
        """Facilitate discussion for a specific agenda item"""
        
        system_prompt = f"""
        Facilitate a discussion for the agenda item: {agenda_item.title}
        
        Consider:
        1. The agenda item objectives
        2. Required participant expertise
        3. Expected outcomes
        4. Time constraints ({agenda_item.duration_minutes} minutes)
        
        Generate a structured discussion that encourages collaboration.
        """
        
        # Identify relevant expertise
        relevant_experts = []
        for participant in agenda_item.required_participants:
            if participant in agent_capabilities:
                capabilities = agent_capabilities[participant]
                relevant_experts.append(f"{participant}: {', '.join(capabilities[:2])}")
        
        user_prompt = f"""
        Agenda Item: {agenda_item.title}
        Description: {agenda_item.description}
        Expected Outcomes: {', '.join(agenda_item.expected_outcomes)}
        Relevant Experts: {'; '.join(relevant_experts)}
        
        Facilitate a productive discussion for this agenda item.
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
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Discussion on {agenda_item.title}: Collaborative exploration of {agenda_item.description}"
    
    async def _get_participant_contribution(
        self, 
        participant: str, 
        agenda_item: MeetingAgenda,
        capabilities: List[str]
    ) -> str:
        """Get contribution from a specific participant"""
        
        system_prompt = f"""
        You are {participant} with expertise in: {', '.join(capabilities)}.
        Contribute to the discussion on: {agenda_item.title}
        
        Your contribution should:
        1. Leverage your expertise
        2. Address the agenda objectives
        3. Be collaborative and constructive
        4. Offer specific insights or recommendations
        """
        
        user_prompt = f"""
        Agenda: {agenda_item.title}
        Description: {agenda_item.description}
        Expected Outcomes: {', '.join(agenda_item.expected_outcomes)}
        
        Provide your contribution as {participant}.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"{participant} contributed expertise in {', '.join(capabilities[:2])}"
    
    async def _facilitate_decision_making(
        self, 
        meeting: Meeting, 
        agenda_item: MeetingAgenda
    ) -> Optional[MeetingDecision]:
        """Facilitate decision making for agenda item"""
        
        system_prompt = """
        Facilitate a decision-making process for the agenda item.
        
        Consider:
        1. Available options and alternatives
        2. Pros and cons of each option
        3. Participant perspectives and expertise
        4. Implementation feasibility
        5. Risk factors
        
        Generate a well-reasoned decision with supporting rationale.
        """
        
        user_prompt = f"""
        Agenda: {agenda_item.title}
        Description: {agenda_item.description}
        Participants: {', '.join(agenda_item.required_participants)}
        Expected Outcomes: {', '.join(agenda_item.expected_outcomes)}
        
        Facilitate decision making and provide the decision details in JSON format:
        {{
            "decision_text": "...",
            "rationale": "...",
            "supporting_evidence": [...],
            "implementation_steps": [...],
            "confidence_level": 0.0-1.0
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
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            decision_data = json.loads(response.choices[0].message.content)
            
            return MeetingDecision(
                id=str(uuid.uuid4()),
                decision_text=decision_data.get("decision_text", ""),
                rationale=decision_data.get("rationale", ""),
                supporting_agents=agenda_item.required_participants,
                opposing_agents=[],
                abstaining_agents=[],
                confidence_level=float(decision_data.get("confidence_level", 0.7)),
                implementation_deadline=None,
                follow_up_actions=decision_data.get("implementation_steps", []),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return None
    
    async def _extract_shared_knowledge(self, discussion: str) -> List[str]:
        """Extract knowledge shared during discussion"""
        
        system_prompt = """
        Extract key knowledge and insights shared in the discussion.
        Focus on:
        1. New information or insights
        2. Best practices mentioned
        3. Lessons learned
        4. Important facts or data
        5. Strategic insights
        
        Return as a list of knowledge items.
        """
        
        user_prompt = f"""
        Discussion: {discussion}
        
        Extract shared knowledge items and return as JSON array of strings.
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
            return result.get("knowledge_items", [])
            
        except Exception as e:
            return ["Knowledge shared during discussion"]
    
    async def _generate_meeting_summary(self, meeting: Meeting, minutes: MeetingMinutes) -> str:
        """Generate meeting summary"""
        
        system_prompt = """
        Generate a concise meeting summary that captures:
        1. Key achievements and outcomes
        2. Important decisions made
        3. Major discussion points
        4. Next steps and action items
        5. Overall meeting effectiveness
        """
        
        user_prompt = f"""
        Meeting: {meeting.title}
        Key Discussions: {'; '.join(minutes.key_discussions[:5])}
        Decisions Made: {len(minutes.decisions_made)}
        Participants: {len(meeting.participants)}
        
        Generate a meeting summary.
        """
        
        try:
            from core.shared_resources import openai_manager
            
            print(f"\n📋 Generating meeting summary...")
            
            # Use streaming for meeting summary
            stream = await openai_manager.create_completion(
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=300,
                stream=True
            )
            
            # Collect streaming content dengan progress display
            content = ""
            print("📄 Summary: ", end="", flush=True)
            
            async for chunk in openai_manager.stream_completion_text(stream):
                content += chunk
                print(chunk, end="", flush=True)
            
            print()  # New line after streaming
            return content
            
        except Exception as e:
            return f"Meeting completed with {len(minutes.decisions_made)} decisions and productive collaboration."
    
    async def _identify_action_items(self, minutes: MeetingMinutes) -> List[Dict[str, Any]]:
        """Identify action items from meeting minutes"""
        
        action_items = []
        
        # Extract action items from decisions
        for decision in minutes.decisions_made:
            for action in decision.follow_up_actions:
                action_items.append({
                    "id": str(uuid.uuid4()),
                    "description": action,
                    "assignee": "TBD",  # Would be determined in real implementation
                    "deadline": decision.implementation_deadline,
                    "priority": "medium",
                    "status": "pending"
                })
        
        # Extract from discussions
        for discussion in minutes.key_discussions:
            if "action:" in discussion.lower() or "todo:" in discussion.lower():
                action_items.append({
                    "id": str(uuid.uuid4()),
                    "description": discussion[:100] + "..." if len(discussion) > 100 else discussion,
                    "assignee": "TBD",
                    "deadline": None,
                    "priority": "low",
                    "status": "pending"
                })
        
        return action_items[:10]  # Limit to 10 action items
    
    async def _identify_unresolved_issues(self, minutes: MeetingMinutes) -> List[str]:
        """Identify unresolved issues from the meeting"""
        
        unresolved = []
        
        # Check for low-confidence decisions
        for decision in minutes.decisions_made:
            if decision.confidence_level < 0.6:
                unresolved.append(f"Low confidence decision: {decision.decision_text}")
        
        # Check for conflicting viewpoints
        if len(minutes.decisions_made) == 0 and len(minutes.key_discussions) > 3:
            unresolved.append("Multiple discussions without clear resolution")
        
        return unresolved
    
    async def _suggest_next_meeting_topics(self, meeting: Meeting, minutes: MeetingMinutes) -> List[str]:
        """Suggest topics for next meeting"""
        
        next_topics = []
        
        # Follow up on unresolved issues
        for issue in minutes.unresolved_issues:
            next_topics.append(f"Follow-up: {issue}")
        
        # Review action item progress
        if minutes.action_items:
            next_topics.append("Review action item progress")
        
        # Continue strategic discussions
        if meeting.meeting_type == MeetingType.TEAM_MEETING:
            next_topics.append("Strategic planning continuation")
        elif meeting.meeting_type == MeetingType.PRODUCTION_MEETING:
            next_topics.append("Production optimization review")
        
        return next_topics[:5]  # Limit to 5 topics
    
    def _calculate_meeting_effectiveness(self, meeting: Meeting, minutes: MeetingMinutes) -> float:
        """Calculate meeting effectiveness score"""
        
        score = 0.0
        
        # Participation score
        active_participants = len([
            p for p, contributions in minutes.participant_contributions.items()
            if len(contributions) > 0
        ])
        participation_score = active_participants / len(meeting.participants) if meeting.participants else 0
        score += participation_score * 0.3
        
        # Decision making score
        decision_score = min(1.0, len(minutes.decisions_made) * 0.2)
        score += decision_score * 0.3
        
        # Knowledge sharing score
        knowledge_score = min(1.0, len(minutes.knowledge_shared) * 0.1)
        score += knowledge_score * 0.2
        
        # Action items score
        action_score = min(1.0, len(minutes.action_items) * 0.1)
        score += action_score * 0.2
        
        return min(1.0, score)

class MeetingOrchestrator:
    """Orchestrates all meeting activities in the multi-agent system"""
    
    def __init__(self):
        self.facilitator = MeetingFacilitator()
        self.scheduled_meetings: Dict[str, Meeting] = {}
        self.completed_meetings: Dict[str, Meeting] = {}
        self.meeting_history: List[Meeting] = []
        
    async def schedule_meeting(
        self, 
        meeting_type: MeetingType,
        title: str,
        description: str,
        organizer: str,
        participants: List[str],
        agenda_items: List[Dict[str, Any]],
        scheduled_time: datetime,
        duration_minutes: int = 60,
        context: Dict[str, Any] = None
    ) -> str:
        """Schedule a new meeting"""
        
        meeting_id = str(uuid.uuid4())
        
        # Create agenda items
        agenda = []
        for i, item_data in enumerate(agenda_items):
            agenda_item = MeetingAgenda(
                id=str(uuid.uuid4()),
                title=item_data.get("title", f"Agenda Item {i+1}"),
                description=item_data.get("description", ""),
                presenter=item_data.get("presenter", organizer),
                duration_minutes=item_data.get("duration_minutes", 15),
                required_participants=item_data.get("required_participants", participants),
                optional_participants=item_data.get("optional_participants", []),
                preparation_materials=item_data.get("preparation_materials", []),
                expected_outcomes=item_data.get("expected_outcomes", []),
                priority=item_data.get("priority", 5)
            )
            agenda.append(agenda_item)
        
        meeting = Meeting(
            id=meeting_id,
            meeting_type=meeting_type,
            title=title,
            description=description,
            organizer=organizer,
            participants=participants,
            required_participants=participants,
            agenda=agenda,
            scheduled_time=scheduled_time,
            duration_minutes=duration_minutes,
            status=MeetingStatus.SCHEDULED,
            context=context or {},
            preparation_materials=[],
            meeting_minutes=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.scheduled_meetings[meeting_id] = meeting
        return meeting_id
    
    async def conduct_meeting(
        self, 
        meeting_id: str, 
        agent_capabilities: Dict[str, List[str]]
    ) -> MeetingMinutes:
        """Conduct a scheduled meeting"""
        
        if meeting_id not in self.scheduled_meetings:
            raise ValueError(f"Meeting {meeting_id} not found")
        
        meeting = self.scheduled_meetings[meeting_id]
        
        # Facilitate the meeting
        minutes = await self.facilitator.facilitate_meeting(meeting, agent_capabilities)
        
        # Move to completed meetings
        self.completed_meetings[meeting_id] = meeting
        self.meeting_history.append(meeting)
        del self.scheduled_meetings[meeting_id]
        
        return minutes
    
    async def schedule_team_meeting(
        self, 
        participants: List[str],
        context: Dict[str, Any],
        agent_capabilities: Dict[str, List[str]]
    ) -> str:
        """Schedule a team meeting for collaboration and alignment"""
        
        agenda_items = [
            {
                "title": "Project Status Review",
                "description": "Review current project status and progress",
                "duration_minutes": 20,
                "expected_outcomes": ["Status clarity", "Issue identification"]
            },
            {
                "title": "Knowledge Sharing",
                "description": "Share insights and learnings from recent work",
                "duration_minutes": 25,
                "expected_outcomes": ["Knowledge transfer", "Best practices"]
            },
            {
                "title": "Collaboration Planning",
                "description": "Plan upcoming collaborative activities",
                "duration_minutes": 15,
                "expected_outcomes": ["Decision: Collaboration approach", "Action items"]
            }
        ]
        
        return await self.schedule_meeting(
            meeting_type=MeetingType.TEAM_MEETING,
            title="Team Collaboration Meeting",
            description="Regular team meeting for alignment and collaboration",
            organizer="system",
            participants=participants,
            agenda_items=agenda_items,
            scheduled_time=datetime.now() + timedelta(minutes=5),
            duration_minutes=60,
            context=context
        )
    
    async def schedule_production_meeting(
        self, 
        participants: List[str],
        project_context: Dict[str, Any],
        agent_capabilities: Dict[str, List[str]]
    ) -> str:
        """Schedule a production meeting for output review and integration"""
        
        agenda_items = [
            {
                "title": "Output Review",
                "description": "Review and validate recent outputs and deliverables",
                "duration_minutes": 25,
                "expected_outcomes": ["Quality validation", "Integration decisions"]
            },
            {
                "title": "Cross-Agent Integration",
                "description": "Integrate outputs from different agents",
                "duration_minutes": 20,
                "expected_outcomes": ["Integrated solution", "Consistency check"]
            },
            {
                "title": "Quality Assurance",
                "description": "Ensure output quality and completeness",
                "duration_minutes": 15,
                "expected_outcomes": ["Quality confirmation", "Improvement recommendations"]
            }
        ]
        
        return await self.schedule_meeting(
            meeting_type=MeetingType.PRODUCTION_MEETING,
            title="Production Review Meeting",
            description="Review and integrate agent outputs for quality assurance",
            organizer="system",
            participants=participants,
            agenda_items=agenda_items,
            scheduled_time=datetime.now() + timedelta(minutes=10),
            duration_minutes=60,
            context=project_context
        )
    
    def get_meeting_statistics(self) -> Dict[str, Any]:
        """Get meeting system statistics"""
        
        total_meetings = len(self.meeting_history)
        
        # Meeting type distribution
        type_counts = {}
        for meeting in self.meeting_history:
            meeting_type = meeting.meeting_type.value
            type_counts[meeting_type] = type_counts.get(meeting_type, 0) + 1
        
        # Average effectiveness
        effectiveness_scores = [
            meeting.meeting_minutes.meeting_effectiveness_score 
            for meeting in self.meeting_history 
            if meeting.meeting_minutes
        ]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0
        
        return {
            "total_meetings_conducted": total_meetings,
            "scheduled_meetings": len(self.scheduled_meetings),
            "completed_meetings": len(self.completed_meetings),
            "meeting_type_distribution": type_counts,
            "average_effectiveness_score": avg_effectiveness,
            "total_decisions_made": sum(
                len(meeting.meeting_minutes.decisions_made) 
                for meeting in self.meeting_history 
                if meeting.meeting_minutes
            )
        }
    
    def get_upcoming_meetings(self, days_ahead: int = 7) -> List[Meeting]:
        """Get meetings scheduled in the next N days"""
        
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        return [
            meeting for meeting in self.scheduled_meetings.values()
            if meeting.scheduled_time <= cutoff_date
        ]
    
    async def auto_schedule_periodic_meetings(
        self, 
        participants: List[str],
        agent_capabilities: Dict[str, List[str]],
        context: Dict[str, Any]
    ):
        """Automatically schedule periodic meetings based on configuration"""
        
        # Schedule team meeting if due
        last_team_meeting = self._get_last_meeting_of_type(MeetingType.TEAM_MEETING)
        if (not last_team_meeting or 
            (datetime.now() - last_team_meeting.scheduled_time).days >= CONFIG.team_meeting_frequency):
            
            await self.schedule_team_meeting(participants, context, agent_capabilities)
        
        # Schedule production meeting if due
        last_production_meeting = self._get_last_meeting_of_type(MeetingType.PRODUCTION_MEETING)
        if (not last_production_meeting or 
            (datetime.now() - last_production_meeting.scheduled_time).days >= CONFIG.production_meeting_frequency):
            
            await self.schedule_production_meeting(participants, context, agent_capabilities)
    
    def _get_last_meeting_of_type(self, meeting_type: MeetingType) -> Optional[Meeting]:
        """Get the most recent meeting of a specific type"""
        
        meetings_of_type = [
            meeting for meeting in self.meeting_history
            if meeting.meeting_type == meeting_type
        ]
        
        if not meetings_of_type:
            return None
        
        return max(meetings_of_type, key=lambda m: m.scheduled_time)
