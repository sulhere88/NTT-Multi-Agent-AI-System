"""
Advanced Communication System for Multi-Agent AI
Implements dialogue-based collaboration and intent understanding
Based on NTT's Multi-Agent AI Technology
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import openai
from config import CONFIG
from core.real_time_monitor import log_system_activity, ActivityType

class MessageType(Enum):
    """Types of messages in agent communication"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    MEETING_INVITATION = "meeting_invitation"
    MEETING_RESPONSE = "meeting_response"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    INTENT_CLARIFICATION = "intent_clarification"
    CONSENSUS_BUILDING = "consensus_building"
    EXPERT_CONSULTATION = "expert_consultation"

class CommunicationStatus(Enum):
    """Status of communication"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_CLARIFICATION = "requires_clarification"

@dataclass
class Message:
    """Individual message in agent communication"""
    id: str
    sender_id: str
    receiver_id: str  # Can be "broadcast" for group messages
    message_type: MessageType
    content: str
    context: Dict[str, Any]
    timestamp: datetime
    priority: int  # 1-10, higher is more urgent
    requires_response: bool
    parent_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class Conversation:
    """A conversation thread between agents"""
    id: str
    participants: List[str]
    topic: str
    messages: List[Message]
    status: CommunicationStatus
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any]
    consensus_reached: bool = False
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)

class IntentAnalyzer:
    """Analyzes and understands agent intentions from messages"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
    
    async def analyze_intent(self, message: Message) -> Dict[str, Any]:
        """Analyze the intent behind a message"""
        
        system_prompt = """
        You are an expert at analyzing communication intent in multi-agent AI systems.
        Analyze the given message and extract:
        1. Primary intent (what the sender wants)
        2. Secondary intents (additional goals)
        3. Emotional tone (collaborative, urgent, questioning, etc.)
        4. Required actions from receiver
        5. Urgency level (1-10)
        6. Complexity level (simple, moderate, complex)
        7. Knowledge domain required
        8. Collaboration type needed (information, validation, brainstorming, decision)
        
        Return as JSON format.
        """
        
        user_prompt = f"""
        Message Type: {message.message_type.value}
        Sender: {message.sender_id}
        Receiver: {message.receiver_id}
        Content: {message.content}
        Context: {json.dumps(message.context, indent=2)}
        Priority: {message.priority}
        
        Analyze this message's intent and requirements.
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
            
            intent_analysis = json.loads(response.choices[0].message.content)
            return intent_analysis
            
        except Exception as e:
            # Fallback analysis
            return {
                "primary_intent": "unknown",
                "secondary_intents": [],
                "emotional_tone": "neutral",
                "required_actions": ["respond"],
                "urgency_level": message.priority,
                "complexity_level": "moderate",
                "knowledge_domain": "general",
                "collaboration_type": "information",
                "error": str(e)
            }
    
    async def suggest_response_strategy(self, intent_analysis: Dict[str, Any], receiver_capabilities: List[str]) -> Dict[str, Any]:
        """Suggest how the receiver should respond based on intent analysis"""
        
        system_prompt = """
        Based on the intent analysis and receiver capabilities, suggest the optimal response strategy.
        Consider:
        1. Response type needed
        2. Information to gather before responding
        3. Other agents to consult
        4. Response priority and timeline
        5. Collaboration approach
        """
        
        user_prompt = f"""
        Intent Analysis: {json.dumps(intent_analysis, indent=2)}
        Receiver Capabilities: {receiver_capabilities}
        
        Suggest response strategy.
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
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "response_type": "direct",
                "information_needed": [],
                "agents_to_consult": [],
                "priority": "normal",
                "timeline": "immediate",
                "collaboration_approach": "individual",
                "error": str(e)
            }

class DialogueManager:
    """Manages dialogue flow and conversation dynamics"""
    
    def __init__(self):
        self.intent_analyzer = IntentAnalyzer()
        self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
    
    async def facilitate_dialogue(
        self, 
        participants: List[str], 
        topic: str, 
        context: Dict[str, Any],
        agent_capabilities: Dict[str, List[str]]
    ) -> Conversation:
        """Facilitate a multi-agent dialogue"""
        
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            participants=participants,
            topic=topic,
            messages=[],
            status=CommunicationStatus.IN_PROGRESS,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            context=context
        )
        
        # Initialize dialogue with opening message
        opening_message = await self._generate_opening_message(topic, participants, context)
        conversation.messages.append(opening_message)
        
        # Facilitate rounds of dialogue
        max_rounds = 10
        round_count = 0
        
        while (not conversation.consensus_reached and 
               round_count < max_rounds and 
               conversation.status == CommunicationStatus.IN_PROGRESS):
            
            # Analyze current state
            dialogue_state = await self._analyze_dialogue_state(conversation)
            
            if dialogue_state["consensus_level"] > 0.8:
                conversation.consensus_reached = True
                conversation.status = CommunicationStatus.COMPLETED
                break
            
            # Generate next round of messages
            next_messages = await self._generate_dialogue_round(
                conversation, dialogue_state, agent_capabilities
            )
            
            conversation.messages.extend(next_messages)
            conversation.updated_at = datetime.now()
            round_count += 1
        
        return conversation
    
    async def _generate_opening_message(
        self, 
        topic: str, 
        participants: List[str], 
        context: Dict[str, Any]
    ) -> Message:
        """Generate an opening message for the dialogue"""
        
        system_prompt = """
        Generate an opening message to start a collaborative dialogue between AI agents.
        The message should:
        1. Clearly state the topic and objectives
        2. Set collaborative tone
        3. Invite participation from all agents
        4. Provide necessary context
        """
        
        user_prompt = f"""
        Topic: {topic}
        Participants: {', '.join(participants)}
        Context: {json.dumps(context, indent=2)}
        
        Generate an opening message for this multi-agent dialogue.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6
            )
            
            content = response.choices[0].message.content
            
        except Exception as e:
            content = f"Let's collaborate on: {topic}. All agents please share your perspectives and expertise."
        
        return Message(
            id=str(uuid.uuid4()),
            sender_id="dialogue_manager",
            receiver_id="broadcast",
            message_type=MessageType.COLLABORATION_REQUEST,
            content=content,
            context=context,
            timestamp=datetime.now(),
            priority=7,
            requires_response=True,
            thread_id=str(uuid.uuid4())
        )
    
    async def _analyze_dialogue_state(self, conversation: Conversation) -> Dict[str, Any]:
        """Analyze the current state of dialogue"""
        
        if not conversation.messages:
            return {"consensus_level": 0.0, "participation_balance": 0.0, "information_completeness": 0.0}
        
        system_prompt = """
        Analyze the dialogue state based on the conversation messages.
        Evaluate:
        1. Consensus level (0.0-1.0) - how much agreement exists
        2. Participation balance (0.0-1.0) - how evenly participants contribute
        3. Information completeness (0.0-1.0) - how complete the information is
        4. Next required actions
        5. Potential blockers or conflicts
        
        Return as JSON.
        """
        
        # Prepare conversation summary
        message_summary = []
        for msg in conversation.messages[-10:]:  # Last 10 messages
            message_summary.append({
                "sender": msg.sender_id,
                "type": msg.message_type.value,
                "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            })
        
        user_prompt = f"""
        Conversation Topic: {conversation.topic}
        Participants: {', '.join(conversation.participants)}
        Recent Messages: {json.dumps(message_summary, indent=2)}
        
        Analyze the current dialogue state.
        """
        
        try:
            from core.shared_resources import openai_manager
            
            response = await openai_manager.create_completion(
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                use_cache=True
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "consensus_level": 0.5,
                "participation_balance": 0.7,
                "information_completeness": 0.6,
                "next_actions": ["continue_discussion"],
                "blockers": [],
                "error": str(e)
            }
    
    async def _generate_dialogue_round(
        self, 
        conversation: Conversation, 
        dialogue_state: Dict[str, Any],
        agent_capabilities: Dict[str, List[str]]
    ) -> List[Message]:
        """Generate the next round of dialogue messages"""
        
        messages = []
        
        # Determine which agents should speak next
        next_speakers = self._determine_next_speakers(
            conversation, dialogue_state, agent_capabilities
        )
        
        for speaker in next_speakers:
            message = await self._generate_agent_message(
                speaker, conversation, dialogue_state, agent_capabilities.get(speaker, [])
            )
            if message:
                messages.append(message)
        
        return messages
    
    def _determine_next_speakers(
        self, 
        conversation: Conversation, 
        dialogue_state: Dict[str, Any],
        agent_capabilities: Dict[str, List[str]]
    ) -> List[str]:
        """Determine which agents should speak in the next round"""
        
        # Count recent participation
        recent_messages = conversation.messages[-5:] if len(conversation.messages) >= 5 else conversation.messages
        speaker_counts = {}
        
        for msg in recent_messages:
            if msg.sender_id != "dialogue_manager":
                speaker_counts[msg.sender_id] = speaker_counts.get(msg.sender_id, 0) + 1
        
        # Prioritize agents who haven't spoken recently
        next_speakers = []
        for participant in conversation.participants:
            if participant not in speaker_counts or speaker_counts[participant] == 0:
                next_speakers.append(participant)
        
        # If all have spoken, select based on expertise relevance
        if not next_speakers and len(conversation.participants) > 0:
            next_speakers = [conversation.participants[0]]  # Fallback to first participant
        
        return next_speakers[:3]  # Limit to 3 speakers per round
    
    async def _generate_agent_message(
        self, 
        agent_id: str, 
        conversation: Conversation, 
        dialogue_state: Dict[str, Any],
        agent_capabilities: List[str]
    ) -> Optional[Message]:
        """Generate a message from a specific agent"""
        
        system_prompt = f"""
        You are agent {agent_id} with capabilities: {', '.join(agent_capabilities)}.
        Generate a thoughtful response to the ongoing dialogue.
        Your response should:
        1. Build on previous messages
        2. Contribute your unique perspective
        3. Move the dialogue forward
        4. Seek collaboration when needed
        5. Be concise but substantive
        """
        
        # Prepare conversation context
        recent_messages = conversation.messages[-5:]
        context_summary = []
        for msg in recent_messages:
            context_summary.append(f"{msg.sender_id}: {msg.content}")
        
        user_prompt = f"""
        Topic: {conversation.topic}
        Dialogue State: {json.dumps(dialogue_state, indent=2)}
        Recent Messages:
        {chr(10).join(context_summary)}
        
        Generate your response as {agent_id}.
        """
        
        try:
            from core.shared_resources import openai_manager
            
            # Gunakan streaming untuk dialog panjang
            stream = await openai_manager.create_completion(
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            
            # Collect streaming content dengan real-time display
            content = ""
            print(f"\n[{agent_id}] ", end="", flush=True)
            
            async for chunk in openai_manager.stream_completion_text(stream):
                content += chunk
                print(chunk, end="", flush=True)  # Real-time display
            
            print()  # New line after streaming
            
            return Message(
                id=str(uuid.uuid4()),
                sender_id=agent_id,
                receiver_id="broadcast",
                message_type=MessageType.COLLABORATION_RESPONSE,
                content=content,
                context={"dialogue_round": len(conversation.messages)},
                timestamp=datetime.now(),
                priority=5,
                requires_response=False,
                thread_id=conversation.messages[0].thread_id if conversation.messages else None
            )
            
        except Exception as e:
            logging.error(f"Error generating message for {agent_id}: {str(e)}")
            return None

class CommunicationProtocol:
    """Defines communication protocols and message handling"""
    
    def __init__(self):
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.conversation_cache: Dict[str, Conversation] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.intent_analyzer = IntentAnalyzer()
        self.dialogue_manager = DialogueManager()
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def send_message(self, message: Message) -> bool:
        """Send a message through the communication system"""
        try:
            # Analyze intent before sending
            intent_analysis = await self.intent_analyzer.analyze_intent(message)
            message.metadata["intent_analysis"] = intent_analysis
            
            # Log to real-time monitor
            log_system_activity(
                ActivityType.AGENT_MESSAGE,
                f"{message.sender_id} → {message.receiver_id}: {message.content[:50]}...",
                agent_id=message.sender_id,
                importance=message.priority,
                details={
                    "message_type": message.message_type.value,
                    "receiver": message.receiver_id,
                    "intent_analysis": intent_analysis
                }
            )
            
            # Queue message for processing
            await self.message_queue.put(message)
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    async def process_messages(self):
        """Process messages from the queue"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._handle_message(message)
                self.message_queue.task_done()
                
            except Exception as e:
                print(f"Error processing message: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: Message):
        """Handle an individual message"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            # Default handling
            print(f"No handler for message type: {message.message_type}")
    
    async def start_conversation(
        self, 
        participants: List[str], 
        topic: str, 
        context: Dict[str, Any],
        agent_capabilities: Dict[str, List[str]]
    ) -> str:
        """Start a new conversation"""
        
        conversation = await self.dialogue_manager.facilitate_dialogue(
            participants, topic, context, agent_capabilities
        )
        
        # Log conversation start
        log_system_activity(
            ActivityType.COLLABORATION_EVENT,
            f"Conversation started: {topic}",
            importance=7,
            details={
                "conversation_id": conversation.id,
                "participants": participants,
                "topic": topic
            }
        )
        
        self.conversation_cache[conversation.id] = conversation
        return conversation.id
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversation_cache.get(conversation_id)
    
    async def broadcast_message(
        self, 
        sender_id: str, 
        message_type: MessageType, 
        content: str, 
        context: Dict[str, Any],
        recipients: List[str]
    ) -> List[str]:
        """Broadcast a message to multiple recipients"""
        
        message_ids = []
        for recipient in recipients:
            message = Message(
                id=str(uuid.uuid4()),
                sender_id=sender_id,
                receiver_id=recipient,
                message_type=message_type,
                content=content,
                context=context,
                timestamp=datetime.now(),
                priority=5,
                requires_response=False
            )
            
            success = await self.send_message(message)
            if success:
                message_ids.append(message.id)
        
        return message_ids
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        total_conversations = len(self.conversation_cache)
        active_conversations = len([
            conv for conv in self.conversation_cache.values()
            if conv.status == CommunicationStatus.IN_PROGRESS
        ])
        
        total_messages = sum(
            len(conv.messages) for conv in self.conversation_cache.values()
        )
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "completed_conversations": total_conversations - active_conversations,
            "total_messages": total_messages,
            "queue_size": self.message_queue.qsize(),
            "registered_handlers": len(self.message_handlers)
        }
    
    def get_real_time_conversation_feed(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get real-time feed of conversation messages"""
        
        feed = []
        
        if conversation_id and conversation_id in self.conversation_cache:
            # Get specific conversation
            conversation = self.conversation_cache[conversation_id]
            for message in conversation.messages[-5:]:  # Last 5 messages
                feed.append({
                    "timestamp": message.timestamp.strftime("%H:%M:%S"),
                    "sender": message.sender_id,
                    "receiver": message.receiver_id,
                    "type": message.message_type.value,
                    "content": message.content[:100] + "..." if len(message.content) > 100 else message.content,
                    "priority": message.priority
                })
        else:
            # Get all recent messages from all conversations
            all_messages = []
            for conversation in self.conversation_cache.values():
                all_messages.extend(conversation.messages)
            
            # Sort by timestamp and get recent ones
            all_messages.sort(key=lambda m: m.timestamp, reverse=True)
            
            for message in all_messages[:10]:  # Last 10 messages across all conversations
                feed.append({
                    "timestamp": message.timestamp.strftime("%H:%M:%S"),
                    "sender": message.sender_id,
                    "receiver": message.receiver_id,
                    "type": message.message_type.value,
                    "content": message.content[:100] + "..." if len(message.content) > 100 else message.content,
                    "priority": message.priority
                })
        
        return feed
    
    def get_active_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active agents in conversations"""
        
        agent_status = {}
        
        for conversation in self.conversation_cache.values():
            if conversation.status == CommunicationStatus.IN_PROGRESS:
                for participant in conversation.participants:
                    if participant not in agent_status:
                        agent_status[participant] = {
                            "active_conversations": 0,
                            "messages_sent": 0,
                            "last_activity": None
                        }
                    
                    agent_status[participant]["active_conversations"] += 1
                    
                    # Count messages sent by this agent
                    agent_messages = [m for m in conversation.messages if m.sender_id == participant]
                    agent_status[participant]["messages_sent"] += len(agent_messages)
                    
                    # Get last activity
                    if agent_messages:
                        last_message = max(agent_messages, key=lambda m: m.timestamp)
                        agent_status[participant]["last_activity"] = last_message.timestamp
        
        return agent_status
