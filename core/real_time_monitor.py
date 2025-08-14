"""
Real-Time Monitoring System for Multi-Agent AI
Sistem monitoring real-time untuk memantau aktivitas agent
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ActivityType(Enum):
    """Jenis aktivitas yang dipantau"""
    AGENT_MESSAGE = "agent_message"
    TASK_PROGRESS = "task_progress"
    VALIDATION_RESULT = "validation_result"
    MEETING_EVENT = "meeting_event"
    MEMORY_UPDATE = "memory_update"
    COLLABORATION_EVENT = "collaboration_event"
    SYSTEM_EVENT = "system_event"

@dataclass
class ActivityEvent:
    """Event aktivitas dalam sistem"""
    id: str
    timestamp: datetime
    activity_type: ActivityType
    agent_id: Optional[str]
    description: str
    details: Dict[str, Any]
    importance: int  # 1-10

class RealTimeMonitor:
    """Monitor real-time untuk sistem multi-agent"""
    
    def __init__(self):
        self.activity_feed: List[ActivityEvent] = []
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_activities: Dict[str, List[ActivityEvent]] = {}
        self.system_metrics: Dict[str, Any] = {
            "total_messages": 0,
            "active_agents": 0,
            "completed_tasks": 0,
            "average_response_time": 0.0
        }
        
    def log_activity(self, event: ActivityEvent):
        """Log aktivitas baru"""
        self.activity_feed.append(event)
        
        # Simpan per agent jika ada
        if event.agent_id:
            if event.agent_id not in self.agent_activities:
                self.agent_activities[event.agent_id] = []
            self.agent_activities[event.agent_id].append(event)
        
        # Update metrics
        self._update_metrics(event)
        
        # Batasi ukuran feed (keep last 1000 events)
        if len(self.activity_feed) > 1000:
            self.activity_feed = self.activity_feed[-1000:]
    
    def _update_metrics(self, event: ActivityEvent):
        """Update system metrics"""
        if event.activity_type == ActivityType.AGENT_MESSAGE:
            self.system_metrics["total_messages"] += 1
        elif event.activity_type == ActivityType.TASK_PROGRESS:
            if "completed" in event.description.lower():
                self.system_metrics["completed_tasks"] += 1
    
    def get_recent_activities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Dapatkan aktivitas terbaru"""
        recent = self.activity_feed[-limit:] if limit > 0 else self.activity_feed
        
        return [
            {
                "timestamp": event.timestamp.strftime("%H:%M:%S"),
                "type": event.activity_type.value,
                "agent": event.agent_id or "System",
                "description": event.description,
                "importance": event.importance,
                "details": event.details
            }
            for event in reversed(recent)
        ]
    
    def get_agent_activity_summary(self) -> Dict[str, Dict[str, Any]]:
        """Dapatkan ringkasan aktivitas per agent"""
        summary = {}
        
        for agent_id, activities in self.agent_activities.items():
            recent_activities = [a for a in activities if a.timestamp > datetime.now() - timedelta(minutes=30)]
            
            summary[agent_id] = {
                "total_activities": len(activities),
                "recent_activities": len(recent_activities),
                "last_activity": activities[-1].timestamp if activities else None,
                "activity_types": list(set(a.activity_type.value for a in recent_activities)),
                "average_importance": sum(a.importance for a in recent_activities) / len(recent_activities) if recent_activities else 0
            }
        
        return summary
    
    def get_live_collaboration_metrics(self) -> Dict[str, Any]:
        """Dapatkan metrics kolaborasi live"""
        
        # Hitung aktivitas dalam 10 menit terakhir
        recent_time = datetime.now() - timedelta(minutes=10)
        recent_activities = [a for a in self.activity_feed if a.timestamp > recent_time]
        
        collaboration_events = [
            a for a in recent_activities 
            if a.activity_type == ActivityType.COLLABORATION_EVENT
        ]
        
        message_events = [
            a for a in recent_activities
            if a.activity_type == ActivityType.AGENT_MESSAGE
        ]
        
        validation_events = [
            a for a in recent_activities
            if a.activity_type == ActivityType.VALIDATION_RESULT
        ]
        
        active_agents = list(set(a.agent_id for a in recent_activities if a.agent_id))
        
        return {
            "active_agents_count": len(active_agents),
            "active_agents": active_agents,
            "messages_per_minute": len(message_events) / 10,
            "collaboration_events": len(collaboration_events),
            "validation_events": len(validation_events),
            "total_recent_activities": len(recent_activities),
            "system_load": min(100, len(recent_activities) * 2)  # Simple load calculation
        }
    
    def format_live_feed(self) -> List[str]:
        """Format live feed untuk display"""
        
        recent_activities = self.get_recent_activities(10)
        formatted_feed = []
        
        for activity in recent_activities:
            icon = self._get_activity_icon(activity["type"])
            importance_indicator = "🔥" if activity["importance"] >= 8 else "⚡" if activity["importance"] >= 6 else "💫"
            
            formatted_line = f"{importance_indicator} {icon} [{activity['timestamp']}] {activity['agent']}: {activity['description']}"
            formatted_feed.append(formatted_line)
        
        return formatted_feed
    
    def _get_activity_icon(self, activity_type: str) -> str:
        """Dapatkan icon untuk jenis aktivitas"""
        icons = {
            "agent_message": "💬",
            "task_progress": "📊", 
            "validation_result": "✅",
            "meeting_event": "📅",
            "memory_update": "🧠",
            "collaboration_event": "🤝",
            "system_event": "⚙️"
        }
        return icons.get(activity_type, "📝")
    
    def simulate_agent_activity(self, agent_id: str, phase: str) -> List[ActivityEvent]:
        """Simulasi aktivitas agent untuk demo"""
        
        events = []
        current_time = datetime.now()
        
        phase_activities = {
            "planning": [
                ("Analyzing task requirements...", 7),
                ("Identifying required expertise domains...", 6),
                ("Creating execution strategy...", 8),
            ],
            "analysis": [
                ("Conducting market research...", 7),
                ("Analyzing competitive landscape...", 6),
                ("Evaluating feasibility factors...", 8),
            ],
            "solution_design": [
                ("Developing solution framework...", 8),
                ("Creating implementation roadmap...", 7),
                ("Designing validation checkpoints...", 6),
            ],
            "validation": [
                ("Performing cross-validation checks...", 9),
                ("Verifying logical consistency...", 8),
                ("Confirming completeness criteria...", 7),
            ],
            "integration": [
                ("Integrating multi-agent outputs...", 8),
                ("Performing final quality assessment...", 9),
                ("Preparing final deliverables...", 7),
            ]
        }
        
        activities = phase_activities.get(phase, [("Working on task...", 5)])
        
        for i, (description, importance) in enumerate(activities):
            event = ActivityEvent(
                id=f"sim_{agent_id}_{phase}_{i}",
                timestamp=current_time + timedelta(seconds=i*2),
                activity_type=ActivityType.AGENT_MESSAGE,
                agent_id=agent_id,
                description=description,
                details={"phase": phase, "simulated": True},
                importance=importance
            )
            events.append(event)
        
        return events

class LiveDashboard:
    """Dashboard live untuk monitoring sistem"""
    
    def __init__(self, monitor: RealTimeMonitor):
        self.monitor = monitor
    
    async def display_live_dashboard(self, duration_seconds: int = 60):
        """Tampilkan dashboard live"""
        
        print("🔴 LIVE DASHBOARD MULTI-AGENT AI SYSTEM")
        print("=" * 60)
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < duration_seconds:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            print("🔴 LIVE DASHBOARD - " + datetime.now().strftime("%H:%M:%S"))
            print("=" * 60)
            
            # System metrics
            metrics = self.monitor.get_live_collaboration_metrics()
            print(f"👥 Active Agents: {metrics['active_agents_count']} | 💬 Msg/min: {metrics['messages_per_minute']:.1f} | 🔄 Load: {metrics['system_load']}%")
            print()
            
            # Live activity feed
            print("📡 LIVE ACTIVITY FEED:")
            print("-" * 30)
            live_feed = self.monitor.format_live_feed()
            for line in live_feed[-8:]:  # Show last 8 activities
                print(line)
            
            print()
            print("📊 AGENT ACTIVITY SUMMARY:")
            print("-" * 30)
            agent_summary = self.monitor.get_agent_activity_summary()
            for agent_id, summary in list(agent_summary.items())[:5]:  # Show top 5 agents
                last_activity = summary['last_activity'].strftime("%H:%M:%S") if summary['last_activity'] else "N/A"
                print(f"🤖 {agent_id}: {summary['recent_activities']} recent | Last: {last_activity}")
            
            print()
            print("⏱️  Refreshing in 3 seconds... (Ctrl+C to stop)")
            
            await asyncio.sleep(3)
    
    def display_collaboration_network(self):
        """Tampilkan network kolaborasi antar agent"""
        
        print("\n🕸️  COLLABORATION NETWORK:")
        print("-" * 35)
        
        # Simulasi network connections
        connections = [
            ("Business Strategy", "Marketing", "Strategy alignment"),
            ("Marketing", "Creative Design", "Brand development"),
            ("Technical Architect", "Quality Assurance", "Implementation review"),
            ("Project Manager", "Business Strategy", "Timeline coordination"),
            ("Quality Assurance", "Marketing", "Output validation")
        ]
        
        for agent1, agent2, purpose in connections:
            print(f"🤖 {agent1} ↔️ {agent2}: {purpose}")
    
    def display_knowledge_flow(self):
        """Tampilkan aliran knowledge antar agent"""
        
        print("\n🧠 KNOWLEDGE FLOW:")
        print("-" * 25)
        
        knowledge_flows = [
            ("Market insights", "Business Strategy → Marketing"),
            ("Technical constraints", "Technical Architect → Project Manager"),
            ("Quality standards", "Quality Assurance → All Agents"),
            ("User feedback", "Marketing → Creative Design"),
            ("Implementation risks", "Project Manager → Business Strategy")
        ]
        
        for knowledge, flow in knowledge_flows:
            print(f"💡 {knowledge}: {flow}")

# Global monitor instance
global_monitor = RealTimeMonitor()

def log_system_activity(activity_type: ActivityType, description: str, agent_id: str = None, importance: int = 5, details: Dict[str, Any] = None):
    """Helper function untuk log aktivitas sistem dengan PII protection"""
    
    # Import security utils
    try:
        from core.security_utils import audit_logger, pii_redactor, SecurityLevel
        
        # Determine security level berdasarkan activity type
        security_level = SecurityLevel.LOW
        if activity_type in [ActivityType.VALIDATION, ActivityType.DECISION]:
            security_level = SecurityLevel.HIGH
        elif activity_type in [ActivityType.COLLABORATION, ActivityType.MEMORY_STORE]:
            security_level = SecurityLevel.MEDIUM
        
        # Log ke audit system dengan PII redaction
        audit_data = {
            "agent_id": agent_id,
            "description": description,
            "details": details or {},
            "importance": importance
        }
        
        audit_logger.log_secure_event(
            f"system_activity_{activity_type.value}",
            audit_data,
            security_level
        )
        
        # Redact PII untuk real-time monitor
        redacted_description = pii_redactor.redact_text(description)
        redacted_details = pii_redactor.redact_dict(details or {})
        
    except ImportError:
        # Fallback jika security utils belum tersedia
        redacted_description = description
        redacted_details = details or {}
    
    event = ActivityEvent(
        id=f"event_{datetime.now().timestamp()}",
        timestamp=datetime.now(),
        activity_type=activity_type,
        agent_id=agent_id,
        description=redacted_description,
        details=redacted_details,
        importance=importance
    )
    
    global_monitor.log_activity(event)
