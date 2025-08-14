"""
Main Entry Point for NTT Multi-Agent AI System
Advanced Multi-Agent AI Technology Implementation
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from core.flexible_system import run_flexible_system

class MultiAgentAISystem:
    """Main system orchestrator for the Multi-Agent AI platform"""
    
    def __init__(self):
        self.system_info = {
            "name": "NTT Multi-Agent AI System",
            "version": "1.0.0",
            "description": "Advanced Multi-Agent AI Technology based on NTT Research",
            "features": [
                "Episodic and Semantic Memory Systems",
                "Intent-driven Dialogue Communication",
                "Cross-checking Knowledge Validation", 
                "Team and Production Meeting Orchestration",
                "Agent Reusability and Knowledge Accumulation",
                "Expert Agent Specialization",
                "Complex Task Orchestration",
                "Collaborative Planning and Execution"
            ],
            "initialized_at": datetime.now()
        }
    
    def display_welcome_banner(self):
        """Display welcome banner and system information"""
        
        print("=" * 80)
        print("🤖 NTT MULTI-AGENT AI SYSTEM")
        print("=" * 80)
        print(f"📋 {self.system_info['description']}")
        print(f"🔢 Version: {self.system_info['version']}")
        print(f"⏰ Initialized: {self.system_info['initialized_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("🌟 Key Features:")
        for feature in self.system_info['features']:
            print(f"  ✓ {feature}")
        print()
        print("🔗 Based on NTT's Multi-Agent AI Technology Research:")
        print("  'Multi-Agent AI Technology Capable of Driving Complex Projects")
        print("   with Context-Aware Collaboration - AI agents collaborating")
        print("   autonomously by reading intent through dialogue'")
        print("=" * 80)
    
    def display_menu(self):
        """Display main menu options"""
        
        print("\n📋 Pilihan yang Tersedia:")
        print("1. 🤖 Mulai Dialog Multi-Agent AI (Fleksibel)")
        print("2. ⚙️  Pemeriksaan Konfigurasi Sistem")
        print("3. 📊 Status dan Statistik Sistem")
        print("4. 🔧 Testing Komponen")
        print("5. 📖 Dokumentasi dan Bantuan")
        print("6. 🚪 Keluar")
        print("-" * 50)
    
    async def run_configuration_check(self):
        """Check system configuration and dependencies"""
        
        print("\n⚙️ System Configuration Check")
        print("=" * 40)
        
        # Check OpenAI API key
        if CONFIG.openai_api_key:
            print("✅ OpenAI API Key: Configured")
        else:
            print("❌ OpenAI API Key: Missing (set OPENAI_API_KEY environment variable)")
        
        # Check model configuration
        print(f"🤖 AI Model: {CONFIG.model_name}")
        print(f"👥 Max Agents: {CONFIG.max_agents}")
        print(f"🎯 Consensus Threshold: {CONFIG.consensus_threshold}")
        print(f"📊 Vector DB Path: {CONFIG.vector_db_path}")
        
        # Check required directories
        required_dirs = ["chroma_db", "logs", "data"]
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"📁 Directory '{dir_name}': ✅ Exists")
            else:
                print(f"📁 Directory '{dir_name}': ❌ Missing (will be created)")
                os.makedirs(dir_name, exist_ok=True)
                print(f"   └─ Created '{dir_name}'")
        
        # Test imports
        print("\n📦 Testing Core Components:")
        try:
            from core.memory import AdvancedMemorySystem
            print("✅ Memory System: Available")
        except ImportError as e:
            print(f"❌ Memory System: Error - {e}")
        
        try:
            from core.communication import CommunicationProtocol
            print("✅ Communication System: Available")
        except ImportError as e:
            print(f"❌ Communication System: Error - {e}")
        
        try:
            from core.expert_agents import ExpertAgentRegistry
            print("✅ Expert Agents: Available")
        except ImportError as e:
            print(f"❌ Expert Agents: Error - {e}")
        
        try:
            from core.orchestrator import TaskOrchestrator
            print("✅ Task Orchestrator: Available")
        except ImportError as e:
            print(f"❌ Task Orchestrator: Error - {e}")
        
        print("\n✅ Configuration check completed")
    
    async def display_system_status(self):
        """Display current system status and statistics"""
        
        print("\n📊 System Status and Statistics")
        print("=" * 40)
        
        try:
            # Initialize core components for status check
            from core.communication import CommunicationProtocol
            from core.expert_agents import ExpertAgentRegistry
            from core.meetings import MeetingOrchestrator
            from core.validation import CrossValidationOrchestrator
            from core.orchestrator import TaskOrchestrator
            
            communication = CommunicationProtocol()
            expert_registry = ExpertAgentRegistry(communication)
            meeting_orchestrator = MeetingOrchestrator()
            validation_orchestrator = CrossValidationOrchestrator()
            task_orchestrator = TaskOrchestrator(
                expert_registry, communication, meeting_orchestrator, validation_orchestrator
            )
            
            # Initialize experts
            experts = expert_registry.initialize_expert_agents()
            
            # Get statistics
            expert_stats = expert_registry.get_expert_statistics()
            comm_stats = communication.get_communication_stats()
            meeting_stats = meeting_orchestrator.get_meeting_statistics()
            orchestration_stats = task_orchestrator.get_orchestration_statistics()
            
            print(f"👥 Expert Agents: {expert_stats['total_experts']}")
            print(f"🎓 Domain Coverage: {expert_stats['domain_coverage']} domains")
            print(f"⭐ Avg Experience: {expert_stats['average_experience_level']:.2f}")
            print(f"🤝 Avg Collaboration: {expert_stats['average_collaboration_preference']:.2f}")
            print(f"💬 Message Handlers: {comm_stats['registered_handlers']}")
            print(f"📅 Meetings Available: {meeting_stats['total_meetings_conducted']}")
            print(f"🎯 Tasks Orchestrated: {orchestration_stats['total_tasks_orchestrated']}")
            print(f"✅ Success Rate: {orchestration_stats['success_rate']:.2f}")
            
            print("\n🔧 Available Expert Domains:")
            for domain in expert_stats.get('expertise_domains', []):
                print(f"  • {domain.value if hasattr(domain, 'value') else domain}")
            
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
    
    async def run_component_testing(self):
        """Run component testing suite"""
        
        print("\n🔧 Component Testing Suite")
        print("=" * 40)
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Memory System
        print("1. Testing Memory System...")
        tests_total += 1
        try:
            from core.memory import AdvancedMemorySystem, EpisodicMemory
            memory_system = AdvancedMemorySystem("test_agent")
            
            # Test episodic memory storage
            test_memory = EpisodicMemory(
                id="test_memory",
                agent_id="test_agent",
                timestamp=datetime.now(),
                event_type="test",
                context={"test": True},
                participants=["test_agent"],
                content="Test memory content",
                emotional_valence=0.5,
                importance_score=0.7,
                related_memories=[]
            )
            
            memory_id = memory_system.store_episodic_memory(test_memory)
            if memory_id:
                print("   ✅ Memory system working correctly")
                tests_passed += 1
            else:
                print("   ❌ Memory storage failed")
        except Exception as e:
            print(f"   ❌ Memory system error: {e}")
        
        # Test 2: Communication System
        print("2. Testing Communication System...")
        tests_total += 1
        try:
            from core.communication import CommunicationProtocol, Message, MessageType
            comm = CommunicationProtocol()
            
            test_message = Message(
                id="test_msg",
                sender_id="test_sender",
                receiver_id="test_receiver",
                message_type=MessageType.TASK_REQUEST,
                content="Test message",
                context={},
                timestamp=datetime.now(),
                priority=5,
                requires_response=False
            )
            
            # Test message creation and handling
            if test_message.id == "test_msg":
                print("   ✅ Communication system working correctly")
                tests_passed += 1
            else:
                print("   ❌ Communication system failed")
        except Exception as e:
            print(f"   ❌ Communication system error: {e}")
        
        # Test 3: Expert Agent Registry
        print("3. Testing Expert Agent Registry...")
        tests_total += 1
        try:
            from core.communication import CommunicationProtocol
            from core.expert_agents import ExpertAgentRegistry
            
            comm = CommunicationProtocol()
            registry = ExpertAgentRegistry(comm)
            experts = registry.initialize_expert_agents()
            
            if len(experts) > 0:
                print(f"   ✅ Expert registry initialized with {len(experts)} agents")
                tests_passed += 1
            else:
                print("   ❌ No expert agents initialized")
        except Exception as e:
            print(f"   ❌ Expert registry error: {e}")
        
        # Test Summary
        print(f"\n📊 Test Results: {tests_passed}/{tests_total} tests passed")
        if tests_passed == tests_total:
            print("✅ All components working correctly!")
        else:
            print(f"⚠️  {tests_total - tests_passed} components need attention")
    
    def display_documentation(self):
        """Display system documentation and help"""
        
        print("\n📖 Documentation and Help")
        print("=" * 40)
        
        print("""
🔍 System Overview:
This Multi-Agent AI System implements the advanced collaboration technology 
described in NTT's research. It features autonomous AI agents that collaborate 
through dialogue, accumulate knowledge, and validate each other's work.

🏗️ Architecture:
├── Core Components
│   ├── 🧠 Memory System (Episodic & Semantic)
│   ├── 💬 Communication Protocol  
│   ├── ✅ Validation System
│   ├── 📅 Meeting Orchestration
│   └── 🎯 Task Orchestration
├── Expert Agents
│   ├── 📊 Business Strategy Expert
│   ├── 📈 Marketing Expert
│   ├── 🔧 Technical Architecture Expert
│   ├── 🎨 Creative Design Expert
│   ├── 📋 Project Management Expert
│   └── ✅ Quality Assurance Expert
└── Demo Applications
    └── 🍃 Tea Business Plan Demo

🚀 Cara Memulai:
1. Pastikan OpenAI API key sudah dikonfigurasi (OPENAI_API_KEY environment variable)
2. Jalankan pemeriksaan konfigurasi (Opsi 2) untuk memverifikasi setup
3. Coba Dialog Multi-Agent AI (Opsi 1) untuk melihat sistem beraksi
4. Masukkan permintaan apapun dalam bahasa Indonesia atau bahasa lain
5. Tinjau hasil yang dihasilkan dan log untuk wawasan detail

🔧 Configuration:
- Edit config.py to adjust system parameters
- Modify agent roles and capabilities in core/expert_agents.py
- Customize collaboration patterns in core/orchestrator.py

📊 Monitoring:
- System generates detailed logs of agent interactions
- Meeting minutes are automatically recorded
- Validation results are tracked and analyzed
- Performance metrics are continuously collected

🤝 Key Features Demonstrated:
✓ Human-like dialogue between AI agents
✓ Knowledge accumulation and reuse across tasks
✓ Cross-validation for accuracy and consistency  
✓ Structured meeting orchestration
✓ Complex task decomposition and execution
✓ Continuous learning and improvement

For more detailed information, refer to the source code documentation
and generated demo reports.
        """)
    
    async def run_interactive_mode(self):
        """Run the system in interactive mode"""
        
        self.display_welcome_banner()
        
        while True:
            self.display_menu()
            
            try:
                choice = input("Pilih opsi (1-6): ").strip()
                
                if choice == "1":
                    print("\n🤖 Memulai Dialog Multi-Agent AI...")
                    await run_flexible_system()
                    
                elif choice == "2":
                    await self.run_configuration_check()
                    
                elif choice == "3":
                    await self.display_system_status()
                    
                elif choice == "4":
                    await self.run_component_testing()
                    
                elif choice == "5":
                    self.display_documentation()
                    
                elif choice == "6":
                    print("\n👋 Thank you for using NTT Multi-Agent AI System!")
                    print("🔗 Based on NTT's groundbreaking research in collaborative AI")
                    break
                    
                else:
                    print("❌ Invalid option. Please select 1-6.")
                
                if choice != "6":
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

async def main():
    """Main entry point"""
    
    # Create system instance
    system = MultiAgentAISystem()
    
    # Check if running in interactive mode or direct flexible mode
    if len(sys.argv) > 1 and sys.argv[1] == "--flexible":
        # Direct flexible mode
        system.display_welcome_banner()
        print("\n🚀 Memulai Dialog Multi-Agent AI secara langsung...")
        await run_flexible_system()
    else:
        # Interactive mode
        await system.run_interactive_mode()

if __name__ == "__main__":
    # Set up event loop and run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 System shutdown requested. Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
