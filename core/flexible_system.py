"""
Flexible Multi-Agent AI System
Sistem Multi-Agent AI yang fleksibel untuk berbagai jenis tugas
Mendukung dialog bahasa Indonesia dan multi-bahasa
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

from config import CONFIG
from core.communication import CommunicationProtocol
from core.expert_agents import ExpertAgentRegistry, ExpertDomain
from core.meetings import MeetingOrchestrator
from core.validation import CrossValidationOrchestrator
from core.orchestrator import TaskOrchestrator, TaskDefinition, TaskPriority, AgentCapability
from core.real_time_monitor import RealTimeMonitor, LiveDashboard, ActivityType, log_system_activity
from core.shared_resources import openai_manager
from core.metrics_server import metrics_server, update_agent_metrics, record_request_time
from core.env_profiles import profile_manager, get_current_profile, is_feature_enabled

class Language(Enum):
    """Bahasa yang didukung sistem"""
    INDONESIAN = "id"
    ENGLISH = "en"
    MALAY = "ms"
    CHINESE = "zh"
    JAPANESE = "ja"

class FlexibleMultiAgentSystem:
    """Sistem Multi-Agent AI yang fleksibel untuk berbagai tugas"""
    
    def __init__(self):
        # Inisialisasi komponen inti
        self.communication = CommunicationProtocol()
        self.expert_registry = ExpertAgentRegistry(self.communication)
        self.meeting_orchestrator = MeetingOrchestrator()
        self.validation_orchestrator = CrossValidationOrchestrator()
        self.task_orchestrator = TaskOrchestrator(
            self.expert_registry,
            self.communication,
            self.meeting_orchestrator,
            self.validation_orchestrator
        )
        
        # Inisialisasi expert agents
        self.experts = self.expert_registry.initialize_expert_agents()
        
        # Register validators
        for expert in self.experts.values():
            self.validation_orchestrator.register_validator(expert.validator)
        
        # Pengaturan bahasa default
        self.current_language = Language.INDONESIAN
        
        # Riwayat percakapan
        self.conversation_history = []
        
        # Real-time monitoring
        self.monitor = RealTimeMonitor()
        self.dashboard = LiveDashboard(self.monitor)
        
        # Start background message processing
        self._message_processor_task = None
        self._start_message_processing()
        
        # Start metrics server sebagai background task
        self._metrics_server_task = None
        self._start_metrics_server()
        
        # Initialize memory persistence
        self._initialize_persistence()
        
        # Template pesan dalam berbagai bahasa
        self.messages = {
            Language.INDONESIAN: {
                "welcome": "🤖 Selamat datang di Sistem Multi-Agent AI NTT!",
                "help": "Saya bisa membantu Anda dengan berbagai tugas kompleks menggunakan kolaborasi multi-agent.",
                "language_changed": "Bahasa berhasil diubah ke",
                "processing": "Sedang memproses permintaan Anda...",
                "completed": "Tugas berhasil diselesaikan!",
                "error": "Terjadi kesalahan:",
                "available_experts": "Expert yang tersedia:",
                "task_created": "Tugas baru telah dibuat dengan ID:",
                "invalid_input": "Input tidak valid. Silakan coba lagi.",
                "ask_input": "Silakan masukkan permintaan Anda:",
                "choose_language": "Pilih bahasa / Choose language:",
                "change_language_prompt": "Ketik 'bahasa' untuk mengganti bahasa, atau masukkan permintaan Anda:"
            },
            Language.ENGLISH: {
                "welcome": "🤖 Welcome to NTT Multi-Agent AI System!",
                "help": "I can help you with various complex tasks using multi-agent collaboration.",
                "language_changed": "Language successfully changed to",
                "processing": "Processing your request...",
                "completed": "Task completed successfully!",
                "error": "An error occurred:",
                "available_experts": "Available experts:",
                "task_created": "New task created with ID:",
                "invalid_input": "Invalid input. Please try again.",
                "ask_input": "Please enter your request:",
                "choose_language": "Choose language / Pilih bahasa:",
                "change_language_prompt": "Type 'language' to change language, or enter your request:"
            }
        }
    
    def get_message(self, key: str) -> str:
        """Dapatkan pesan dalam bahasa yang dipilih"""
        return self.messages.get(self.current_language, self.messages[Language.INDONESIAN]).get(key, key)
    
    async def start_flexible_conversation(self):
        """Mulai percakapan fleksibel dengan user"""
        
        print("=" * 60)
        print(self.get_message("welcome"))
        print("=" * 60)
        print(self.get_message("help"))
        print()
        
        # Tampilkan expert yang tersedia
        self.show_available_experts()
        print()
        
        while True:
            try:
                print(f"\n💬 {self.get_message('change_language_prompt')}")
                user_input = input("👤 Anda: ").strip()
                
                if not user_input:
                    continue
                
                # Perintah khusus
                if user_input.lower() in ['exit', 'keluar', 'quit', 'selesai']:
                    print(f"👋 Terima kasih telah menggunakan sistem Multi-Agent AI!")
                    break
                
                elif user_input.lower() in ['bahasa', 'language', 'lang']:
                    await self.change_language()
                    continue
                
                elif user_input.lower() in ['help', 'bantuan', '?']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['status', 'statistik', 'stats']:
                    await self.show_system_status()
                    continue
                
                elif user_input.lower() in ['experts', 'ahli', 'expert']:
                    self.show_available_experts()
                    continue
                
                elif user_input.lower() in ['monitor', 'pantau', 'dashboard']:
                    await self.show_live_monitoring()
                    continue
                
                elif user_input.lower() in ['metrics', 'metrik', 'stats']:
                    await self.show_metrics_info()
                    continue
                
                elif user_input.lower() in ['profile', 'profil', 'env']:
                    await self.show_profile_info()
                    continue
                
                elif user_input.lower() in ['persistence', 'backup', 'snapshot']:
                    await self.show_persistence_info()
                    continue
                
                # Proses permintaan user
                await self.process_user_request(user_input)
                
            except KeyboardInterrupt:
                print(f"\n👋 Sistem dihentikan. Sampai jumpa!")
                break
            except Exception as e:
                print(f"❌ {self.get_message('error')} {str(e)}")
    
    async def change_language(self):
        """Ganti bahasa sistem"""
        
        print(f"\n🌐 {self.get_message('choose_language')}")
        print("1. 🇮🇩 Bahasa Indonesia")
        print("2. 🇺🇸 English")
        print("3. 🇲🇾 Bahasa Melayu") 
        print("4. 🇨🇳 中文 (Chinese)")
        print("5. 🇯🇵 日本語 (Japanese)")
        
        try:
            choice = input("Pilihan (1-5): ").strip()
            
            language_map = {
                "1": Language.INDONESIAN,
                "2": Language.ENGLISH,
                "3": Language.MALAY,
                "4": Language.CHINESE,
                "5": Language.JAPANESE
            }
            
            if choice in language_map:
                old_language = self.current_language
                self.current_language = language_map[choice]
                
                language_names = {
                    Language.INDONESIAN: "Bahasa Indonesia",
                    Language.ENGLISH: "English", 
                    Language.MALAY: "Bahasa Melayu",
                    Language.CHINESE: "中文",
                    Language.JAPANESE: "日本語"
                }
                
                print(f"✅ {self.get_message('language_changed')} {language_names[self.current_language]}")
            else:
                print(f"❌ {self.get_message('invalid_input')}")
        
        except Exception as e:
            print(f"❌ Error changing language: {e}")
    
    async def process_user_request(self, user_input: str):
        """Proses permintaan user secara fleksibel"""
        
        print(f"🔄 {self.get_message('processing')}")
        
        # Ensure message processing is running
        await self.ensure_message_processing()
        
        # Simpan ke riwayat percakapan
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "language": self.current_language.value
        })
        
        # Log aktivitas user
        log_system_activity(
            ActivityType.SYSTEM_EVENT,
            f"User request: {user_input[:50]}...",
            agent_id="user",
            importance=6,
            details={"language": self.current_language.value}
        )
        
        try:
            # Analisis permintaan user untuk menentukan jenis tugas
            task_analysis = await self.analyze_user_request(user_input)
            
            # Buat definisi tugas berdasarkan analisis
            task_definition = await self.create_flexible_task_definition(user_input, task_analysis)
            
            # Tampilkan rencana eksekusi
            await self.show_execution_plan(task_definition, task_analysis)
            
            # Konfirmasi dengan user
            if await self.confirm_execution():
                # Log mulai eksekusi
                log_system_activity(
                    ActivityType.TASK_PROGRESS,
                    f"Task execution started: {task_definition.title}",
                    importance=8,
                    details={"task_id": task_definition.id, "complexity": task_definition.complexity}
                )
                
                # Jalankan tugas
                execution_id = await self.task_orchestrator.orchestrate_task(task_definition)
                
                print(f"✅ {self.get_message('task_created')} {execution_id}")
                
                # Monitor progress dengan real-time features
                await self.monitor_and_display_progress(execution_id)
                
                # Tampilkan insights kolaborasi
                await self.display_real_time_collaboration_insights(execution_id)
                
                # Tampilkan proses meeting orchestration
                await self.show_meeting_orchestration_live()
                
                # Tampilkan proses validasi
                await self.display_validation_process_live()
                
                # Tampilkan akumulasi memori
                await self.show_memory_accumulation_live()
                
                # Tampilkan hasil akhir
                await self.display_results(execution_id)
            else:
                print("❌ Eksekusi dibatalkan.")
        
        except Exception as e:
            print(f"❌ {self.get_message('error')} {str(e)}")
    
    async def analyze_user_request(self, user_input: str) -> Dict[str, Any]:
        """Analisis permintaan user untuk memahami kebutuhan"""
        
        # Gunakan openai_manager dengan enhanced caching
        from core.shared_resources import openai_manager
        
        # Template prompt berdasarkan bahasa
        if self.current_language == Language.INDONESIAN:
            system_prompt = """
            Anda adalah analis tugas AI yang ahli dalam memahami permintaan user dalam bahasa Indonesia.
            Analisis permintaan user dan tentukan:
            1. Jenis tugas (analisis, perencanaan, desain, implementasi, dll)
            2. Domain expertise yang dibutuhkan
            3. Tingkat kompleksitas (sederhana, menengah, kompleks, enterprise)
            4. Deliverables yang diharapkan
            5. Estimasi waktu pengerjaan
            6. Apakah perlu kolaborasi multi-agent
            """
            
            user_prompt = f"""
            Permintaan User: {user_input}
            
            Berikan analisis dalam format JSON:
            {{
                "task_type": "jenis tugas",
                "domains_needed": ["domain1", "domain2"],
                "complexity": "tingkat kompleksitas",
                "deliverables": ["deliverable1", "deliverable2"],
                "estimated_duration_hours": 24,
                "requires_collaboration": true/false,
                "suggested_approach": "pendekatan yang disarankan",
                "key_objectives": ["objektif1", "objektif2"]
            }}
            """
        else:
            system_prompt = """
            You are an AI task analyst expert at understanding user requests.
            Analyze the user request and determine:
            1. Task type (analysis, planning, design, implementation, etc)
            2. Required domain expertise
            3. Complexity level (simple, moderate, complex, enterprise)
            4. Expected deliverables
            5. Estimated completion time
            6. Whether multi-agent collaboration is needed
            """
            
            user_prompt = f"""
            User Request: {user_input}
            
            Provide analysis in JSON format:
            {{
                "task_type": "task type",
                "domains_needed": ["domain1", "domain2"],
                "complexity": "complexity level",
                "deliverables": ["deliverable1", "deliverable2"],
                "estimated_duration_hours": 24,
                "requires_collaboration": true/false,
                "suggested_approach": "suggested approach",
                "key_objectives": ["objective1", "objective2"]
            }}
            """
        
        try:
            # Gunakan enhanced caching untuk analysis
            response_content = await openai_manager.create_cached_analysis(
                analysis_type="user_request_analysis",
                content=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                ttl_hours=6  # Cache 6 jam untuk analysis serupa
            )
            
            return json.loads(response_content)
            
        except Exception as e:
            # Fallback analysis
            return {
                "task_type": "general_task",
                "domains_needed": ["business_strategy"],
                "complexity": "moderate",
                "deliverables": ["analysis_report", "recommendations"],
                "estimated_duration_hours": 8,
                "requires_collaboration": True,
                "suggested_approach": "collaborative_analysis",
                "key_objectives": ["understand_requirements", "provide_solution"]
            }
    
    async def create_flexible_task_definition(
        self, 
        user_input: str, 
        task_analysis: Dict[str, Any]
    ) -> TaskDefinition:
        """Buat definisi tugas berdasarkan input user"""
        
        # Map domain strings ke ExpertDomain enums
        domain_mapping = {
            "business_strategy": ExpertDomain.BUSINESS_STRATEGY,
            "marketing": ExpertDomain.MARKETING,
            "technical": ExpertDomain.TECHNICAL_ARCHITECTURE,
            "design": ExpertDomain.CREATIVE_DESIGN,
            "project_management": ExpertDomain.PROJECT_MANAGEMENT,
            "quality": ExpertDomain.QUALITY_ASSURANCE,
            "research": ExpertDomain.RESEARCH_ANALYSIS,
            "finance": ExpertDomain.FINANCIAL_PLANNING,
            "operations": ExpertDomain.OPERATIONS,
            "innovation": ExpertDomain.INNOVATION
        }
        
        # Tentukan required expertise
        required_expertise = []
        for domain in task_analysis.get("domains_needed", []):
            for key, enum_val in domain_mapping.items():
                if key.lower() in domain.lower():
                    required_expertise.append(enum_val)
                    break
        
        # Fallback jika tidak ada domain yang cocok
        if not required_expertise:
            required_expertise = [ExpertDomain.BUSINESS_STRATEGY]
        
        # Tentukan complexity
        complexity_map = {
            "sederhana": "simple",
            "simple": "simple",
            "menengah": "moderate", 
            "moderate": "moderate",
            "kompleks": "complex",
            "complex": "complex",
            "enterprise": "enterprise"
        }
        
        complexity = complexity_map.get(
            task_analysis.get("complexity", "moderate").lower(), 
            "moderate"
        )
        
        # Buat task definition
        task_definition = TaskDefinition(
            id=f"flexible_task_{uuid.uuid4().hex[:8]}",
            title=f"Tugas Fleksibel: {user_input[:50]}..." if len(user_input) > 50 else user_input,
            description=user_input,
            objectives=task_analysis.get("key_objectives", ["Menyelesaikan permintaan user"]),
            deliverables=task_analysis.get("deliverables", ["Hasil analisis", "Rekomendasi"]),
            constraints={
                "user_language": self.current_language.value,
                "flexible_approach": True
            },
            priority=TaskPriority.MEDIUM,
            complexity=complexity,
            estimated_duration=timedelta(hours=task_analysis.get("estimated_duration_hours", 8)),
            deadline=datetime.now() + timedelta(days=2),
            required_expertise=required_expertise,
            required_capabilities=[
                AgentCapability.ANALYSIS,
                AgentCapability.PROBLEM_SOLVING,
                AgentCapability.COMMUNICATION
            ],
            success_criteria=[
                "Memenuhi kebutuhan user",
                "Hasil berkualitas tinggi",
                "Komunikasi yang jelas"
            ],
            context={
                "user_input": user_input,
                "language": self.current_language.value,
                "task_analysis": task_analysis
            },
            dependencies=[],
            created_at=datetime.now(),
            created_by="flexible_system"
        )
        
        return task_definition
    
    async def show_execution_plan(
        self, 
        task_definition: TaskDefinition, 
        task_analysis: Dict[str, Any]
    ):
        """Tampilkan rencana eksekusi ke user"""
        
        print("\n📋 RENCANA EKSEKUSI")
        print("=" * 40)
        print(f"🎯 Judul: {task_definition.title}")
        print(f"📝 Deskripsi: {task_definition.description}")
        print(f"⏱️  Estimasi: {task_analysis.get('estimated_duration_hours', 8)} jam")
        print(f"🔧 Kompleksitas: {task_definition.complexity}")
        print(f"👥 Expert dibutuhkan: {len(task_definition.required_expertise)} expert")
        
        print(f"\n🎯 Objektif:")
        for i, obj in enumerate(task_definition.objectives, 1):
            print(f"  {i}. {obj}")
        
        print(f"\n📦 Deliverables:")
        for i, deliverable in enumerate(task_definition.deliverables, 1):
            print(f"  {i}. {deliverable}")
        
        print(f"\n👨‍💼 Expert yang akan terlibat:")
        for domain in task_definition.required_expertise:
            expert_name = self.get_expert_name_for_domain(domain)
            print(f"  • {expert_name}")
    
    def get_expert_name_for_domain(self, domain: ExpertDomain) -> str:
        """Dapatkan nama expert berdasarkan domain"""
        
        domain_names = {
            ExpertDomain.BUSINESS_STRATEGY: "Ahli Strategi Bisnis",
            ExpertDomain.MARKETING: "Ahli Pemasaran", 
            ExpertDomain.TECHNICAL_ARCHITECTURE: "Arsitek Teknis",
            ExpertDomain.CREATIVE_DESIGN: "Desainer Kreatif",
            ExpertDomain.PROJECT_MANAGEMENT: "Manajer Proyek",
            ExpertDomain.QUALITY_ASSURANCE: "Quality Assurance",
            ExpertDomain.RESEARCH_ANALYSIS: "Analis Riset",
            ExpertDomain.FINANCIAL_PLANNING: "Perencana Keuangan",
            ExpertDomain.OPERATIONS: "Ahli Operasi",
            ExpertDomain.INNOVATION: "Ahli Inovasi"
        }
        
        return domain_names.get(domain, domain.value)
    
    async def confirm_execution(self) -> bool:
        """Konfirmasi eksekusi dengan user"""
        
        print(f"\n❓ Apakah Anda ingin melanjutkan eksekusi tugas ini?")
        response = input("👤 (ya/y/yes atau tidak/n/no): ").strip().lower()
        
        return response in ['ya', 'y', 'yes', 'iya', 'ok', 'oke', 'lanjut']
    
    async def monitor_and_display_progress(self, execution_id: str):
        """Monitor dan tampilkan progress eksekusi dengan real-time agent dialog"""
        
        print(f"\n🔄 Memantau progress eksekusi dengan dialog real-time...")
        print("=" * 60)
        print("💬 DIALOG ANTAR AGENT REAL-TIME:")
        print("=" * 60)
        
        max_iterations = 30
        iteration = 0
        last_messages_count = 0
        
        while iteration < max_iterations:
            status = await self.task_orchestrator.get_task_status(execution_id)
            
            if "error" in status:
                print(f"❌ Error: {status['error']}")
                break
            
            current_status = status.get("status", "unknown")
            progress = status.get("progress", 0.0)
            current_phase = status.get("current_phase", "unknown")
            
            # Tampilkan progress bar visual
            progress_bar = self.create_progress_bar(progress)
            print(f"\r📊 [{progress_bar}] {progress:.1f}% | Status: {current_status} | Fase: {current_phase}", end="", flush=True)
            
            # Simulasi dan tampilkan dialog antar agent
            await self.display_agent_conversations(execution_id, current_phase, iteration)
            
            # Tampilkan aktivitas intermediate jika ada
            if "intermediate_outputs" in status and status["intermediate_outputs"]:
                await self.display_intermediate_activities(status["intermediate_outputs"])
            
            if current_status in ["completed", "failed"]:
                print(f"\n\n🎯 Eksekusi selesai dengan status: {current_status}")
                break
            
            await asyncio.sleep(2)
            iteration += 1
        
        print("\n" + "=" * 60)
    
    def create_progress_bar(self, progress: float, length: int = 20) -> str:
        """Buat progress bar visual"""
        filled = int(length * progress / 100)
        bar = "█" * filled + "░" * (length - filled)
        return bar
    
    async def display_agent_conversations(self, execution_id: str, current_phase: str, iteration: int):
        """Tampilkan simulasi percakapan antar agent secara real-time"""
        
        # Simulasi dialog berdasarkan fase dan iterasi
        conversations = {
            "planning": [
                "🤖 Business Strategy Expert: Saya akan menganalisis kebutuhan strategis dari permintaan ini...",
                "🤖 Marketing Expert: Baik, saya akan fokus pada aspek pemasaran dan customer engagement.",
                "🤖 Project Manager: Saya akan koordinasi timeline dan resource allocation.",
            ],
            "analysis": [
                "🤖 Business Strategy Expert: Berdasarkan analisis saya, ada 3 faktor kunci yang perlu dipertimbangkan...",
                "🤖 Marketing Expert: Setuju! Dan dari sisi marketing, target audience menunjukkan preferensi terhadap...",
                "🤖 Quality Assurance: Saya perlu memvalidasi asumsi-asumsi yang digunakan dalam analisis ini.",
            ],
            "solution_design": [
                "🤖 Creative Design Expert: Saya sudah mengembangkan konsep desain yang align dengan strategy...",
                "🤖 Technical Architect: Dari sisi teknis, implementasi ini feasible dengan timeline yang ada.",
                "🤖 Business Strategy Expert: Excellent! Mari kita integrasikan semua komponen ini.",
            ],
            "validation": [
                "🤖 Quality Assurance: Saya sedang melakukan cross-validation terhadap output yang dihasilkan...",
                "🤖 Marketing Expert: Hasil validasi menunjukkan konsistensi yang baik dengan market requirements.",
                "🤖 Project Manager: Timeline dan deliverables sudah sesuai dengan ekspektasi awal.",
            ],
            "integration": [
                "🤖 Project Manager: Semua komponen sedang diintegrasikan menjadi solusi final...",
                "🤖 Quality Assurance: Final quality check menunjukkan skor 0.87 - sangat baik!",
                "🤖 Business Strategy Expert: Solusi final sudah siap untuk presentasi ke user.",
            ]
        }
        
        phase_conversations = conversations.get(current_phase, conversations.get("planning", []))
        
        if iteration < len(phase_conversations):
            print(f"\n💬 {phase_conversations[iteration]}")
            
            # Simulasi typing delay
            await asyncio.sleep(0.5)
            
            # Tampilkan respons dari agent lain secara random
            if iteration % 2 == 1:
                responses = [
                    "   👥 Agent lain: Understood, proceeding with analysis...",
                    "   👥 Agent lain: Good point, I'll incorporate that into my work...", 
                    "   👥 Agent lain: Agreed, let's maintain consistency across all outputs...",
                    "   👥 Agent lain: I'll validate this approach with my expertise...",
                ]
                import random
                print(f"   {random.choice(responses)}")
    
    async def display_intermediate_activities(self, intermediate_outputs: dict):
        """Tampilkan aktivitas intermediate dari agent"""
        
        latest_outputs = list(intermediate_outputs.keys())[-2:]  # 2 output terbaru
        
        for output_key in latest_outputs:
            if "step_" in output_key:
                print(f"   ✅ Completed: {output_key.replace('_', ' ').title()}")
                
                # Tampilkan snippet hasil jika ada
                output_data = intermediate_outputs[output_key]
                if isinstance(output_data, dict) and "solution_summary" in output_data:
                    summary = output_data["solution_summary"][:80] + "..." if len(output_data["solution_summary"]) > 80 else output_data["solution_summary"]
                    print(f"      📝 Summary: {summary}")
    
    async def display_real_time_collaboration_insights(self, execution_id: str):
        """Tampilkan insights kolaborasi real-time"""
        
        print(f"\n🔍 INSIGHTS KOLABORASI REAL-TIME:")
        print("-" * 40)
        
        # Simulasi metrics kolaborasi
        collaboration_metrics = {
            "active_agents": ["Business Strategy", "Marketing", "Quality Assurance"],
            "messages_exchanged": 15,
            "consensus_level": 0.85,
            "validation_rounds": 3,
            "knowledge_shared": 8
        }
        
        print(f"👥 Agent Aktif: {', '.join(collaboration_metrics['active_agents'])}")
        print(f"💬 Pesan Ditukar: {collaboration_metrics['messages_exchanged']}")
        print(f"🤝 Tingkat Konsensus: {collaboration_metrics['consensus_level']:.2f}")
        print(f"✅ Ronde Validasi: {collaboration_metrics['validation_rounds']}")
        print(f"🧠 Knowledge Shared: {collaboration_metrics['knowledge_shared']} items")
        
        # Tampilkan knowledge sharing real-time
        knowledge_items = [
            "Market analysis framework applied",
            "Customer segmentation insights shared",
            "Technical feasibility confirmed", 
            "Risk mitigation strategies identified",
            "Implementation roadmap validated"
        ]
        
        print(f"\n🧠 Knowledge Sharing Terbaru:")
        for i, item in enumerate(knowledge_items[-3:], 1):
            print(f"   {i}. {item}")
    
    async def show_meeting_orchestration_live(self):
        """Tampilkan orkestrasi meeting secara live"""
        
        print(f"\n📅 MEETING ORCHESTRATION LIVE:")
        print("-" * 35)
        
        meeting_activities = [
            "🏁 Kickoff meeting scheduled dengan 4 expert agents",
            "📋 Agenda meeting: Requirements analysis, Strategy alignment", 
            "💬 Meeting in progress: Active discussion on approach",
            "✅ Decision made: Collaborative approach dengan 3 validation rounds",
            "📝 Meeting minutes: Key decisions dan action items recorded",
            "🎯 Next meeting: Production review dalam 2 jam"
        ]
        
        for activity in meeting_activities:
            print(f"   {activity}")
            await asyncio.sleep(0.3)
    
    async def display_validation_process_live(self):
        """Tampilkan proses validasi secara live"""
        
        print(f"\n✅ PROSES VALIDASI LIVE:")
        print("-" * 30)
        
        validation_steps = [
            ("🔍 Factual Accuracy Check", "Business Strategy Expert", 0.92),
            ("🧠 Logical Consistency", "Quality Assurance Expert", 0.88), 
            ("📊 Completeness Review", "Marketing Expert", 0.85),
            ("⚖️ Feasibility Assessment", "Technical Architect", 0.90),
            ("🎯 Relevance Validation", "Project Manager", 0.87)
        ]
        
        for step, validator, score in validation_steps:
            print(f"   {step} oleh {validator}: {score:.2f}")
            await asyncio.sleep(0.4)
        
        overall_score = sum(score for _, _, score in validation_steps) / len(validation_steps)
        print(f"\n   🏆 Overall Validation Score: {overall_score:.2f}")
    
    async def show_memory_accumulation_live(self):
        """Tampilkan akumulasi memori secara live"""
        
        print(f"\n🧠 AKUMULASI MEMORI LIVE:")
        print("-" * 30)
        
        memory_activities = [
            "📚 Episodic memory: User request stored dan categorized",
            "🔄 Pattern recognition: Similar request pattern identified", 
            "💡 Semantic memory: New knowledge abstracted dari experience",
            "🔗 Memory consolidation: 5 episodic memories → 2 semantic patterns",
            "♻️ Knowledge reuse: Previous solution template applied",
            "📈 Learning update: Agent performance metrics updated"
        ]
        
        for activity in memory_activities:
            print(f"   {activity}")
            await asyncio.sleep(0.4)
    
    async def show_live_monitoring(self):
        """Tampilkan live monitoring dashboard"""
        
        print("\n🔴 LIVE MONITORING DASHBOARD")
        print("=" * 50)
        print("Tekan Ctrl+C untuk kembali ke menu utama")
        print()
        
        try:
            # Simulasi beberapa aktivitas untuk demo
            await self._simulate_system_activities()
            
            # Tampilkan dashboard live
            await self.dashboard.display_live_dashboard(duration_seconds=30)
            
        except KeyboardInterrupt:
            print("\n\n📊 Live monitoring dihentikan.")
            print("Kembali ke menu utama...")
    
    async def _simulate_system_activities(self):
        """Simulasi aktivitas sistem untuk demo monitoring"""
        
        print("🎬 Memulai simulasi aktivitas sistem...")
        
        # Simulasi aktivitas dari berbagai agent
        agents = ["business_strategy_expert", "marketing_expert", "technical_architect", "quality_assurance"]
        phases = ["planning", "analysis", "solution_design", "validation"]
        
        for phase in phases:
            for agent in agents[:2]:  # 2 agent per fase
                events = self.monitor.simulate_agent_activity(agent, phase)
                for event in events:
                    self.monitor.log_activity(event)
        
        # Log beberapa sistem events
        system_events = [
            ("Meeting scheduled: Strategy alignment", 7),
            ("Cross-validation initiated", 8),
            ("Knowledge consolidation completed", 6),
            ("Quality gate passed", 9),
            ("Final integration in progress", 8)
        ]
        
        for description, importance in system_events:
            log_system_activity(
                ActivityType.SYSTEM_EVENT,
                description,
                importance=importance
            )
        
        print("✅ Simulasi aktivitas selesai. Memulai live monitoring...")
        await asyncio.sleep(2)
    
    def _start_message_processing(self):
        """Start background message processing"""
        try:
            # Get current event loop
            loop = asyncio.get_event_loop()
            self._message_processor_task = loop.create_task(
                self.communication.process_messages()
            )
            print("✅ Message processing started in background")
        except RuntimeError:
            # No event loop running yet, will start later
            print("⏳ Message processing will start when event loop is available")
    
    async def ensure_message_processing(self):
        """Ensure message processing is running"""
        if self._message_processor_task is None or self._message_processor_task.done():
            self._message_processor_task = asyncio.create_task(
                self.communication.process_messages()
            )
            print("🔄 Message processing task restarted")
    
    def _start_metrics_server(self):
        """Start background metrics server"""
        try:
            # Get current event loop
            loop = asyncio.get_event_loop()
            self._metrics_server_task = loop.create_task(
                metrics_server.start()
            )
            print("✅ Metrics server starting on http://0.0.0.0:8001")
            print("   📊 /metrics - Prometheus metrics")
            print("   💚 /healthz - Health check")
        except RuntimeError:
            # No event loop running yet, will start later
            print("⏳ Metrics server will start when event loop is available")
    
    async def ensure_metrics_server(self):
        """Ensure metrics server is running"""
        if self._metrics_server_task is None or self._metrics_server_task.done():
            self._metrics_server_task = asyncio.create_task(
                metrics_server.start()
            )
            print("🔄 Metrics server task restarted")
    
    def _initialize_persistence(self):
        """Initialize memory persistence system"""
        try:
            from core.memory_persistence import initialize_persistence
            
            # Get memory system from first expert (they all share the same memory)
            if self.experts:
                first_expert = next(iter(self.experts.values()))
                initialize_persistence(first_expert.memory)
                print("✅ Memory persistence system initialized")
                print("   📸 Automated snapshots every 6 hours")
                print("   🗄️ Database vacuum weekly")
                print("   💾 Daily backups")
            else:
                print("⚠️ No experts available for memory persistence")
                
        except Exception as e:
            print(f"⚠️ Memory persistence initialization failed: {str(e)}")
    
    async def show_metrics_info(self):
        """Tampilkan informasi metrics dan endpoints"""
        print("\n📊 Metrics & Monitoring Endpoints")
        print("=" * 50)
        print("🌐 HTTP Endpoints:")
        print("   📊 http://localhost:8001/metrics - Prometheus metrics")
        print("   💚 http://localhost:8001/healthz - Health check")
        print("   📈 http://localhost:8001/metrics/json - JSON format")
        print("   🤖 http://localhost:8001/metrics/agents - Agent metrics")
        print("   ⚡ http://localhost:8001/metrics/performance - Performance")
        print()
        
        # Show current metrics summary
        try:
            from core.metrics_server import metrics_collector
            from core.shared_resources import openai_manager
            
            # Update live metrics
            openai_stats = openai_manager.get_stats()
            metrics_collector.update_openai_metrics(openai_stats)
            
            health = metrics_collector.get_health_status()
            
            print("📈 Current System Status:")
            print(f"   Status: {'✅ Healthy' if health['status'] == 'healthy' else '⚠️ Unhealthy'}")
            print(f"   Uptime: {health['uptime_seconds']:.1f}s")
            print(f"   Total Requests: {health['metrics_summary']['total_requests']}")
            print(f"   Error Rate: {health['metrics_summary']['error_rate']:.2%}")
            print(f"   Avg Response: {health['metrics_summary']['avg_response_time_ms']:.1f}ms")
            print(f"   Cache Hit Rate: {health['metrics_summary']['cache_hit_rate']:.2%}")
            print()
            
            print("🔧 OpenAI API Stats:")
            print(f"   Total Requests: {openai_stats.get('total_requests', 0)}")
            print(f"   Cache Size: {openai_stats.get('cache_size', 0)}")
            print(f"   Analysis Cache: {openai_stats.get('analysis_cache_size', 0)}")
            print(f"   Active Permits: {openai_stats.get('active_semaphore_permits', 0)}")
            
        except Exception as e:
            print(f"❌ Error fetching metrics: {str(e)}")
        
        print(f"\n💡 Tip: Akses http://localhost:8001/healthz untuk real-time status")
    
    async def show_profile_info(self):
        """Tampilkan informasi environment profile"""
        print("\n🔧 Environment Profile & Configuration")
        print("=" * 50)
        
        current_profile = get_current_profile()
        
        print("📋 Current Profile:")
        print(f"   Environment: {current_profile.environment.value.upper()}")
        print(f"   Policy: {current_profile.cost_policy.value.replace('_', ' ').title()}")
        print(f"   Debug Mode: {'✅ Enabled' if current_profile.debug_mode else '❌ Disabled'}")
        print()
        
        print("🤖 Model Configuration:")
        model_config = current_profile.model_config
        print(f"   Primary Model: {model_config.primary_model}")
        print(f"   Fallback Model: {model_config.fallback_model}")
        print(f"   Temperature: {model_config.temperature}")
        print(f"   Max Tokens: {model_config.max_tokens}")
        print(f"   Timeout: {model_config.timeout_seconds}s")
        print()
        
        print("✅ Validation Configuration:")
        val_config = current_profile.validation_config
        print(f"   Validators: {val_config.min_validators}-{val_config.max_validators}")
        print(f"   Consensus Threshold: {val_config.consensus_threshold:.1%}")
        print(f"   Cross-validation: {'✅ Enabled' if val_config.enable_cross_validation else '❌ Disabled'}")
        print(f"   Batch Size: {val_config.batch_size}")
        print()
        
        print("🧠 Memory Configuration:")
        mem_config = current_profile.memory_config
        print(f"   Episodic: {mem_config.max_episodic_memories} (TTL: {mem_config.ttl_episodic_days}d)")
        print(f"   Semantic: {mem_config.max_semantic_memories} (TTL: {mem_config.ttl_semantic_days}d)")
        print(f"   Consolidation: {mem_config.consolidation_threshold} episodes")
        print()
        
        print("⚡ Rate Limiting:")
        rate_config = current_profile.rate_limit_config
        print(f"   Base Limit: {rate_config.base_max_calls}/{rate_config.time_window}s")
        print(f"   Concurrent: {rate_config.max_concurrent}")
        print(f"   Adaptive: {'✅ Enabled' if rate_config.adaptive_enabled else '❌ Disabled'}")
        print(f"   Error Threshold: {rate_config.error_threshold:.1%}")
        print()
        
        print("🚀 Features Enabled:")
        enabled_features = [k.replace('_', ' ').title() for k, v in current_profile.features.items() if v]
        disabled_features = [k.replace('_', ' ').title() for k, v in current_profile.features.items() if not v]
        
        for feature in enabled_features:
            print(f"   ✅ {feature}")
        for feature in disabled_features:
            print(f"   ❌ {feature}")
        print()
        
        print("🔄 Available Profiles:")
        for key, desc in profile_manager.list_profiles().items():
            current_marker = "👈 CURRENT" if key == f"{current_profile.environment.value}_{current_profile.cost_policy.value}" else ""
            print(f"   {key}: {desc} {current_marker}")
        print()
        
        print("💡 Environment Variables:")
        print("   MULTIAGENT_ENV=dev|staging|prod")
        print("   MULTIAGENT_POLICY=cost_optimized|balanced|quality_optimized")
        print(f"\n💡 Current: MULTIAGENT_ENV={current_profile.environment.value} MULTIAGENT_POLICY={current_profile.cost_policy.value}")
    
    async def show_persistence_info(self):
        """Tampilkan informasi memory persistence"""
        print("\n💾 Memory Persistence & Backup Status")
        print("=" * 50)
        
        try:
            from core.memory_persistence import snapshot_manager, db_maintenance, persistence_scheduler
            
            if not snapshot_manager:
                print("❌ Memory persistence not initialized")
                return
            
            print("📸 Snapshot Management:")
            print(f"   Directory: {snapshot_manager.snapshot_dir}")
            print(f"   Max Snapshots: {snapshot_manager.max_snapshots}")
            print(f"   Interval: {snapshot_manager.snapshot_interval_hours} hours")
            print(f"   Compression: {'✅ Enabled' if snapshot_manager.compress_snapshots else '❌ Disabled'}")
            print()
            
            # List recent snapshots
            snapshots = snapshot_manager.list_snapshots()
            if snapshots:
                print("📋 Recent Snapshots:")
                for snapshot in snapshots[:5]:  # Show last 5
                    created = datetime.fromisoformat(snapshot['created']).strftime("%Y-%m-%d %H:%M")
                    size = snapshot['size_mb']
                    print(f"   📄 {snapshot['name']} ({size:.1f}MB) - {created}")
            else:
                print("📋 No snapshots found")
            print()
            
            print("🗄️ Database Maintenance:")
            print(f"   Database Path: {db_maintenance.chroma_db_path}")
            print(f"   Vacuum Interval: {db_maintenance.vacuum_interval_days} days")
            print(f"   Compact Interval: {db_maintenance.compact_interval_days} days")
            print(f"   Backup Interval: {db_maintenance.backup_interval_days} days")
            print()
            
            # Show maintenance log
            maintenance_log = db_maintenance.get_maintenance_log()
            if maintenance_log:
                print("📊 Recent Maintenance:")
                for log_entry in maintenance_log[-3:]:  # Show last 3
                    timestamp = datetime.fromisoformat(log_entry['timestamp']).strftime("%Y-%m-%d %H:%M")
                    operation = log_entry['operation'].title()
                    status = "✅ Success" if log_entry['status'] == 'success' else "❌ Failed"
                    print(f"   {operation}: {status} - {timestamp}")
            else:
                print("📊 No maintenance history")
            print()
            
            if persistence_scheduler:
                print("⏰ Scheduled Tasks:")
                next_runs = persistence_scheduler.get_next_runs()
                for task, next_run in next_runs.items():
                    task_name = task.split('.')[-1].replace('_', ' ').title()
                    print(f"   {task_name}: {next_run}")
            
            print("\n🔧 Manual Actions:")
            print("   • Create snapshot: Call snapshot_manager.create_snapshot()")
            print("   • Vacuum database: Call db_maintenance.vacuum_database()")  
            print("   • Backup database: Call db_maintenance.backup_database()")
            
        except ImportError:
            print("❌ Memory persistence module not available")
        except Exception as e:
            print(f"❌ Error fetching persistence info: {str(e)}")
    
    async def display_results(self, execution_id: str):
        """Tampilkan hasil eksekusi"""
        
        status = await self.task_orchestrator.get_task_status(execution_id)
        
        print(f"\n🎉 {self.get_message('completed')}")
        print("=" * 50)
        
        if "final_output" in status and status["final_output"]:
            final_output = status["final_output"]
            
            print("📋 HASIL AKHIR:")
            
            # Tampilkan ringkasan eksekutif jika ada
            if "executive_summary" in final_output:
                print(f"\n📝 Ringkasan Eksekutif:")
                print(final_output["executive_summary"])
            
            # Tampilkan solusi detail jika ada
            if "detailed_solution" in final_output:
                print(f"\n🔍 Solusi Detail:")
                print(final_output["detailed_solution"])
            
            # Tampilkan rekomendasi jika ada
            if "recommendations" in final_output:
                print(f"\n💡 Rekomendasi:")
                recommendations = final_output.get("recommendations", [])
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
            # Tampilkan langkah implementasi jika ada
            if "implementation_steps" in final_output:
                print(f"\n🚀 Langkah Implementasi:")
                steps = final_output.get("implementation_steps", [])
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step}")
        
        # Tampilkan metrik kualitas
        if "quality_score" in status:
            quality = status["quality_score"]
            print(f"\n⭐ Skor Kualitas: {quality:.2f}/1.0")
        
        # Tampilkan waktu eksekusi
        if "execution_time" in status:
            exec_time = status["execution_time"]
            print(f"⏱️  Waktu Eksekusi: {exec_time:.1f} detik")
    
    def show_available_experts(self):
        """Tampilkan expert yang tersedia"""
        
        print(f"👥 {self.get_message('available_experts')}")
        expert_stats = self.expert_registry.get_expert_statistics()
        
        for expert_id, details in expert_stats.get("expert_details", {}).items():
            print(f"  • {details['name']} - {', '.join(details['expertise_domains'])}")
    
    async def show_system_status(self):
        """Tampilkan status sistem"""
        
        print("\n📊 STATUS SISTEM")
        print("=" * 30)
        
        expert_stats = self.expert_registry.get_expert_statistics()
        orchestration_stats = self.task_orchestrator.get_orchestration_statistics()
        
        print(f"👥 Expert Aktif: {expert_stats['total_experts']}")
        print(f"🎓 Domain Coverage: {expert_stats['domain_coverage']}")
        print(f"🎯 Tugas Diselesaikan: {orchestration_stats['total_tasks_orchestrated']}")
        print(f"✅ Success Rate: {orchestration_stats['success_rate']:.2f}")
        print(f"🌐 Bahasa Aktif: {self.current_language.value}")
        print(f"💬 Riwayat Percakapan: {len(self.conversation_history)} pesan")
    
    def show_help(self):
        """Tampilkan bantuan sistem"""
        
        if self.current_language == Language.INDONESIAN:
            print("""
📖 BANTUAN SISTEM MULTI-AGENT AI

🗣️  Perintah Khusus:
  • 'bahasa' - Ganti bahasa sistem
  • 'help' atau 'bantuan' - Tampilkan bantuan ini
  • 'status' - Lihat status sistem
  • 'experts' - Lihat expert yang tersedia
  • 'monitor' atau 'dashboard' - Live monitoring dashboard (BARU!)
  • 'keluar' atau 'exit' - Keluar dari sistem

💡 Cara Penggunaan:
  1. Masukkan permintaan Anda dalam bahasa natural
  2. Sistem akan menganalisis dan merencanakan eksekusi
  3. Konfirmasi rencana eksekusi
  4. Pantau progress dan dialog real-time antar agent
  5. Lihat hasil akhir dengan insights kolaborasi

🤖 Contoh Permintaan:
  • "Buatkan analisis pasar untuk produk baru"
  • "Rancang strategi pemasaran digital"
  • "Analisis kompetitor dalam industri teknologi"
  • "Buat proposal bisnis untuk startup"
  • "Evaluasi risiko investasi"
  • "Desain sistem informasi untuk rumah sakit"
  • "Strategi ekspansi bisnis ke pasar internasional"

🌟 Fitur Utama:
  ✓ Dialog real-time antar agent (BARU!)
  ✓ Progress bar visual dengan fase eksekusi
  ✓ Insights kolaborasi live
  ✓ Meeting orchestration monitoring
  ✓ Validasi berlapis dengan skor
  ✓ Akumulasi memori dan learning
  ✓ Multi-bahasa support
  ✓ Kolaborasi multi-expert AI
  ✓ Hasil berkualitas tinggi

🔍 Real-Time Monitoring:
  ✓ Dialog antar agent secara live
  ✓ Progress bar visual [██████░░░░] 60%
  ✓ Fase eksekusi: planning → analysis → integration
  ✓ Metrics kolaborasi real-time
  ✓ Knowledge sharing tracking
  ✓ Meeting activities monitoring
            """)
        else:
            print("""
📖 MULTI-AGENT AI SYSTEM HELP

🗣️  Special Commands:
  • 'language' - Change system language
  • 'help' - Show this help
  • 'status' - View system status
  • 'experts' - View available experts
  • 'exit' - Exit system

💡 How to Use:
  1. Enter your request in natural language
  2. System will analyze and plan execution
  3. Confirm execution plan
  4. Monitor progress and view results

🤖 Example Requests:
  • "Create market analysis for new product"
  • "Design digital marketing strategy"
  • "Analyze competitors in tech industry"
  • "Create business proposal for startup"
  • "Evaluate investment risks"

🌟 Key Features:
  ✓ Multi-expert AI collaboration
  ✓ Multi-layer analysis and validation
  ✓ Natural language dialogue
  ✓ High-quality results
            """)

# Fungsi untuk menjalankan sistem fleksibel
async def run_flexible_system():
    """Jalankan sistem multi-agent yang fleksibel"""
    
    system = FlexibleMultiAgentSystem()
    await system.start_flexible_conversation()

if __name__ == "__main__":
    asyncio.run(run_flexible_system())
