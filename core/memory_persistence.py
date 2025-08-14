"""
Memory Persistence & Rotation untuk Multi-Agent AI System
Automated snapshot/restore dan ChromaDB maintenance
"""

import os
import json
import asyncio
import logging
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import schedule
import time
from threading import Thread

class MemorySnapshot:
    """Snapshot management untuk memory system"""
    
    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Retention policy
        self.max_snapshots = 10
        self.snapshot_interval_hours = 6
        self.compress_snapshots = True
        
        logging.info(f"Memory snapshot manager initialized: {self.snapshot_dir}")
    
    async def create_snapshot(self, memory_system, agent_id: str = "system") -> str:
        """Create memory snapshot"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"memory_snapshot_{agent_id}_{timestamp}"
        
        try:
            # Export memories
            memory_data = memory_system.export_memories()
            
            # Add metadata
            snapshot_data = {
                "metadata": {
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "snapshot_name": snapshot_name,
                    "version": "1.0",
                    "episodic_count": len(memory_data.get("episodic", {})),
                    "semantic_count": len(memory_data.get("semantic", {}))
                },
                "data": memory_data
            }
            
            # Save snapshot
            snapshot_path = self.snapshot_dir / f"{snapshot_name}.json"
            
            if self.compress_snapshots:
                # Save compressed
                snapshot_path = self.snapshot_dir / f"{snapshot_name}.json.gz"
                with gzip.open(snapshot_path, 'wt', encoding='utf-8') as f:
                    json.dump(snapshot_data, f, indent=2)
            else:
                # Save uncompressed
                with open(snapshot_path, 'w', encoding='utf-8') as f:
                    json.dump(snapshot_data, f, indent=2)
            
            # Cleanup old snapshots
            await self._cleanup_old_snapshots(agent_id)
            
            logging.info(f"Memory snapshot created: {snapshot_path}")
            return str(snapshot_path)
            
        except Exception as e:
            logging.error(f"Failed to create memory snapshot: {str(e)}")
            raise
    
    async def restore_snapshot(self, memory_system, snapshot_path: str) -> bool:
        """Restore memory dari snapshot"""
        
        try:
            snapshot_path = Path(snapshot_path)
            
            if not snapshot_path.exists():
                logging.error(f"Snapshot file not found: {snapshot_path}")
                return False
            
            # Load snapshot
            if snapshot_path.suffix == '.gz':
                with gzip.open(snapshot_path, 'rt', encoding='utf-8') as f:
                    snapshot_data = json.load(f)
            else:
                with open(snapshot_path, 'r', encoding='utf-8') as f:
                    snapshot_data = json.load(f)
            
            # Validate snapshot
            if "data" not in snapshot_data or "metadata" not in snapshot_data:
                logging.error("Invalid snapshot format")
                return False
            
            # Restore memories
            memory_system.import_memories(snapshot_data["data"])
            
            metadata = snapshot_data["metadata"]
            logging.info(f"Memory snapshot restored: {metadata['snapshot_name']}")
            logging.info(f"Restored {metadata['episodic_count']} episodic, {metadata['semantic_count']} semantic memories")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore memory snapshot: {str(e)}")
            return False
    
    async def _cleanup_old_snapshots(self, agent_id: str):
        """Cleanup old snapshots berdasarkan retention policy"""
        
        # Find all snapshots untuk agent
        pattern = f"memory_snapshot_{agent_id}_*.json*"
        snapshots = list(self.snapshot_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        snapshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove excess snapshots
        if len(snapshots) > self.max_snapshots:
            for old_snapshot in snapshots[self.max_snapshots:]:
                try:
                    old_snapshot.unlink()
                    logging.info(f"Removed old snapshot: {old_snapshot.name}")
                except Exception as e:
                    logging.error(f"Failed to remove old snapshot {old_snapshot}: {str(e)}")
    
    def list_snapshots(self, agent_id: str = None) -> List[Dict[str, Any]]:
        """List available snapshots"""
        
        pattern = f"memory_snapshot_{agent_id}_*.json*" if agent_id else "memory_snapshot_*.json*"
        snapshots = []
        
        for snapshot_path in self.snapshot_dir.glob(pattern):
            try:
                # Get file stats
                stat = snapshot_path.stat()
                
                # Try to read metadata
                metadata = None
                try:
                    if snapshot_path.suffix == '.gz':
                        with gzip.open(snapshot_path, 'rt', encoding='utf-8') as f:
                            data = json.load(f)
                    else:
                        with open(snapshot_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    
                    metadata = data.get("metadata", {})
                except:
                    pass
                
                snapshots.append({
                    "path": str(snapshot_path),
                    "name": snapshot_path.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "metadata": metadata
                })
                
            except Exception as e:
                logging.error(f"Error reading snapshot {snapshot_path}: {str(e)}")
        
        # Sort by creation time (newest first)
        snapshots.sort(key=lambda x: x["created"], reverse=True)
        return snapshots

class ChromaDBMaintenance:
    """ChromaDB maintenance dan optimization"""
    
    def __init__(self, chroma_db_path: str = "chroma_db"):
        self.chroma_db_path = Path(chroma_db_path)
        self.maintenance_log = []
        
        # Maintenance schedule
        self.vacuum_interval_days = 7
        self.compact_interval_days = 30
        self.backup_interval_days = 1
        
        logging.info(f"ChromaDB maintenance manager initialized: {self.chroma_db_path}")
    
    async def vacuum_database(self) -> bool:
        """Vacuum ChromaDB untuk reclaim space"""
        
        try:
            start_time = datetime.now()
            
            # Get database size before
            size_before = await self._get_db_size()
            
            # Run vacuum operation (ChromaDB specific)
            # Note: ChromaDB doesn't have explicit VACUUM, but we can optimize collections
            await self._optimize_collections()
            
            # Get database size after
            size_after = await self._get_db_size()
            
            duration = (datetime.now() - start_time).total_seconds()
            space_saved = size_before - size_after
            
            self.maintenance_log.append({
                "operation": "vacuum",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "size_before_mb": size_before,
                "size_after_mb": size_after,
                "space_saved_mb": space_saved,
                "status": "success"
            })
            
            logging.info(f"ChromaDB vacuum completed: {space_saved:.2f}MB saved in {duration:.1f}s")
            return True
            
        except Exception as e:
            self.maintenance_log.append({
                "operation": "vacuum",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            logging.error(f"ChromaDB vacuum failed: {str(e)}")
            return False
    
    async def compact_database(self) -> bool:
        """Compact ChromaDB untuk optimize storage"""
        
        try:
            start_time = datetime.now()
            
            # Get database size before
            size_before = await self._get_db_size()
            
            # Run compaction
            await self._compact_collections()
            
            # Get database size after
            size_after = await self._get_db_size()
            
            duration = (datetime.now() - start_time).total_seconds()
            space_saved = size_before - size_after
            
            self.maintenance_log.append({
                "operation": "compact",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "size_before_mb": size_before,
                "size_after_mb": size_after,
                "space_saved_mb": space_saved,
                "status": "success"
            })
            
            logging.info(f"ChromaDB compact completed: {space_saved:.2f}MB saved in {duration:.1f}s")
            return True
            
        except Exception as e:
            self.maintenance_log.append({
                "operation": "compact",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            logging.error(f"ChromaDB compact failed: {str(e)}")
            return False
    
    async def backup_database(self, backup_dir: str = "backups") -> str:
        """Create backup dari ChromaDB"""
        
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"chromadb_backup_{timestamp}"
            backup_full_path = backup_path / backup_name
            
            # Copy database files
            if self.chroma_db_path.exists():
                shutil.copytree(self.chroma_db_path, backup_full_path)
                
                # Compress backup
                shutil.make_archive(str(backup_full_path), 'gztar', backup_path, backup_name)
                
                # Remove uncompressed backup
                shutil.rmtree(backup_full_path)
                
                backup_file = f"{backup_full_path}.tar.gz"
                
                self.maintenance_log.append({
                    "operation": "backup",
                    "timestamp": datetime.now().isoformat(),
                    "backup_path": backup_file,
                    "status": "success"
                })
                
                logging.info(f"ChromaDB backup created: {backup_file}")
                return backup_file
            else:
                logging.warning("ChromaDB path does not exist, skipping backup")
                return ""
                
        except Exception as e:
            self.maintenance_log.append({
                "operation": "backup",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            logging.error(f"ChromaDB backup failed: {str(e)}")
            raise
    
    async def _get_db_size(self) -> float:
        """Get database size in MB"""
        
        if not self.chroma_db_path.exists():
            return 0.0
        
        total_size = 0
        for file_path in self.chroma_db_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def _optimize_collections(self):
        """Optimize ChromaDB collections"""
        # This would interface with ChromaDB's optimization APIs
        # For now, we'll just log the operation
        logging.info("Optimizing ChromaDB collections...")
        await asyncio.sleep(1)  # Simulate optimization work
    
    async def _compact_collections(self):
        """Compact ChromaDB collections"""
        # This would interface with ChromaDB's compaction APIs
        # For now, we'll just log the operation
        logging.info("Compacting ChromaDB collections...")
        await asyncio.sleep(2)  # Simulate compaction work
    
    def get_maintenance_log(self) -> List[Dict[str, Any]]:
        """Get maintenance operation log"""
        return self.maintenance_log[-20:]  # Last 20 operations

class MemoryPersistenceScheduler:
    """Scheduler untuk automated memory persistence tasks"""
    
    def __init__(self, memory_system, snapshot_manager: MemorySnapshot, db_maintenance: ChromaDBMaintenance):
        self.memory_system = memory_system
        self.snapshot_manager = snapshot_manager
        self.db_maintenance = db_maintenance
        self.scheduler_thread = None
        self.running = False
        
        # Setup scheduled tasks
        self._setup_schedule()
    
    def _setup_schedule(self):
        """Setup scheduled tasks"""
        
        # Memory snapshots every 6 hours
        schedule.every(6).hours.do(self._run_async_task, self._create_scheduled_snapshot)
        
        # Database vacuum weekly
        schedule.every().sunday.at("02:00").do(self._run_async_task, self.db_maintenance.vacuum_database)
        
        # Database backup daily
        schedule.every().day.at("01:00").do(self._run_async_task, self.db_maintenance.backup_database)
        
        # Database compact monthly (first Sunday)
        schedule.every().sunday.at("03:00").do(self._run_async_task, self._monthly_compact)
        
        logging.info("Memory persistence scheduler configured")
    
    def _run_async_task(self, coro):
        """Run async task in scheduler thread"""
        try:
            # Create new event loop for scheduler thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro())
            loop.close()
        except Exception as e:
            logging.error(f"Scheduled task failed: {str(e)}")
    
    async def _create_scheduled_snapshot(self):
        """Create scheduled memory snapshot"""
        try:
            snapshot_path = await self.snapshot_manager.create_snapshot(self.memory_system)
            logging.info(f"Scheduled snapshot created: {snapshot_path}")
        except Exception as e:
            logging.error(f"Scheduled snapshot failed: {str(e)}")
    
    async def _monthly_compact(self):
        """Monthly database compact (first Sunday only)"""
        import calendar
        
        today = datetime.now()
        # Check if this is the first Sunday of the month
        first_sunday = None
        for day in range(1, 8):
            date = datetime(today.year, today.month, day)
            if date.weekday() == 6:  # Sunday
                first_sunday = day
                break
        
        if today.day == first_sunday:
            await self.db_maintenance.compact_database()
        else:
            logging.debug("Skipping monthly compact - not first Sunday")
    
    def start(self):
        """Start scheduler thread"""
        if self.running:
            logging.warning("Scheduler already running")
            return
        
        self.running = True
        self.scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logging.info("Memory persistence scheduler started")
    
    def stop(self):
        """Stop scheduler thread"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logging.info("Memory persistence scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Scheduler loop error: {str(e)}")
                time.sleep(60)
    
    def get_next_runs(self) -> Dict[str, str]:
        """Get next scheduled run times"""
        jobs = schedule.jobs
        next_runs = {}
        
        for job in jobs:
            next_runs[str(job.job_func)] = str(job.next_run) if job.next_run else "Not scheduled"
        
        return next_runs

# Global instances (akan diinisialisasi di main system)
snapshot_manager = None
db_maintenance = None
persistence_scheduler = None

def initialize_persistence(memory_system):
    """Initialize memory persistence components"""
    global snapshot_manager, db_maintenance, persistence_scheduler
    
    snapshot_manager = MemorySnapshot()
    db_maintenance = ChromaDBMaintenance()
    persistence_scheduler = MemoryPersistenceScheduler(
        memory_system, snapshot_manager, db_maintenance
    )
    
    # Start scheduler
    persistence_scheduler.start()
    
    logging.info("Memory persistence system initialized")

def shutdown_persistence():
    """Shutdown memory persistence components"""
    global persistence_scheduler
    
    if persistence_scheduler:
        persistence_scheduler.stop()
    
    logging.info("Memory persistence system shutdown")

# Test function
if __name__ == "__main__":
    # Mock memory system for testing
    class MockMemorySystem:
        def export_memories(self):
            return {
                "episodic": {"mem1": {"content": "test episodic"}},
                "semantic": {"mem2": {"content": "test semantic"}}
            }
        
        def import_memories(self, data):
            print(f"Imported: {len(data.get('episodic', {}))} episodic, {len(data.get('semantic', {}))} semantic")
    
    async def test_persistence():
        mock_memory = MockMemorySystem()
        
        # Test snapshot
        snapshot_mgr = MemorySnapshot()
        snapshot_path = await snapshot_mgr.create_snapshot(mock_memory, "test_agent")
        print(f"Created snapshot: {snapshot_path}")
        
        # Test restore
        success = await snapshot_mgr.restore_snapshot(mock_memory, snapshot_path)
        print(f"Restore success: {success}")
        
        # Test DB maintenance
        db_maint = ChromaDBMaintenance()
        await db_maint.vacuum_database()
        await db_maint.backup_database()
    
    asyncio.run(test_persistence())
