"""
Advanced Memory System for Multi-Agent AI
Implements Episodic and Semantic Memory as described in NTT's research
"""

import json
import uuid
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from core.shared_resources import global_encoder

@dataclass
class EpisodicMemory:
    """Individual experience memory - specific events and interactions"""
    id: str
    agent_id: str
    timestamp: datetime
    event_type: str  # dialogue, task_completion, meeting, validation
    context: Dict[str, Any]
    participants: List[str]
    content: str
    emotional_valence: float  # -1.0 to 1.0
    importance_score: float  # 0.0 to 1.0
    related_memories: List[str]
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class SemanticMemory:
    """Generalized knowledge and facts"""
    id: str
    agent_id: str
    concept: str
    knowledge_type: str  # fact, rule, pattern, expertise
    content: str
    confidence_level: float  # 0.0 to 1.0
    source_episodes: List[str]  # References to episodic memories
    validation_count: int
    last_updated: datetime
    expertise_domain: str
    
    def __post_init__(self):
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)

class MemoryEncoder:
    """Handles encoding and similarity calculations for memories - now using global encoder"""
    
    def __init__(self):
        # Use global encoder singleton instead of creating new instance
        pass
        
    def encode_content(self, content: str) -> np.ndarray:
        """Encode text content to vector representation"""
        return global_encoder.encode_content(content)
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two pieces of content"""
        return global_encoder.calculate_similarity(content1, content2)

class MemoryConsolidation:
    """Handles the conversion of episodic memories to semantic knowledge"""
    
    def __init__(self, encoder: MemoryEncoder):
        self.encoder = encoder
        
    def consolidate_episodes(self, episodes: List[EpisodicMemory]) -> List[SemanticMemory]:
        """Convert multiple related episodic memories into semantic knowledge"""
        if len(episodes) < 2:
            return []
            
        # Group episodes by similarity and context
        clusters = self._cluster_episodes(episodes)
        semantic_memories = []
        
        for cluster in clusters:
            if len(cluster) >= 2:  # Need at least 2 episodes to form semantic memory
                semantic_memory = self._create_semantic_from_cluster(cluster)
                if semantic_memory:
                    semantic_memories.append(semantic_memory)
                    
        return semantic_memories
    
    def _cluster_episodes(self, episodes: List[EpisodicMemory]) -> List[List[EpisodicMemory]]:
        """Cluster similar episodes together"""
        clusters = []
        similarity_threshold = 0.7
        
        for episode in episodes:
            placed = False
            for cluster in clusters:
                # Check similarity with cluster representative (first episode)
                similarity = self.encoder.calculate_similarity(
                    episode.content, cluster[0].content
                )
                if similarity > similarity_threshold:
                    cluster.append(episode)
                    placed = True
                    break
            
            if not placed:
                clusters.append([episode])
                
        return clusters
    
    def _create_semantic_from_cluster(self, cluster: List[EpisodicMemory]) -> Optional[SemanticMemory]:
        """Create semantic memory from a cluster of related episodes"""
        if not cluster:
            return None
            
        # Extract common patterns and knowledge
        agent_id = cluster[0].agent_id
        concepts = self._extract_concepts(cluster)
        
        if not concepts:
            return None
            
        return SemanticMemory(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            concept=concepts[0],  # Primary concept
            knowledge_type="pattern",
            content=self._synthesize_knowledge(cluster),
            confidence_level=min(1.0, len(cluster) * 0.2),  # Higher confidence with more episodes
            source_episodes=[ep.id for ep in cluster],
            validation_count=len(cluster),
            last_updated=datetime.now(),
            expertise_domain=self._determine_domain(cluster)
        )
    
    def _extract_concepts(self, cluster: List[EpisodicMemory]) -> List[str]:
        """Extract key concepts from episode cluster"""
        all_content = " ".join([ep.content for ep in cluster])
        # Simplified concept extraction - in real implementation, use NLP techniques
        words = all_content.lower().split()
        concept_counts = {}
        
        for word in words:
            if len(word) > 4:  # Filter short words
                concept_counts[word] = concept_counts.get(word, 0) + 1
                
        # Return most frequent concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, count in sorted_concepts[:5] if count > 1]
    
    def _synthesize_knowledge(self, cluster: List[EpisodicMemory]) -> str:
        """Synthesize knowledge from multiple episodes"""
        patterns = []
        for episode in cluster:
            if episode.importance_score > 0.7:
                patterns.append(f"Pattern from {episode.event_type}: {episode.content}")
        
        return "Synthesized knowledge: " + " | ".join(patterns[:3])
    
    def _determine_domain(self, cluster: List[EpisodicMemory]) -> str:
        """Determine the expertise domain for the semantic memory"""
        contexts = [ep.context.get('domain', 'general') for ep in cluster]
        # Return most common domain
        domain_counts = {}
        for domain in contexts:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return max(domain_counts.items(), key=lambda x: x[1])[0]

class AdvancedMemorySystem:
    """Advanced memory management system combining episodic and semantic memories"""
    
    def __init__(self, agent_id: str, db_path: str = "./chroma_db"):
        self.agent_id = agent_id
        self.encoder = MemoryEncoder()
        self.consolidation = MemoryConsolidation(self.encoder)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections for different memory types
        self.episodic_collection = self.chroma_client.get_or_create_collection(
            name=f"episodic_{agent_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.semantic_collection = self.chroma_client.get_or_create_collection(
            name=f"semantic_{agent_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # In-memory caches
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        
    def store_episodic_memory(self, memory: EpisodicMemory) -> str:
        """Store an episodic memory"""
        # Check and enforce memory limits
        self._enforce_episodic_memory_limit()
        
        # Store in memory cache
        self.episodic_memories[memory.id] = memory
        
        # Store in vector database
        embedding = self.encoder.encode_content(memory.content)
        
        self.episodic_collection.add(
            embeddings=[embedding.tolist()],
            documents=[memory.content],
            metadatas=[{
                "agent_id": memory.agent_id,
                "event_type": memory.event_type,
                "timestamp": memory.timestamp.isoformat(),
                "importance_score": memory.importance_score,
                "emotional_valence": memory.emotional_valence
            }],
            ids=[memory.id]
        )
        
        # Trigger consolidation if enough episodes accumulated
        self._trigger_consolidation()
        
        return memory.id
    
    def store_semantic_memory(self, memory: SemanticMemory) -> str:
        """Store a semantic memory"""
        # Check and enforce memory limits
        self._enforce_semantic_memory_limit()
        
        # Store in memory cache
        self.semantic_memories[memory.id] = memory
        
        # Store in vector database
        embedding = self.encoder.encode_content(memory.content)
        
        self.semantic_collection.add(
            embeddings=[embedding.tolist()],
            documents=[memory.content],
            metadatas=[{
                "agent_id": memory.agent_id,
                "concept": memory.concept,
                "knowledge_type": memory.knowledge_type,
                "confidence_level": memory.confidence_level,
                "expertise_domain": memory.expertise_domain,
                "last_updated": memory.last_updated.isoformat()
            }],
            ids=[memory.id]
        )
        
        return memory.id
    
    def retrieve_episodic_memories(
        self, 
        query: str, 
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[EpisodicMemory]:
        """Retrieve relevant episodic memories based on query"""
        
        results = self.episodic_collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        retrieved_memories = []
        if results['ids'] and results['ids'][0]:
            for memory_id, distance in zip(results['ids'][0], results['distances'][0]):
                similarity = 1 - distance  # Convert distance to similarity
                if similarity >= similarity_threshold:
                    if memory_id in self.episodic_memories:
                        retrieved_memories.append(self.episodic_memories[memory_id])
        
        return retrieved_memories
    
    def retrieve_semantic_memories(
        self, 
        query: str, 
        limit: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[SemanticMemory]:
        """Retrieve relevant semantic memories based on query"""
        
        results = self.semantic_collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        retrieved_memories = []
        if results['ids'] and results['ids'][0]:
            for memory_id, distance in zip(results['ids'][0], results['distances'][0]):
                similarity = 1 - distance
                if similarity >= similarity_threshold:
                    if memory_id in self.semantic_memories:
                        retrieved_memories.append(self.semantic_memories[memory_id])
        
        return retrieved_memories
    
    def get_related_memories(self, memory_id: str) -> Tuple[List[EpisodicMemory], List[SemanticMemory]]:
        """Get memories related to a specific memory"""
        episodic_related = []
        semantic_related = []
        
        # Check if it's an episodic memory
        if memory_id in self.episodic_memories:
            episode = self.episodic_memories[memory_id]
            # Find related episodic memories
            for related_id in episode.related_memories:
                if related_id in self.episodic_memories:
                    episodic_related.append(self.episodic_memories[related_id])
            
            # Find semantic memories that reference this episode
            for semantic in self.semantic_memories.values():
                if memory_id in semantic.source_episodes:
                    semantic_related.append(semantic)
        
        # Check if it's a semantic memory
        elif memory_id in self.semantic_memories:
            semantic = self.semantic_memories[memory_id]
            # Find source episodes
            for episode_id in semantic.source_episodes:
                if episode_id in self.episodic_memories:
                    episodic_related.append(self.episodic_memories[episode_id])
        
        return episodic_related, semantic_related
    
    def _trigger_consolidation(self):
        """Trigger memory consolidation when conditions are met"""
        # Consolidate every 20 episodic memories
        if len(self.episodic_memories) % 20 == 0:
            recent_episodes = [
                ep for ep in self.episodic_memories.values()
                if ep.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if len(recent_episodes) >= 5:
                new_semantic_memories = self.consolidation.consolidate_episodes(recent_episodes)
                for semantic_memory in new_semantic_memories:
                    self.store_semantic_memory(semantic_memory)
    
    def update_memory_importance(self, memory_id: str, new_importance: float):
        """Update the importance score of a memory"""
        if memory_id in self.episodic_memories:
            self.episodic_memories[memory_id].importance_score = new_importance
        elif memory_id in self.semantic_memories:
            self.semantic_memories[memory_id].confidence_level = new_importance
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        return {
            "episodic_count": len(self.episodic_memories),
            "semantic_count": len(self.semantic_memories),
            "total_memories": len(self.episodic_memories) + len(self.semantic_memories),
            "recent_episodes": len([
                ep for ep in self.episodic_memories.values()
                if ep.timestamp > datetime.now() - timedelta(hours=24)
            ]),
            "high_confidence_semantic": len([
                sm for sm in self.semantic_memories.values()
                if sm.confidence_level > 0.8
            ]),
            "episodic_limit": CONFIG.max_episodic_memories,
            "semantic_limit": CONFIG.max_semantic_memories
        }
    
    def export_memories(self) -> Dict[str, Any]:
        """Export all memories for persistence or transfer"""
        return {
            "episodic": {
                mid: {
                    **asdict(memory),
                    "timestamp": memory.timestamp.isoformat()
                }
                for mid, memory in self.episodic_memories.items()
            },
            "semantic": {
                mid: {
                    **asdict(memory),
                    "last_updated": memory.last_updated.isoformat()
                }
                for mid, memory in self.semantic_memories.items()
            }
        }
    
    def import_memories(self, memory_data: Dict[str, Any]):
        """Import memories from exported data"""
        if "episodic" in memory_data:
            for memory_data_item in memory_data["episodic"].values():
                memory = EpisodicMemory(**memory_data_item)
                self.episodic_memories[memory.id] = memory
        
        if "semantic" in memory_data:
            for memory_data_item in memory_data["semantic"].values():
                memory = SemanticMemory(**memory_data_item)
                self.semantic_memories[memory.id] = memory
    
    def _enforce_episodic_memory_limit(self):
        """Enforce episodic memory limits dengan LRU + importance eviction"""
        if len(self.episodic_memories) >= CONFIG.max_episodic_memories:
            # Hitung berapa yang perlu dihapus (hapus 20%)
            evict_count = max(1, int(CONFIG.max_episodic_memories * 0.2))
            
            # Sort berdasarkan composite score: timestamp (age) + importance
            memories_with_score = []
            current_time = datetime.now()
            
            for memory in self.episodic_memories.values():
                age_hours = (current_time - memory.timestamp).total_seconds() / 3600
                # Score: importance (0-1) - age_penalty (0-1)
                age_penalty = min(1.0, age_hours / (24 * 7))  # Week normalize
                composite_score = memory.importance_score - (age_penalty * 0.5)
                
                memories_with_score.append((memory.id, composite_score, memory.timestamp))
            
            # Sort by composite score (ascending = worst first)
            memories_with_score.sort(key=lambda x: x[1])
            
            # Evict lowest scoring memories
            evicted_count = 0
            for memory_id, score, timestamp in memories_with_score:
                if evicted_count >= evict_count:
                    break
                    
                # Hapus dari cache dan vector DB
                if memory_id in self.episodic_memories:
                    del self.episodic_memories[memory_id]
                    try:
                        self.episodic_collection.delete(ids=[memory_id])
                    except:
                        pass  # Ignore vector DB errors
                    evicted_count += 1
            
            logging.info(f"Evicted {evicted_count} episodic memories (LRU + importance)")
    
    def _enforce_semantic_memory_limit(self):
        """Enforce semantic memory limits dengan confidence-based eviction"""
        if len(self.semantic_memories) >= CONFIG.max_semantic_memories:
            # Hitung berapa yang perlu dihapus (hapus 20%)
            evict_count = max(1, int(CONFIG.max_semantic_memories * 0.2))
            
            # Sort berdasarkan confidence level + last_updated
            memories_with_score = []
            current_time = datetime.now()
            
            for memory in self.semantic_memories.values():
                age_hours = (current_time - memory.last_updated).total_seconds() / 3600
                # Score: confidence - staleness_penalty
                staleness_penalty = min(0.3, age_hours / (24 * 30))  # Month normalize
                composite_score = memory.confidence_level - staleness_penalty
                
                memories_with_score.append((memory.id, composite_score, memory.last_updated))
            
            # Sort by composite score (ascending = worst first)
            memories_with_score.sort(key=lambda x: x[1])
            
            # Evict lowest scoring memories
            evicted_count = 0
            for memory_id, score, last_updated in memories_with_score:
                if evicted_count >= evict_count:
                    break
                    
                # Hapus dari cache dan vector DB
                if memory_id in self.semantic_memories:
                    del self.semantic_memories[memory_id]
                    try:
                        self.semantic_collection.delete(ids=[memory_id])
                    except:
                        pass  # Ignore vector DB errors
                    evicted_count += 1
            
            logging.info(f"Evicted {evicted_count} semantic memories (confidence + staleness)")
    
    def cleanup_expired_memories(self):
        """Clean up expired memories berdasarkan TTL"""
        current_time = datetime.now()
        
        # TTL untuk episodic: 30 hari
        episodic_ttl = timedelta(days=30)
        expired_episodic = []
        
        for memory_id, memory in self.episodic_memories.items():
            if current_time - memory.timestamp > episodic_ttl:
                expired_episodic.append(memory_id)
        
        # Hapus expired episodic memories
        for memory_id in expired_episodic:
            del self.episodic_memories[memory_id]
            try:
                self.episodic_collection.delete(ids=[memory_id])
            except:
                pass
        
        # TTL untuk semantic: 90 hari tanpa update
        semantic_ttl = timedelta(days=90)
        expired_semantic = []
        
        for memory_id, memory in self.semantic_memories.items():
            if current_time - memory.last_updated > semantic_ttl:
                expired_semantic.append(memory_id)
        
        # Hapus expired semantic memories
        for memory_id in expired_semantic:
            del self.semantic_memories[memory_id]
            try:
                self.semantic_collection.delete(ids=[memory_id])
            except:
                pass
        
        if expired_episodic or expired_semantic:
            logging.info(f"Cleaned up {len(expired_episodic)} expired episodic, {len(expired_semantic)} expired semantic memories")
