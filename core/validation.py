"""
Advanced Knowledge Validation and Cross-Checking System
Implements multi-agent validation as described in NTT's research
"""

import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import openai
from config import CONFIG
import numpy as np
from sentence_transformers import SentenceTransformer

class ValidationType(Enum):
    """Types of validation performed"""
    FACTUAL_ACCURACY = "factual_accuracy"
    LOGICAL_CONSISTENCY = "logical_consistency"
    COMPLETENESS = "completeness"
    FEASIBILITY = "feasibility"
    RELEVANCE = "relevance"
    EXPERTISE_ALIGNMENT = "expertise_alignment"
    CROSS_REFERENCE = "cross_reference"
    CONSENSUS_CHECK = "consensus_check"

class ValidationResult(Enum):
    """Results of validation"""
    VALIDATED = "validated"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"
    NEEDS_MORE_INFO = "needs_more_info"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EXPERTISE = "insufficient_expertise"

@dataclass
class ValidationRequest:
    """Request for knowledge validation"""
    id: str
    requester_id: str
    content: str
    validation_types: List[ValidationType]
    context: Dict[str, Any]
    priority: int
    deadline: Optional[datetime]
    required_validators: List[str]  # Specific agents required for validation
    minimum_validators: int
    created_at: datetime
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.deadline, str):
            self.deadline = datetime.fromisoformat(self.deadline)

@dataclass
class ValidationResponse:
    """Response to validation request"""
    id: str
    request_id: str
    validator_id: str
    validation_type: ValidationType
    result: ValidationResult
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]  # Supporting evidence or references
    suggested_improvements: List[str]
    alternative_perspectives: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class CrossValidationResult:
    """Result of cross-validation across multiple agents"""
    request_id: str
    individual_responses: List[ValidationResponse]
    consensus_level: float  # 0.0 to 1.0
    final_result: ValidationResult
    conflicting_opinions: List[Tuple[str, str]]  # (validator_id, opinion)
    synthesized_feedback: str
    confidence_score: float
    recommendations: List[str]
    created_at: datetime

class KnowledgeValidator:
    """Individual knowledge validator with specific expertise"""
    
    def __init__(self, validator_id: str, expertise_domains: List[str]):
        self.validator_id = validator_id
        self.expertise_domains = expertise_domains
        self.client = openai.OpenAI(api_key=CONFIG.openai_api_key)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def validate_knowledge(
        self, 
        request: ValidationRequest, 
        validation_type: ValidationType,
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate knowledge based on specific validation type"""
        
        # Check if this validator has relevant expertise
        expertise_relevance = self._calculate_expertise_relevance(
            request.content, request.context
        )
        
        if expertise_relevance < 0.3:
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=validation_type,
                result=ValidationResult.INSUFFICIENT_EXPERTISE,
                confidence=0.1,
                reasoning=f"Insufficient expertise in required domain for {validation_type.value}",
                evidence=[],
                suggested_improvements=[],
                alternative_perspectives=[],
                timestamp=datetime.now(),
                metadata={"expertise_relevance": expertise_relevance}
            )
        
        # Perform validation based on type
        validation_methods = {
            ValidationType.FACTUAL_ACCURACY: self._validate_factual_accuracy,
            ValidationType.LOGICAL_CONSISTENCY: self._validate_logical_consistency,
            ValidationType.COMPLETENESS: self._validate_completeness,
            ValidationType.FEASIBILITY: self._validate_feasibility,
            ValidationType.RELEVANCE: self._validate_relevance,
            ValidationType.EXPERTISE_ALIGNMENT: self._validate_expertise_alignment,
            ValidationType.CROSS_REFERENCE: self._validate_cross_reference,
            ValidationType.CONSENSUS_CHECK: self._validate_consensus
        }
        
        validation_method = validation_methods.get(validation_type)
        if validation_method:
            return await validation_method(request, reference_knowledge)
        else:
            return await self._generic_validation(request, validation_type)
    
    def _calculate_expertise_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate how relevant this validator's expertise is to the content"""
        
        # Extract domain keywords from content
        content_embedding = self.sentence_transformer.encode(content.lower())
        
        # Calculate similarity with expertise domains
        max_similarity = 0.0
        for domain in self.expertise_domains:
            domain_embedding = self.sentence_transformer.encode(domain.lower())
            similarity = np.dot(content_embedding, domain_embedding) / (
                np.linalg.norm(content_embedding) * np.linalg.norm(domain_embedding)
            )
            max_similarity = max(max_similarity, similarity)
        
        # Boost relevance if domain is explicitly mentioned in context
        context_domains = context.get('domains', [])
        for domain in context_domains:
            if any(expertise in domain.lower() for expertise in self.expertise_domains):
                max_similarity += 0.2
        
        return min(1.0, max_similarity)
    
    async def _validate_factual_accuracy(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate factual accuracy of the content"""
        
        system_prompt = f"""
        You are an expert validator with expertise in: {', '.join(self.expertise_domains)}.
        Validate the factual accuracy of the provided content.
        
        Consider:
        1. Are the facts stated correctly?
        2. Are there any factual errors or misconceptions?
        3. Is the information up-to-date?
        4. Are claims supported by evidence?
        5. Are there any contradictions?
        
        Provide detailed reasoning and evidence.
        """
        
        reference_context = ""
        if reference_knowledge:
            reference_context = f"\nReference Knowledge:\n{chr(10).join(reference_knowledge[:5])}"
        
        user_prompt = f"""
        Content to validate: {request.content}
        Context: {json.dumps(request.context, indent=2)}
        {reference_context}
        
        Validate the factual accuracy and provide:
        1. Overall result (validated/rejected/requires_revision/needs_more_info)
        2. Confidence level (0.0-1.0)
        3. Detailed reasoning
        4. Specific evidence or sources
        5. Suggested improvements if needed
        6. Alternative perspectives
        
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
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=ValidationType.FACTUAL_ACCURACY,
                result=ValidationResult(result_data.get("result", "needs_more_info")),
                confidence=float(result_data.get("confidence", 0.5)),
                reasoning=result_data.get("reasoning", ""),
                evidence=result_data.get("evidence", []),
                suggested_improvements=result_data.get("suggested_improvements", []),
                alternative_perspectives=result_data.get("alternative_perspectives", []),
                timestamp=datetime.now(),
                metadata={"validation_method": "ai_assisted"}
            )
            
        except Exception as e:
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=ValidationType.FACTUAL_ACCURACY,
                result=ValidationResult.NEEDS_MORE_INFO,
                confidence=0.1,
                reasoning=f"Error during validation: {str(e)}",
                evidence=[],
                suggested_improvements=[],
                alternative_perspectives=[],
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    async def _validate_logical_consistency(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate logical consistency of the content"""
        
        system_prompt = f"""
        You are a logic expert with expertise in: {', '.join(self.expertise_domains)}.
        Validate the logical consistency of the provided content.
        
        Check for:
        1. Internal contradictions
        2. Logical flow and reasoning
        3. Cause-and-effect relationships
        4. Consistency with established principles
        5. Coherent argumentation
        """
        
        user_prompt = f"""
        Content: {request.content}
        Context: {json.dumps(request.context, indent=2)}
        
        Analyze logical consistency and provide validation results in JSON format.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=ValidationType.LOGICAL_CONSISTENCY,
                result=ValidationResult(result_data.get("result", "validated")),
                confidence=float(result_data.get("confidence", 0.7)),
                reasoning=result_data.get("reasoning", ""),
                evidence=result_data.get("evidence", []),
                suggested_improvements=result_data.get("suggested_improvements", []),
                alternative_perspectives=result_data.get("alternative_perspectives", []),
                timestamp=datetime.now(),
                metadata={"validation_method": "logical_analysis"}
            )
            
        except Exception as e:
            return self._create_error_response(request, ValidationType.LOGICAL_CONSISTENCY, e)
    
    async def _validate_completeness(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate completeness of the content"""
        
        system_prompt = f"""
        Evaluate the completeness of the provided content based on your expertise in: {', '.join(self.expertise_domains)}.
        
        Assess:
        1. Are all necessary components covered?
        2. What important aspects are missing?
        3. Is the depth of coverage appropriate?
        4. Are there gaps in reasoning or information?
        """
        
        user_prompt = f"""
        Content: {request.content}
        Context: {json.dumps(request.context, indent=2)}
        
        Evaluate completeness and return JSON validation results.
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
            
            result_data = json.loads(response.choices[0].message.content)
            
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=ValidationType.COMPLETENESS,
                result=ValidationResult(result_data.get("result", "requires_revision")),
                confidence=float(result_data.get("confidence", 0.6)),
                reasoning=result_data.get("reasoning", ""),
                evidence=result_data.get("evidence", []),
                suggested_improvements=result_data.get("suggested_improvements", []),
                alternative_perspectives=result_data.get("alternative_perspectives", []),
                timestamp=datetime.now(),
                metadata={"validation_method": "completeness_analysis"}
            )
            
        except Exception as e:
            return self._create_error_response(request, ValidationType.COMPLETENESS, e)
    
    async def _validate_feasibility(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate feasibility of proposed solutions or plans"""
        
        system_prompt = f"""
        As an expert in {', '.join(self.expertise_domains)}, evaluate the feasibility of the proposed content.
        
        Consider:
        1. Technical feasibility
        2. Resource requirements
        3. Time constraints
        4. Risk factors
        5. Practical implementation challenges
        """
        
        user_prompt = f"""
        Content: {request.content}
        Context: {json.dumps(request.context, indent=2)}
        
        Assess feasibility and provide JSON validation results.
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
            
            result_data = json.loads(response.choices[0].message.content)
            
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=ValidationType.FEASIBILITY,
                result=ValidationResult(result_data.get("result", "validated")),
                confidence=float(result_data.get("confidence", 0.7)),
                reasoning=result_data.get("reasoning", ""),
                evidence=result_data.get("evidence", []),
                suggested_improvements=result_data.get("suggested_improvements", []),
                alternative_perspectives=result_data.get("alternative_perspectives", []),
                timestamp=datetime.now(),
                metadata={"validation_method": "feasibility_analysis"}
            )
            
        except Exception as e:
            return self._create_error_response(request, ValidationType.FEASIBILITY, e)
    
    async def _validate_relevance(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate relevance of content to the specified context"""
        
        relevance_score = self._calculate_content_relevance(request.content, request.context)
        
        if relevance_score > 0.8:
            result = ValidationResult.VALIDATED
        elif relevance_score > 0.6:
            result = ValidationResult.REQUIRES_REVISION
        else:
            result = ValidationResult.REJECTED
        
        return ValidationResponse(
            id=str(uuid.uuid4()),
            request_id=request.id,
            validator_id=self.validator_id,
            validation_type=ValidationType.RELEVANCE,
            result=result,
            confidence=relevance_score,
            reasoning=f"Content relevance score: {relevance_score:.2f}",
            evidence=[f"Semantic similarity analysis"],
            suggested_improvements=["Focus more on core topic", "Remove tangential information"] if relevance_score < 0.8 else [],
            alternative_perspectives=[],
            timestamp=datetime.now(),
            metadata={"relevance_score": relevance_score}
        )
    
    def _calculate_content_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate relevance of content to context"""
        
        # Extract key terms from context
        context_text = " ".join([
            str(v) for v in context.values() if isinstance(v, (str, int, float))
        ])
        
        if not context_text:
            return 0.5  # Neutral relevance if no context
        
        # Calculate semantic similarity
        content_embedding = self.sentence_transformer.encode(content.lower())
        context_embedding = self.sentence_transformer.encode(context_text.lower())
        
        similarity = np.dot(content_embedding, context_embedding) / (
            np.linalg.norm(content_embedding) * np.linalg.norm(context_embedding)
        )
        
        return max(0.0, min(1.0, similarity))
    
    async def _validate_expertise_alignment(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Validate alignment with domain expertise"""
        
        expertise_relevance = self._calculate_expertise_relevance(request.content, request.context)
        
        if expertise_relevance > 0.8:
            result = ValidationResult.VALIDATED
        elif expertise_relevance > 0.5:
            result = ValidationResult.REQUIRES_REVISION
        else:
            result = ValidationResult.INSUFFICIENT_EXPERTISE
        
        return ValidationResponse(
            id=str(uuid.uuid4()),
            request_id=request.id,
            validator_id=self.validator_id,
            validation_type=ValidationType.EXPERTISE_ALIGNMENT,
            result=result,
            confidence=expertise_relevance,
            reasoning=f"Expertise alignment score: {expertise_relevance:.2f}",
            evidence=[f"Domain expertise: {', '.join(self.expertise_domains)}"],
            suggested_improvements=["Consult domain expert", "Add domain-specific details"] if expertise_relevance < 0.8 else [],
            alternative_perspectives=[],
            timestamp=datetime.now(),
            metadata={"expertise_relevance": expertise_relevance}
        )
    
    async def _validate_cross_reference(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Cross-reference with existing knowledge"""
        
        if not reference_knowledge:
            return ValidationResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                validator_id=self.validator_id,
                validation_type=ValidationType.CROSS_REFERENCE,
                result=ValidationResult.NEEDS_MORE_INFO,
                confidence=0.1,
                reasoning="No reference knowledge provided for cross-referencing",
                evidence=[],
                suggested_improvements=["Provide reference knowledge"],
                alternative_perspectives=[],
                timestamp=datetime.now(),
                metadata={}
            )
        
        # Calculate similarity with reference knowledge
        content_embedding = self.sentence_transformer.encode(request.content)
        similarities = []
        
        for ref in reference_knowledge:
            ref_embedding = self.sentence_transformer.encode(ref)
            similarity = np.dot(content_embedding, ref_embedding) / (
                np.linalg.norm(content_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0.0
        
        if max_similarity > 0.9:
            result = ValidationResult.VALIDATED
        elif max_similarity > 0.7:
            result = ValidationResult.REQUIRES_REVISION
        else:
            result = ValidationResult.CONFLICTING_EVIDENCE
        
        return ValidationResponse(
            id=str(uuid.uuid4()),
            request_id=request.id,
            validator_id=self.validator_id,
            validation_type=ValidationType.CROSS_REFERENCE,
            result=result,
            confidence=max_similarity,
            reasoning=f"Cross-reference similarity: {max_similarity:.2f}",
            evidence=[f"Compared with {len(reference_knowledge)} reference sources"],
            suggested_improvements=["Align with existing knowledge"] if max_similarity < 0.8 else [],
            alternative_perspectives=[],
            timestamp=datetime.now(),
            metadata={"max_similarity": max_similarity, "reference_count": len(reference_knowledge)}
        )
    
    async def _validate_consensus(
        self, 
        request: ValidationRequest, 
        reference_knowledge: List[str] = None
    ) -> ValidationResponse:
        """Check consensus with peer validators (placeholder - requires coordination)"""
        
        # This would typically coordinate with other validators
        # For now, return a neutral response
        return ValidationResponse(
            id=str(uuid.uuid4()),
            request_id=request.id,
            validator_id=self.validator_id,
            validation_type=ValidationType.CONSENSUS_CHECK,
            result=ValidationResult.NEEDS_MORE_INFO,
            confidence=0.5,
            reasoning="Consensus validation requires coordination with other validators",
            evidence=[],
            suggested_improvements=["Coordinate with peer validators"],
            alternative_perspectives=[],
            timestamp=datetime.now(),
            metadata={"status": "pending_coordination"}
        )
    
    async def _generic_validation(
        self, 
        request: ValidationRequest, 
        validation_type: ValidationType
    ) -> ValidationResponse:
        """Generic validation for unspecified types"""
        
        return ValidationResponse(
            id=str(uuid.uuid4()),
            request_id=request.id,
            validator_id=self.validator_id,
            validation_type=validation_type,
            result=ValidationResult.VALIDATED,
            confidence=0.6,
            reasoning=f"Generic validation for {validation_type.value}",
            evidence=[],
            suggested_improvements=[],
            alternative_perspectives=[],
            timestamp=datetime.now(),
            metadata={"validation_method": "generic"}
        )
    
    def _create_error_response(
        self, 
        request: ValidationRequest, 
        validation_type: ValidationType, 
        error: Exception
    ) -> ValidationResponse:
        """Create error response for failed validations"""
        
        return ValidationResponse(
            id=str(uuid.uuid4()),
            request_id=request.id,
            validator_id=self.validator_id,
            validation_type=validation_type,
            result=ValidationResult.NEEDS_MORE_INFO,
            confidence=0.1,
            reasoning=f"Validation error: {str(error)}",
            evidence=[],
            suggested_improvements=["Retry validation", "Check input format"],
            alternative_perspectives=[],
            timestamp=datetime.now(),
            metadata={"error": str(error)}
        )

class CrossValidationOrchestrator:
    """Orchestrates cross-validation across multiple validators"""
    
    def __init__(self):
        self.validators: Dict[str, KnowledgeValidator] = {}
        self.validation_requests: Dict[str, ValidationRequest] = {}
        self.validation_results: Dict[str, List[ValidationResponse]] = {}
        
    def register_validator(self, validator: KnowledgeValidator):
        """Register a validator with the orchestrator"""
        self.validators[validator.validator_id] = validator
    
    async def cross_validate(
        self, 
        request: ValidationRequest,
        reference_knowledge: List[str] = None
    ) -> CrossValidationResult:
        """Perform cross-validation using multiple validators"""
        
        self.validation_requests[request.id] = request
        
        # Select appropriate validators
        selected_validators = self._select_validators(request)
        
        if len(selected_validators) < request.minimum_validators:
            raise ValueError(f"Insufficient validators: need {request.minimum_validators}, found {len(selected_validators)}")
        
        # Collect validation responses in parallel
        validation_tasks = []
        
        for validator_id in selected_validators:
            validator = self.validators[validator_id]
            
            # Create tasks for each validation type
            for validation_type in request.validation_types:
                task = validator.validate_knowledge(
                    request, validation_type, reference_knowledge
                )
                validation_tasks.append(task)
        
        # Execute all validations in parallel
        all_responses = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_responses = []
        for i, response in enumerate(all_responses):
            if isinstance(response, Exception):
                logging.error(f"Validation task {i} failed: {str(response)}")
            else:
                valid_responses.append(response)
        
        all_responses = valid_responses
        
        # Store responses
        self.validation_results[request.id] = all_responses
        
        # Synthesize cross-validation result
        cross_result = await self._synthesize_validation_results(request.id, all_responses)
        
        return cross_result
    
    async def batch_cross_validate(
        self,
        requests: List[ValidationRequest],
        reference_knowledge: List[str] = None
    ) -> List[CrossValidationResult]:
        """Perform batch cross-validation untuk multiple requests"""
        
        if not requests:
            return []
        
        logging.info(f"Starting batch validation for {len(requests)} requests")
        
        # Prepare batch requests untuk OpenAI
        batch_requests = []
        request_metadata = []
        
        for request in requests:
            selected_validators = self._select_validators(request)
            
            for validator_id in selected_validators:
                validator = self.validators[validator_id]
                
                for validation_type in request.validation_types:
                    # Create validation prompt
                    system_prompt = f"""
                    You are a {validation_type.value} validator with expertise in {', '.join(validator.expertise_domains)}.
                    Validate the provided content for {validation_type.value}.
                    """
                    
                    user_prompt = f"""
                    Content to validate: {request.content[:1000]}
                    Context: {request.context}
                    Validation Type: {validation_type.value}
                    
                    Provide validation result as JSON with:
                    - is_valid: boolean
                    - confidence: 0.0-1.0
                    - feedback: string
                    - issues_found: list of issues
                    """
                    
                    batch_requests.append({
                        'system_prompt': system_prompt,
                        'user_prompt': user_prompt,
                        'temperature': 0.3,
                        'response_format': {"type": "json_object"},
                        'use_cache': True,
                        'use_analysis_cache': True,
                        'analysis_type': f'validation_{validation_type.value}'
                    })
                    
                    request_metadata.append({
                        'request_id': request.id,
                        'validator_id': validator_id,
                        'validation_type': validation_type,
                        'validator': validator
                    })
        
        # Execute batch processing
        from core.shared_resources import openai_manager
        batch_results = await openai_manager.create_batch_completions(
            batch_requests, 
            max_concurrent=8  # Higher concurrency for validation
        )
        
        # Group results by request
        request_results = {}
        
        for i, (result, metadata) in enumerate(zip(batch_results, request_metadata)):
            request_id = metadata['request_id']
            
            if request_id not in request_results:
                request_results[request_id] = []
            
            if result.get('success'):
                try:
                    # Parse validation result
                    validation_data = json.loads(result['content'])
                    
                    response = ValidationResponse(
                        id=str(uuid.uuid4()),
                        request_id=request_id,
                        validator_id=metadata['validator_id'],
                        validation_type=metadata['validation_type'],
                        is_valid=validation_data.get('is_valid', False),
                        confidence=validation_data.get('confidence', 0.0),
                        feedback=validation_data.get('feedback', ''),
                        issues_found=validation_data.get('issues_found', []),
                        suggestions=validation_data.get('suggestions', []),
                        timestamp=datetime.now()
                    )
                    
                    request_results[request_id].append(response)
                    
                except Exception as e:
                    logging.error(f"Failed to parse validation result {i}: {str(e)}")
            else:
                logging.error(f"Validation request {i} failed: {result.get('error', 'Unknown error')}")
        
        # Synthesize results for each request
        final_results = []
        
        for request in requests:
            responses = request_results.get(request.id, [])
            self.validation_results[request.id] = responses
            
            if responses:
                cross_result = await self._synthesize_validation_results(request.id, responses)
                final_results.append(cross_result)
            else:
                # Create fallback result
                final_results.append(CrossValidationResult(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    consensus_level=0.0,
                    final_result=ValidationResult.INVALID,
                    conflicting_opinions=[],
                    synthesized_feedback="Batch validation failed - no valid responses",
                    confidence_score=0.0,
                    participating_validators=[],
                    timestamp=datetime.now()
                ))
        
        logging.info(f"Batch validation completed: {len(final_results)} results")
        return final_results
    
    def _select_validators(self, request: ValidationRequest) -> List[str]:
        """Select appropriate validators for the request"""
        
        # Start with required validators
        selected = []
        if request.required_validators:
            selected.extend([
                vid for vid in request.required_validators 
                if vid in self.validators
            ])
        
        # Add additional validators based on expertise relevance
        remaining_needed = request.minimum_validators - len(selected)
        
        if remaining_needed > 0:
            # Calculate relevance scores for all validators
            validator_scores = []
            
            for validator_id, validator in self.validators.items():
                if validator_id not in selected:
                    relevance = validator._calculate_expertise_relevance(
                        request.content, request.context
                    )
                    validator_scores.append((validator_id, relevance))
            
            # Sort by relevance and select top validators
            validator_scores.sort(key=lambda x: x[1], reverse=True)
            
            for validator_id, _ in validator_scores[:remaining_needed]:
                selected.append(validator_id)
        
        return selected
    
    async def _synthesize_validation_results(
        self, 
        request_id: str, 
        responses: List[ValidationResponse]
    ) -> CrossValidationResult:
        """Synthesize multiple validation responses into final result"""
        
        if not responses:
            raise ValueError("No validation responses to synthesize")
        
        # Calculate consensus level
        consensus_level = self._calculate_consensus(responses)
        
        # Determine final result
        final_result = self._determine_final_result(responses, consensus_level)
        
        # Identify conflicting opinions
        conflicting_opinions = self._identify_conflicts(responses)
        
        # Synthesize feedback
        synthesized_feedback = await self._synthesize_feedback(responses)
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(responses, consensus_level)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(responses, final_result)
        
        return CrossValidationResult(
            request_id=request_id,
            individual_responses=responses,
            consensus_level=consensus_level,
            final_result=final_result,
            conflicting_opinions=conflicting_opinions,
            synthesized_feedback=synthesized_feedback,
            confidence_score=confidence_score,
            recommendations=recommendations,
            created_at=datetime.now()
        )
    
    def _calculate_consensus(self, responses: List[ValidationResponse]) -> float:
        """Calculate consensus level among validators"""
        
        if len(responses) <= 1:
            return 1.0
        
        # Group responses by validation result
        result_counts = {}
        for response in responses:
            result = response.result
            result_counts[result] = result_counts.get(result, 0) + 1
        
        # Calculate consensus as proportion of most common result
        max_count = max(result_counts.values())
        consensus = max_count / len(responses)
        
        return consensus
    
    def _determine_final_result(
        self, 
        responses: List[ValidationResponse], 
        consensus_level: float
    ) -> ValidationResult:
        """Determine final validation result"""
        
        # Count results
        result_counts = {}
        confidence_weighted_results = {}
        
        for response in responses:
            result = response.result
            result_counts[result] = result_counts.get(result, 0) + 1
            
            if result not in confidence_weighted_results:
                confidence_weighted_results[result] = 0
            confidence_weighted_results[result] += response.confidence
        
        # If high consensus, use majority result
        if consensus_level >= CONFIG.consensus_threshold:
            return max(result_counts.items(), key=lambda x: x[1])[0]
        
        # If low consensus, use confidence-weighted result
        return max(confidence_weighted_results.items(), key=lambda x: x[1])[0]
    
    def _identify_conflicts(self, responses: List[ValidationResponse]) -> List[Tuple[str, str]]:
        """Identify conflicting opinions among validators"""
        
        conflicts = []
        results_by_validator = {}
        
        for response in responses:
            validator = response.validator_id
            result = response.result.value
            
            if validator not in results_by_validator:
                results_by_validator[validator] = []
            results_by_validator[validator].append(result)
        
        # Find validators with different results
        all_results = set()
        for results in results_by_validator.values():
            all_results.update(results)
        
        if len(all_results) > 1:
            for validator, results in results_by_validator.items():
                unique_results = set(results)
                if len(unique_results) > 1 or unique_results != all_results:
                    conflicts.append((validator, ', '.join(unique_results)))
        
        return conflicts
    
    async def _synthesize_feedback(self, responses: List[ValidationResponse]) -> str:
        """Synthesize feedback from multiple validators"""
        
        # Collect all reasoning and suggestions
        all_reasoning = [r.reasoning for r in responses if r.reasoning]
        all_improvements = []
        all_perspectives = []
        
        for response in responses:
            all_improvements.extend(response.suggested_improvements)
            all_perspectives.extend(response.alternative_perspectives)
        
        # Create synthesis prompt
        system_prompt = """
        Synthesize the feedback from multiple validators into a coherent summary.
        Highlight common themes, important disagreements, and actionable recommendations.
        """
        
        user_prompt = f"""
        Validator Reasoning:
        {chr(10).join([f"- {r}" for r in all_reasoning[:10]])}
        
        Suggested Improvements:
        {chr(10).join([f"- {i}" for i in set(all_improvements)[:10]])}
        
        Alternative Perspectives:
        {chr(10).join([f"- {p}" for p in set(all_perspectives)[:10]])}
        
        Synthesize this feedback into a coherent summary.
        """
        
        try:
            client = openai.OpenAI(api_key=CONFIG.openai_api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback to simple concatenation
            return f"Summary of {len(responses)} validator responses. Common themes: {', '.join(set(all_improvements[:3]))}"
    
    def _calculate_overall_confidence(
        self, 
        responses: List[ValidationResponse], 
        consensus_level: float
    ) -> float:
        """Calculate overall confidence score"""
        
        if not responses:
            return 0.0
        
        # Average confidence weighted by consensus
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Boost confidence if high consensus
        confidence_boost = consensus_level * 0.2
        
        return min(1.0, avg_confidence + confidence_boost)
    
    def _generate_recommendations(
        self, 
        responses: List[ValidationResponse], 
        final_result: ValidationResult
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Collect all suggested improvements
        all_improvements = []
        for response in responses:
            all_improvements.extend(response.suggested_improvements)
        
        # Remove duplicates and prioritize
        unique_improvements = list(set(all_improvements))
        
        if final_result == ValidationResult.REJECTED:
            recommendations.append("Major revision required based on validator feedback")
            recommendations.extend(unique_improvements[:3])
        elif final_result == ValidationResult.REQUIRES_REVISION:
            recommendations.append("Minor revisions recommended")
            recommendations.extend(unique_improvements[:2])
        elif final_result == ValidationResult.VALIDATED:
            recommendations.append("Content validated successfully")
            if unique_improvements:
                recommendations.append("Consider optional improvements for enhancement")
        else:
            recommendations.append("Additional information or validation required")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics"""
        
        total_requests = len(self.validation_requests)
        total_responses = sum(len(responses) for responses in self.validation_results.values())
        
        # Calculate result distribution
        result_counts = {}
        for responses in self.validation_results.values():
            for response in responses:
                result = response.result.value
                result_counts[result] = result_counts.get(result, 0) + 1
        
        return {
            "total_requests": total_requests,
            "total_responses": total_responses,
            "registered_validators": len(self.validators),
            "result_distribution": result_counts,
            "average_responses_per_request": total_responses / max(1, total_requests)
        }
