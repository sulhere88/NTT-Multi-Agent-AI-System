"""
Security utilities untuk Multi-Agent AI System
PII redaction, audit logging, dan security functions
"""

import re
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class PIIType(Enum):
    """Types of PII yang bisa dideteksi"""
    EMAIL = "email"
    PHONE = "phone"
    ID_NUMBER = "id_number"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    NAME = "name"  # Basic name detection

class SecurityLevel(Enum):
    """Security levels untuk audit"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PIIRedactor:
    """PII redaction utility"""
    
    def __init__(self):
        # Regex patterns untuk deteksi PII
        self.patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'(?:\+62|62|0)(?:\s?-?\s?)(?:\d{2,4})(?:\s?-?\s?)(?:\d{3,4})(?:\s?-?\s?)(?:\d{3,4})\b'),
            PIIType.ID_NUMBER: re.compile(r'\b(?:NIK|KTP|SIM|NPWP)?\s*:?\s*(\d{15,16})\b', re.IGNORECASE),
            PIIType.CREDIT_CARD: re.compile(r'\b(?:\d{4}[\s-]?){3}\d{4}\b'),
            PIIType.IP_ADDRESS: re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            PIIType.URL: re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'),
            # Basic name detection (Indonesian context)
            PIIType.NAME: re.compile(r'\b(?:Bapak|Ibu|Sdr|Sdri|Mr|Ms|Mrs|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        }
        
        # Replacement templates
        self.replacements = {
            PIIType.EMAIL: "[EMAIL_REDACTED]",
            PIIType.PHONE: "[PHONE_REDACTED]",
            PIIType.ID_NUMBER: "[ID_REDACTED]",
            PIIType.CREDIT_CARD: "[CARD_REDACTED]",
            PIIType.IP_ADDRESS: "[IP_REDACTED]",
            PIIType.URL: "[URL_REDACTED]",
            PIIType.NAME: r"\1 [NAME_REDACTED]"
        }
    
    def redact_text(self, text: str, pii_types: Optional[List[PIIType]] = None) -> str:
        """Redact PII dari text"""
        if not text:
            return text
        
        redacted_text = text
        pii_found = []
        
        # Default: redact semua jenis PII
        if pii_types is None:
            pii_types = list(PIIType)
        
        for pii_type in pii_types:
            if pii_type in self.patterns:
                pattern = self.patterns[pii_type]
                replacement = self.replacements[pii_type]
                
                # Count matches sebelum redact
                matches = pattern.findall(redacted_text)
                if matches:
                    pii_found.append({
                        "type": pii_type.value,
                        "count": len(matches)
                    })
                
                # Apply redaction
                redacted_text = pattern.sub(replacement, redacted_text)
        
        # Log PII detection untuk audit
        if pii_found:
            logging.info(f"PII detected and redacted: {pii_found}")
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any], sensitive_keys: List[str] = None) -> Dict[str, Any]:
        """Redact PII dari dictionary/JSON data"""
        if not isinstance(data, dict):
            return data
        
        # Default sensitive keys
        if sensitive_keys is None:
            sensitive_keys = [
                'email', 'phone', 'name', 'address', 'id', 'password',
                'token', 'secret', 'key', 'credential', 'user_data'
            ]
        
        redacted_data = {}
        
        for key, value in data.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                if isinstance(value, str):
                    redacted_data[key] = self.redact_text(value)
                else:
                    redacted_data[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted_data[key] = self.redact_dict(value, sensitive_keys)
            elif isinstance(value, list):
                redacted_data[key] = [
                    self.redact_dict(item, sensitive_keys) if isinstance(item, dict)
                    else self.redact_text(str(item)) if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                redacted_data[key] = self.redact_text(value)
            else:
                redacted_data[key] = value
        
        return redacted_data

class AuditLogger:
    """Structured audit logging"""
    
    def __init__(self):
        self.redactor = PIIRedactor()
        
        # Setup audit logger
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Audit file handler
        audit_handler = logging.FileHandler('logs/audit.log')
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
    
    def log_agent_decision(
        self,
        agent_id: str,
        decision_type: str,
        decision_data: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.MEDIUM
    ):
        """Log keputusan penting agent"""
        
        # Redact PII dari decision data
        redacted_data = self.redactor.redact_dict(decision_data)
        
        audit_entry = {
            "event_type": "agent_decision",
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "decision_type": decision_type,
            "security_level": security_level.value,
            "data": redacted_data,
            "event_id": hashlib.md5(f"{agent_id}{decision_type}{datetime.now()}".encode()).hexdigest()[:16]
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def log_validation_result(
        self,
        validator_id: str,
        validation_type: str,
        result: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.HIGH
    ):
        """Log hasil validasi"""
        
        redacted_result = self.redactor.redact_dict(result)
        
        audit_entry = {
            "event_type": "validation_result",
            "timestamp": datetime.now().isoformat(),
            "validator_id": validator_id,
            "validation_type": validation_type,
            "security_level": security_level.value,
            "result": redacted_result,
            "event_id": hashlib.md5(f"{validator_id}{validation_type}{datetime.now()}".encode()).hexdigest()[:16]
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def log_memory_operation(
        self,
        agent_id: str,
        operation: str,
        memory_type: str,
        content_summary: str,
        security_level: SecurityLevel = SecurityLevel.LOW
    ):
        """Log operasi memori"""
        
        redacted_summary = self.redactor.redact_text(content_summary)
        
        audit_entry = {
            "event_type": "memory_operation",
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "operation": operation,
            "memory_type": memory_type,
            "security_level": security_level.value,
            "content_summary": redacted_summary,
            "event_id": hashlib.md5(f"{agent_id}{operation}{datetime.now()}".encode()).hexdigest()[:16]
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def log_collaboration_event(
        self,
        initiator_agent: str,
        collaborator_agents: List[str],
        task_description: str,
        security_level: SecurityLevel = SecurityLevel.MEDIUM
    ):
        """Log event kolaborasi antar agent"""
        
        redacted_description = self.redactor.redact_text(task_description)
        
        audit_entry = {
            "event_type": "collaboration",
            "timestamp": datetime.now().isoformat(),
            "initiator_agent": initiator_agent,
            "collaborator_agents": collaborator_agents,
            "security_level": security_level.value,
            "task_description": redacted_description,
            "event_id": hashlib.md5(f"{initiator_agent}{str(collaborator_agents)}{datetime.now()}".encode()).hexdigest()[:16]
        }
        
        self.audit_logger.info(json.dumps(audit_entry))

# Global instances
pii_redactor = PIIRedactor()
audit_logger = AuditLogger()

def redact_log_message(message: str) -> str:
    """Utility function untuk redact log messages"""
    return pii_redactor.redact_text(message)

def log_secure_event(event_type: str, data: Dict[str, Any], security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Utility function untuk secure logging"""
    redacted_data = pii_redactor.redact_dict(data)
    
    audit_entry = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "security_level": security_level.value,
        "data": redacted_data,
        "event_id": hashlib.md5(f"{event_type}{str(data)}{datetime.now()}".encode()).hexdigest()[:16]
    }
    
    audit_logger.audit_logger.info(json.dumps(audit_entry))

# Test function
if __name__ == "__main__":
    # Test PII redaction
    test_text = """
    Halo, saya John Doe dengan email john.doe@example.com 
    dan nomor telepon 081234567890. 
    NIK saya 1234567890123456 dan alamat IP 192.168.1.1.
    """
    
    redacted = pii_redactor.redact_text(test_text)
    print("Original:", test_text)
    print("Redacted:", redacted)
    
    # Test audit logging
    audit_logger.log_agent_decision(
        "agent_001",
        "task_assignment",
        {"task": "Analyze customer data", "customer_email": "customer@example.com"},
        SecurityLevel.HIGH
    )
