"""
Audit logging utilities for tracking important system events.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a dedicated logger for auditing
audit_logger = logging.getLogger("mentorzero.audit")

# Set up file handler if audit directory exists
AUDIT_DIR = os.environ.get("MZ_AUDIT_DIR", "data/audit")
Path(AUDIT_DIR).mkdir(parents=True, exist_ok=True)

# Add file handler
file_handler = logging.FileHandler(f"{AUDIT_DIR}/audit_{datetime.now().strftime('%Y%m%d')}.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
audit_logger.addHandler(file_handler)

# Event types
EVENT_STRATEGY_CHANGE = "strategy_change"
EVENT_AZL_PROPOSAL = "azl_proposal"
EVENT_AZL_VALIDATION = "azl_validation"
EVENT_AZL_APPROVAL = "azl_approval"
EVENT_MEMORY_RULE_UPDATE = "memory_rule_update"
EVENT_MEMORY_RULE_ROLLBACK = "memory_rule_rollback"

def log_strategy_change(
    session_id: str,
    user_id: str,
    old_strategy: str,
    new_strategy: str,
    reason: str,
    topic: str,
    retry_count: int,
) -> None:
    """
    Log a teaching strategy change.
    
    Args:
        session_id: The session ID
        user_id: The user ID
        old_strategy: The previous strategy
        new_strategy: The new strategy
        reason: Why the strategy was changed
        topic: The topic being taught
        retry_count: Number of retries that triggered the change
    """
    event_data = {
        "event_type": EVENT_STRATEGY_CHANGE,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "old_strategy": old_strategy,
        "new_strategy": new_strategy,
        "reason": reason,
        "topic": topic,
        "retry_count": retry_count,
    }
    
    # Log to audit log
    audit_logger.info(f"STRATEGY_CHANGE: {json.dumps(event_data)}")
    
    # Also save to strategy changes file
    strategy_file = f"{AUDIT_DIR}/strategy_changes.jsonl"
    with open(strategy_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")

def log_azl_proposal(
    proposal_id: str,
    topic: str,
    user_id: Optional[str],
    example_count: int,
) -> None:
    """
    Log an AZL proposal generation event.
    
    Args:
        proposal_id: The proposal ID
        topic: The topic for which examples were generated
        user_id: The user ID (if available)
        example_count: Number of examples generated
    """
    event_data = {
        "event_type": EVENT_AZL_PROPOSAL,
        "timestamp": datetime.now().isoformat(),
        "proposal_id": proposal_id,
        "topic": topic,
        "user_id": user_id,
        "example_count": example_count,
    }
    
    # Log to audit log
    audit_logger.info(f"AZL_PROPOSAL: {json.dumps(event_data)}")
    
    # Also save to AZL events file
    azl_file = f"{AUDIT_DIR}/azl_events.jsonl"
    with open(azl_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")

def log_azl_validation(
    proposal_id: str,
    example_idx: int,
    validation_results: Dict[str, Any],
    passed: bool,
) -> None:
    """
    Log an AZL validation event.
    
    Args:
        proposal_id: The proposal ID
        example_idx: The example index
        validation_results: Results of validation checks
        passed: Whether the example passed all validations
    """
    event_data = {
        "event_type": EVENT_AZL_VALIDATION,
        "timestamp": datetime.now().isoformat(),
        "proposal_id": proposal_id,
        "example_idx": example_idx,
        "validation_results": validation_results,
        "passed": passed,
    }
    
    # Log to audit log
    audit_logger.info(f"AZL_VALIDATION: {json.dumps(event_data)}")
    
    # Also save to AZL events file
    azl_file = f"{AUDIT_DIR}/azl_events.jsonl"
    with open(azl_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")

def log_azl_approval(
    proposal_id: str,
    example_idx: int,
    user_id: str,
    accepted: bool,
    message: str,
) -> None:
    """
    Log an AZL example approval or rejection.
    
    Args:
        proposal_id: The proposal ID
        example_idx: The example index
        user_id: The user who approved/rejected
        accepted: Whether the example was accepted
        message: Message explaining the decision
    """
    event_data = {
        "event_type": EVENT_AZL_APPROVAL,
        "timestamp": datetime.now().isoformat(),
        "proposal_id": proposal_id,
        "example_idx": example_idx,
        "user_id": user_id,
        "accepted": accepted,
        "message": message,
    }
    
    # Log to audit log
    audit_logger.info(f"AZL_APPROVAL: {json.dumps(event_data)}")
    
    # Also save to AZL events file
    azl_file = f"{AUDIT_DIR}/azl_events.jsonl"
    with open(azl_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")

def log_memory_rule_update(
    rule_id: str,
    user_id: str,
    strategy: str,
    old_success_rate: float,
    new_success_rate: float,
) -> None:
    """
    Log a memory rule update.
    
    Args:
        rule_id: The rule ID
        user_id: The user ID
        strategy: The strategy name
        old_success_rate: Previous success rate
        new_success_rate: New success rate
    """
    event_data = {
        "event_type": EVENT_MEMORY_RULE_UPDATE,
        "timestamp": datetime.now().isoformat(),
        "rule_id": rule_id,
        "user_id": user_id,
        "strategy": strategy,
        "old_success_rate": old_success_rate,
        "new_success_rate": new_success_rate,
        "delta": new_success_rate - old_success_rate,
    }
    
    # Log to audit log
    audit_logger.info(f"MEMORY_RULE_UPDATE: {json.dumps(event_data)}")
    
    # Also save to memory rules file
    rules_file = f"{AUDIT_DIR}/memory_rules.jsonl"
    with open(rules_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")

def get_memory_rule_history(rule_id: str) -> List[Dict[str, Any]]:
    """
    Get the history of updates to a memory rule.
    
    Args:
        rule_id: The rule ID
    
    Returns:
        List of update events for the rule
    """
    rules_file = f"{AUDIT_DIR}/memory_rules.jsonl"
    history = []
    
    if not os.path.exists(rules_file):
        return history
    
    with open(rules_file, "r") as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                if event.get("rule_id") == rule_id:
                    history.append(event)
            except json.JSONDecodeError:
                continue
    
    return sorted(history, key=lambda x: x.get("timestamp", ""))

def rollback_memory_rule(
    rule_id: str,
    user_id: str,
    reason: str,
    target_timestamp: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find a previous state of a memory rule for rollback.
    
    Args:
        rule_id: The rule ID
        user_id: User requesting the rollback
        reason: Reason for rollback
        target_timestamp: Optional timestamp to roll back to (ISO format)
                         If None, rolls back to previous state
    
    Returns:
        The state to roll back to, or None if not found
    """
    history = get_memory_rule_history(rule_id)
    
    if not history:
        return None
    
    # If target timestamp provided, find the most recent state before that time
    if target_timestamp:
        valid_states = [
            state for state in history 
            if state.get("timestamp", "") < target_timestamp
        ]
        
        if not valid_states:
            return None
        
        target_state = max(valid_states, key=lambda x: x.get("timestamp", ""))
    else:
        # Otherwise, get the second most recent state (previous to current)
        if len(history) < 2:
            return None
        
        # Sort by timestamp descending
        sorted_history = sorted(
            history, 
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        target_state = sorted_history[1]  # Second most recent
    
    # Log the rollback
    event_data = {
        "event_type": EVENT_MEMORY_RULE_ROLLBACK,
        "timestamp": datetime.now().isoformat(),
        "rule_id": rule_id,
        "user_id": user_id,
        "reason": reason,
        "target_timestamp": target_state.get("timestamp"),
        "target_success_rate": target_state.get("new_success_rate"),
    }
    
    # Log to audit log
    audit_logger.info(f"MEMORY_RULE_ROLLBACK: {json.dumps(event_data)}")
    
    # Also save to memory rules file
    rules_file = f"{AUDIT_DIR}/memory_rules.jsonl"
    with open(rules_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")
    
    return target_state
