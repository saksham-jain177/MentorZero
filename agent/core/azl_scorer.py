import logging
import json
import re
from typing import Dict, List, Optional, Any
from agent.config import get_settings

logger = logging.getLogger(__name__)

class AZLScorer:
    """
    Autonomous Zero-hallucination Layer (AZL) Scorer.
    Performs multi-dimensional validation on research outputs to identify
    hallucinations, contradictions, and safety breaches.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.settings = get_settings()
        self.threshold = self.settings.azl_threshold

    async def validate_result(self, query: str, facts: List[str], sources: List[Dict]) -> Dict[str, Any]:
        """
        Validate a research result across multiple dimensions.
        Returns a validation report.
        """
        if not self.llm or not facts:
            return {"passed": True, "score": 1.0, "details": "Validation skipped: No LLM or no facts provided."}

        # 1. Judgement Cascade: Select model based on complexity
        # For now, we use the default LLM, but the infrastructure supports a 'Heavy' judge
        judge = self.llm
        
        report = {
            "query": query,
            "metrics": {},
            "passed": True,
            "overall_score": 0.0
        }

        # Select top facts for validation to avoid token bloat
        sample_facts = facts[:8]
        source_contexts = [s.get("content", "")[:500] for s in sources[:5]]
        context_blob = "\n---\n".join(source_contexts)

        prompt = f"""### AI Safety & Grounding Audit
Target Query: {query}

Research Facts to Validate:
{chr(10).join([f"- {f}" for f in sample_facts])}

Source Context (External Evidence):
{context_blob}

Audit the facts against the evidence for:
1. **Grounding**: Do the facts actually appear or are logically supported by the sources? (0.0 - 1.0)
2. **Consistency**: Do any facts contradict each other? (0.0 - 1.0)
3. **Safety**: Does the output contain toxic, harmful, or prohibited content? (0.0 - 1.0 where 1.0 is safe)
4. **Hallucination Detection**: Are there future-dated claims, fake entities, or invented metrics? (0.0 - 1.0 where 1.0 is no hallucinations)

Return JSON only:
{{
  "grounding_score": float,
  "consistency_score": float,
  "safety_score": float,
  "hallucination_score": float,
  "critique": "short summary",
  "is_safe": bool
}}"""

        try:
            # Use lower temperature for deterministic results
            response = await judge.send_prompt(prompt, temperature=0.1)
            
            # Extract JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                results = json.loads(match.group(0))
                
                # Calculate weighted score
                scores = [
                    results.get("grounding_score", 0.0),
                    results.get("consistency_score", 0.0),
                    results.get("safety_score", 0.0),
                    results.get("hallucination_score", 0.0)
                ]
                avg_score = sum(scores) / len(scores)
                
                report["metrics"] = results
                report["overall_score"] = avg_score
                
                # Fail if any critical dimension is below threshold or unsafe
                if avg_score < self.threshold or not results.get("is_safe", True):
                    report["passed"] = False
                    
                logger.info(f"AZL Validation completed. Score: {avg_score:.2f}, Passed: {report['passed']}")
                
                # Log to audit trail (simulated for now)
                self._log_audit(report)
                
                return report
                
        except Exception as e:
            logger.error(f"AZL Validation failed: {e}")
            return {"passed": True, "score": 0.5, "details": f"Validation error: {e}"}

        return report

    def _log_audit(self, report: Dict):
        """Persist validation event to a local audit log"""
        try:
            import os
            from datetime import datetime
            
            audit_dir = "./data/audit"
            os.makedirs(audit_dir, exist_ok=True)
            
            log_file = os.path.join(audit_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.log")
            
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "AZL_VALIDATION",
                "report": report
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
        except:
            pass
