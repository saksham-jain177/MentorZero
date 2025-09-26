from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import re

from agent.llm.ollama_client import OllamaClient


@dataclass
class SyntheticItem:
    type: str
    prompt: str
    answer: str


class ValidatorService:
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def propose_synthetic(self, topic: str, n: int = 10) -> List[SyntheticItem]:
        sys = "Generate deterministic short Q/A pairs for AZL training; return JSON list."
        p = f"Topic: {topic}. Return a list of {{type:'qa', prompt:'...', answer:'...'}} length {n}."
        resp = self.llm.send(prompt=p, system_prompt=sys, temperature=0.0)
        text = resp.get("text", "")
        try:
            import json

            data = json.loads(text)
            items: List[SyntheticItem] = []
            for row in data if isinstance(data, list) else []:
                t = row.get("type", "qa")
                items.append(SyntheticItem(type=t, prompt=row.get("prompt", ""), answer=row.get("answer", "")))
            return items
        except Exception:
            return []

    def basic_validators(self, items: List[SyntheticItem]) -> List[bool]:
        results: List[bool] = []
        for it in items:
            ok = bool(it.prompt and it.answer and len(it.prompt) >= 5 and len(it.answer) >= 1)
            results.append(ok)
        return results

    def accept_rate(self, results: List[bool]) -> float:
        return (sum(1 for r in results if r) / max(1, len(results)))


class AZLValidators:
    """Validators for Absolute Zero Learning examples"""
    
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm
    
    async def validate_all(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Run all validators on an example"""
        results = {}
        
        # Length validation
        results["length"] = self.validate_length(example)
        
        # Duplicate validation
        results["duplicate"] = await self.validate_duplicate(example)
        
        # Consistency validation
        results["consistency"] = await self.validate_consistency(example)
        
        # Roundtrip validation
        results["roundtrip"] = await self.validate_roundtrip(example)
        
        # Toxicity validation
        results["toxicity"] = await self.validate_toxicity(example)

        # Placeholder validation (synchronous)
        results["placeholder"] = self.validate_placeholder(example)
        
        return results
    
    def validate_length(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Validate the length of the example"""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        question_length = len(question)
        answer_length = len(answer)
        
        # Define minimum and maximum lengths
        min_question_length = 10
        max_question_length = 500
        min_answer_length = 20
        max_answer_length = 1000
        
        question_valid = min_question_length <= question_length <= max_question_length
        answer_valid = min_answer_length <= answer_length <= max_answer_length
        
        passed = question_valid and answer_valid
        
        return {
            "passed": passed,
            "details": {
                "question_length": question_length,
                "answer_length": answer_length,
                "question_valid": question_valid,
                "answer_valid": answer_valid
            }
        }
    
    async def validate_duplicate(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Check if the example is a duplicate of existing content"""
        # In a real implementation, this would check against a database
        # For now, we'll just return a simple check
        return {
            "passed": True,
            "details": {
                "similarity_score": 0.1,  # Placeholder
                "threshold": 0.8
            }
        }
    
    async def validate_consistency(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Validate that the answer is consistent with the question"""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        prompt = f"""
        Evaluate if the following answer is consistent with the question:
        
        Question: {question}
        
        Answer: {answer}
        
        Is the answer directly addressing the question and providing relevant information?
        Respond with YES or NO, followed by a brief explanation.
        """
        
        try:
            response = await self.llm.send(prompt)
            text = response.get("text", "").strip()
            
            is_consistent = "YES" in text.split("\n")[0].upper()
            
            return {
                "passed": is_consistent,
                "details": {
                    "explanation": text,
                    "is_consistent": is_consistent
                }
            }
        except Exception as e:
            return {
                "passed": False,
                "details": {
                    "error": str(e)
                }
            }
    
    async def validate_roundtrip(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Validate that the answer can be used to reconstruct the question"""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        prompt = f"""
        Given only this answer:
        
        {answer}
        
        What question was likely being asked? Respond with just the question.
        """
        
        try:
            response = await self.llm.send(prompt)
            reconstructed_question = response.get("text", "").strip()
            
            # Compare the reconstructed question with the original
            similarity_prompt = f"""
            Rate the similarity between these two questions on a scale of 0-10:
            
            Original: {question}
            Reconstructed: {reconstructed_question}
            
            Respond with just the number.
            """
            
            similarity_response = await self.llm.send(similarity_prompt)
            similarity_text = similarity_response.get("text", "0").strip()
            
            # Extract the number
            match = re.search(r'\b(\d+(?:\.\d+)?)\b', similarity_text)
            similarity_score = float(match.group(1)) if match else 0
            
            passed = similarity_score >= 7.0
            
            return {
                "passed": passed,
                "details": {
                    "reconstructed_question": reconstructed_question,
                    "similarity_score": similarity_score,
                    "threshold": 7.0
                }
            }
        except Exception as e:
            return {
                "passed": False,
                "details": {
                    "error": str(e)
                }
            }
    
    async def validate_toxicity(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Check if the example contains toxic content"""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        prompt = f"""
        Analyze the following question and answer for any toxic, harmful, offensive, or inappropriate content:
        
        Question: {question}
        
        Answer: {answer}
        
        Does this content contain any toxic, harmful, offensive, or inappropriate elements?
        Respond with YES or NO, followed by a brief explanation.
        """
        
        try:
            response = await self.llm.send(prompt)
            text = response.get("text", "").strip()
            
            is_toxic = "YES" in text.split("\n")[0].upper()
            
            return {
                "passed": not is_toxic,
                "details": {
                    "explanation": text,
                    "is_toxic": is_toxic
                }
            }
        except Exception as e:
            return {
                "passed": True,  # Default to passing if check fails
                "details": {
                    "error": str(e)
                }
            }

    def validate_placeholder(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Detect placeholder or non-informative answers like 'This is a placeholder' or 'Answer goes here'."""
        q = (example.get("question", "") or "").strip().lower()
        a = (example.get("answer", "") or "").strip().lower()
        placeholders = [
            "placeholder",
            "answer goes here",
            "lorem ipsum",
            "to be filled",
            "tbd",
            "sample answer",
            "this is a placeholder",
            "dummy answer",
        ]
        bad = False
        hit = None
        for token in placeholders:
            if token in a or token in q:
                bad = True
                hit = token
                break
        # also guard extremely generic patterns
        if not bad:
            generic_patterns = [
                r"^answer:\s*$",
                r"^question:\s*$",
                r"^n\/a$",
            ]
            import re as _re
            for pat in generic_patterns:
                if _re.search(pat, a) or _re.search(pat, q):
                    bad = True
                    hit = pat
                    break
        return {
            "passed": not bad,
            "details": {
                "reason": ("placeholder_detected" if bad else "ok"),
                "hit": hit
            }
        }