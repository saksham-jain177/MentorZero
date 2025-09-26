from typing import Dict, List, Tuple, Any, Optional
import uuid
import json
import re

from agent.llm.ollama_client import OllamaClient

# In-memory storage for examples (would be DB in production)
_PROPOSALS = {}

class AZLGenerator:
    """Generator for Absolute Zero Learning examples"""
    
    def __init__(self, llm: OllamaClient):
        self.llm = llm
    
    async def propose_examples(self, topic: str, count: int = 5) -> Tuple[List[Dict[str, str]], str]:
        """Generate synthetic examples for a topic with robust parsing."""
        system_prompt = (
            "You generate educational Q/A pairs. Return ONLY a JSON array of objects with exactly two keys: "
            "question (string) and answer (string). No prose, no code fences, no extra keys."
        )
        prompt = (
            f"Generate {count} high-quality, diverse question-answer pairs about: {topic}. "
            "Rules: (1) No placeholders or meta text; (2) Keep answers factual, concise, and directly address the question; "
            "(3) Avoid redundancy across items; (4) Return JSON only."
        )
        
        try:
            response = await self.llm.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            text = response.get("text", "")

            examples = self._parse_examples(text, topic, count)
            if not examples:
                return self._generate_fallback_examples(topic, count)
            
            # Store the examples with a unique ID
            proposal_id = str(uuid.uuid4())
            _PROPOSALS[proposal_id] = examples
            
            return examples, proposal_id
        
        except Exception as e:
            print(f"Error generating examples: {e}")
            return self._generate_fallback_examples(topic, count)
    
    def _parse_examples(self, text: str, topic: str, count: int) -> List[Dict[str, str]]:
        """Coerce LLM output into a list of {question, answer} items."""
        cleaned = text.replace("\r", "")
        # Prefer JSON code block
        m = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned)
        payload = m.group(1) if m else cleaned
        # Remove trailing commas
        payload = re.sub(r",\s*([}\]])", r"\1", payload)
        # Try parse as list
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                return self._normalize_list(data)
        except Exception:
            pass
        # Try first array in text
        m2 = re.search(r"\[([\s\S]*?)\]", payload)
        if m2:
            candidate = "[" + m2.group(1) + "]"
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return self._normalize_list(data)
            except Exception:
                pass
        # Fallback: parse Q:/A: pairs
        pairs: List[Dict[str, str]] = []
        qa_matches = re.findall(r"Q\s*[:\-]\s*(.*?)\s*A\s*[:\-]\s*(.*?)(?=\n\s*Q\s*[:\-]|$)", cleaned, flags=re.IGNORECASE | re.DOTALL)
        for q, a in qa_matches:
            q2 = q.strip()
            a2 = a.strip()
            if q2 and a2:
                pairs.append({"question": q2, "answer": a2})
        return pairs[:count]

    def _normalize_list(self, data: List[Any]) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        for row in data:
            if isinstance(row, dict):
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if q and a:
                    items.append({"question": q, "answer": a})
        return items
    
    def _generate_fallback_examples(self, topic: str, count: int) -> Tuple[List[Dict[str, str]], str]:
        """Generate fallback examples if JSON parsing fails"""
        examples = []
        for i in range(count):
            examples.append({
                "question": f"Question {i+1} about {topic}?",
                "answer": f"This is a placeholder answer about {topic}."
            })
        
        proposal_id = str(uuid.uuid4())
        _PROPOSALS[proposal_id] = examples
        
        return examples, proposal_id
    
    @staticmethod
    def get_example(proposal_id: str, example_idx: int) -> Optional[Dict[str, str]]:
        """Get a specific example from a proposal"""
        if proposal_id not in _PROPOSALS:
            return None
        
        examples = _PROPOSALS[proposal_id]
        if example_idx < 0 or example_idx >= len(examples):
            return None
        
        return examples[example_idx]

    @staticmethod
    def set_example(proposal_id: str, example_idx: int, example: Dict[str, str]) -> bool:
        """Replace a specific example in a proposal."""
        if proposal_id not in _PROPOSALS:
            return False
        examples = _PROPOSALS[proposal_id]
        if example_idx < 0 or example_idx >= len(examples):
            return False
        examples[example_idx] = example
        _PROPOSALS[proposal_id] = examples
        return True

    @staticmethod
    def parse_single_example(text: str) -> Optional[Dict[str, str]]:
        """Parse a single QA from possibly messy text."""
        t = text.replace("\r", "")
        t = re.sub(r",\s*([}\]])", r"\1", t)
        # JSON object
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                return {"question": str(obj["question"]).strip(), "answer": str(obj["answer"]).strip()}
            # If it's a list, take first valid object
            if isinstance(obj, list):
                for row in obj:
                    if isinstance(row, dict) and "question" in row and "answer" in row:
                        return {"question": str(row["question"]).strip(), "answer": str(row["answer"]).strip()}
        except Exception:
            pass
        m = re.search(r"```json\s*([\s\S]*?)\s*```", t)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                    return {"question": str(obj["question"]).strip(), "answer": str(obj["answer"]).strip()}
                if isinstance(obj, list):
                    for row in obj:
                        if isinstance(row, dict) and "question" in row and "answer" in row:
                            return {"question": str(row["question"]).strip(), "answer": str(row["answer"]).strip()}
            except Exception:
                pass
        # Q/A fallback
        m2 = re.search(r"Q\s*[:\-]\s*(.*?)\s*A\s*[:\-]\s*(.*)", t, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            q = m2.group(1).strip()
            a = m2.group(2).strip()
            if q and a:
                return {"question": q, "answer": a}
        return None