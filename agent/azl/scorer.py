from __future__ import annotations

from typing import Any, Dict

from agent.services.validators import AZLValidators


class AZLScorer:
    """Aggregate AZL validation with an LLM-as-a-judge score into one final score."""

    def __init__(self, ollama_client, weights: Dict[str, float], judge_client=None):
        # Generator client (used by validators that may need LLM)
        self.ollama = ollama_client
        # Dedicated judge client if provided; otherwise fall back to generator
        self.judge = judge_client or ollama_client
        self.weights = weights or {}

    async def score(self, example: Dict[str, str]) -> Dict[str, Any]:
        """
        Returns a dict:
        {
          checks: {name: {passed, details}},
          judge: {correctness: float, clarity: float, usefulness: float, rationale: str},
          score: float,
          passed: bool
        }
        """
        validators = AZLValidators(self.ollama)
        checks = await validators.validate_all(example)

        # Convert boolean checks into scores 0/1
        check_scores: Dict[str, float] = {}
        for name, result in checks.items():
            check_scores[name] = 1.0 if result.get("passed", False) else 0.0

        # LLM-as-a-judge
        judge = await self._judge(example)

        # Weighted aggregate
        total = 0.0
        weight_sum = 0.0
        for name, val in check_scores.items():
            w = float(self.weights.get(name, 0.0))
            total += w * val
            weight_sum += w

        judge_weight = float(self.weights.get("judge", 0.0))
        judge_avg = (judge["correctness"] + judge["clarity"] + judge["usefulness"]) / 3.0
        total += judge_weight * judge_avg
        weight_sum += judge_weight

        final_score = total / max(1e-9, weight_sum)

        return {
            "checks": checks,
            "judge": judge,
            "score": final_score,
            "passed": final_score >= 0.0,  # pass/fail threshold decided by caller
        }

    async def _judge(self, example: Dict[str, str]) -> Dict[str, float]:
        system_prompt = (
            "You are a strict educational quality judge. Score a Q/A pair for tutoring use. "
            "Output ONLY JSON with fields: correctness (0..1), clarity (0..1), usefulness (0..1), rationale (string)."
        )
        prompt = (
            f"Question: {example.get('question','')}\n"
            f"Answer: {example.get('answer','')}\n\n"
            "Return compact JSON."
        )
        try:
            resp = await self.judge.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            text = resp.get("text", "{}")
            import json, re
            m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            payload = m.group(1) if m else text
            obj = json.loads(payload)
            c = float(obj.get("correctness", 0))
            l = float(obj.get("clarity", 0))
            u = float(obj.get("usefulness", 0))
            rat = str(obj.get("rationale", ""))
            # clamp
            def clamp01(x: float) -> float:
                return max(0.0, min(1.0, x))
            return {"correctness": clamp01(c), "clarity": clamp01(l), "usefulness": clamp01(u), "rationale": rat}
        except Exception as e:
            # Provide detailed exception info so UI can display something meaningful
            try:
                import traceback
                etype = type(e).__name__
                msg = str(e) or repr(e)
                tb_short = " | ".join([line.strip() for line in traceback.format_exception_only(type(e), e)])
                detail = msg or tb_short or etype
            except Exception:
                detail = repr(e)
            return {"correctness": 0.0, "clarity": 0.0, "usefulness": 0.0, "rationale": f"judge_error: {detail}"}
