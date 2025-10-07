"""
Extended Agent Capabilities
Beyond just research - actual useful features
"""
from typing import Dict, List, Any, Optional
import asyncio
import json
import re
from datetime import datetime


class CodeGenerationAgent:
    """Generates, analyzes, and improves code"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def generate_code(self, requirements: str, language: str = "python") -> Dict:
        """Generate code from requirements"""
        prompt = f"""Generate {language} code for: {requirements}
        
Include:
- Clean, documented code
- Error handling
- Example usage
- Dependencies needed

Return only the code."""
        
        if self.llm:
            code = await self.llm.send_prompt(prompt)
            return {
                "language": language,
                "code": code,
                "requirements": requirements,
                "timestamp": datetime.now().isoformat()
            }
        
        # Mock response for testing
        return {
            "language": language,
            "code": f"# Code for: {requirements}\n# Implementation here",
            "requirements": requirements
        }
    
    async def analyze_code(self, code: str) -> Dict:
        """Analyze code for issues, improvements"""
        analysis = {
            "complexity": self._calculate_complexity(code),
            "issues": [],
            "improvements": [],
            "security": []
        }
        
        # Basic analysis
        lines = code.split('\n')
        
        # Check for common issues
        if 'eval(' in code or 'exec(' in code:
            analysis["security"].append("Dangerous eval/exec usage detected")
        
        if not any('try:' in line for line in lines):
            analysis["issues"].append("No error handling found")
        
        if not any(line.strip().startswith('#') for line in lines):
            analysis["issues"].append("No comments/documentation")
        
        # Complexity check
        if analysis["complexity"] > 10:
            analysis["improvements"].append("Consider breaking into smaller functions")
        
        return analysis
    
    async def refactor_code(self, code: str, style: str = "clean") -> str:
        """Refactor code for better quality"""
        if not self.llm:
            return code
        
        prompt = f"""Refactor this code to be {style}:
{code}

Improve:
- Readability
- Performance  
- Error handling
- Documentation"""
        
        return await self.llm.send_prompt(prompt)
    
    def _calculate_complexity(self, code: str) -> int:
        """Simple cyclomatic complexity calculation"""
        complexity = 1
        keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'except:', 'with ']
        for keyword in keywords:
            complexity += code.count(keyword)
        return complexity


class LearningAgent:
    """Creates personalized learning paths and materials"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.learning_styles = ["visual", "practical", "theoretical", "interactive"]
    
    async def create_learning_path(self, topic: str, level: str = "beginner") -> Dict:
        """Create structured learning path"""
        path = {
            "topic": topic,
            "level": level,
            "modules": [],
            "estimated_time": "2-4 weeks",
            "resources": []
        }
        
        # Define learning modules
        if level == "beginner":
            path["modules"] = [
                {"week": 1, "focus": "Fundamentals", "topics": [f"What is {topic}?", "Basic concepts", "Terminology"]},
                {"week": 2, "focus": "Core Skills", "topics": ["Hands-on basics", "Common patterns", "Best practices"]},
                {"week": 3, "focus": "Practice", "topics": ["Build projects", "Common mistakes", "Debugging"]},
                {"week": 4, "focus": "Advanced", "topics": ["Advanced concepts", "Real-world applications", "Next steps"]}
            ]
        
        return path
    
    async def generate_quiz(self, topic: str, difficulty: str = "medium") -> Dict:
        """Generate quiz questions"""
        questions = []
        
        if self.llm:
            prompt = f"""Create 5 {difficulty} quiz questions about {topic}.
Format: Question | A) option | B) option | C) option | D) option | Answer"""
            
            response = await self.llm.send_prompt(prompt)
            # Parse response into questions
            
        # Mock questions for testing
        questions = [
            {
                "question": f"What is the main purpose of {topic}?",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "answer": "A",
                "explanation": "This is the fundamental concept."
            }
        ]
        
        return {
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
            "total": len(questions)
        }
    
    async def explain_like_im_five(self, concept: str) -> str:
        """Explain complex concepts simply"""
        if self.llm:
            prompt = f"""Explain {concept} like I'm 5 years old.
Use simple words, analogies, and examples a child would understand."""
            return await self.llm.send_prompt(prompt)
        
        return f"{concept} is like a toy that helps grown-ups do their work faster!"


class AnalysisAgent:
    """Analyzes data, documents, and provides insights"""
    
    def __init__(self):
        self.analysis_types = ["sentiment", "summary", "key_points", "comparison"]
    
    async def analyze_text(self, text: str, analysis_type: str = "summary") -> Dict:
        """Analyze text for various insights"""
        result = {
            "type": analysis_type,
            "text_length": len(text),
            "word_count": len(text.split())
        }
        
        if analysis_type == "summary":
            result["summary"] = self._summarize(text)
        elif analysis_type == "sentiment":
            result["sentiment"] = self._analyze_sentiment(text)
        elif analysis_type == "key_points":
            result["key_points"] = self._extract_key_points(text)
        
        return result
    
    def _summarize(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization"""
        sentences = text.split('. ')
        if len(sentences) <= max_sentences:
            return text
        
        # Return first and last sentences as simple summary
        summary = f"{sentences[0]}. ... {sentences[-1]}"
        return summary
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'poor', 'worst']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_signals": positive_count,
            "negative_signals": negative_count
        }
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        sentences = text.split('. ')
        key_points = []
        
        # Look for sentences with key indicators
        indicators = ['important', 'key', 'main', 'critical', 'essential', 'must', 'should']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in indicators):
                key_points.append(sentence.strip())
        
        # If no indicators found, return first 3 sentences
        if not key_points and sentences:
            key_points = sentences[:3]
        
        return key_points


class AutomationAgent:
    """Automates workflows and repetitive tasks"""
    
    def __init__(self):
        self.workflows = {}
    
    async def create_workflow(self, name: str, steps: List[Dict]) -> Dict:
        """Create an automation workflow"""
        workflow = {
            "name": name,
            "steps": steps,
            "created": datetime.now().isoformat(),
            "status": "ready"
        }
        
        self.workflows[name] = workflow
        return workflow
    
    async def execute_workflow(self, name: str, context: Dict = None) -> Dict:
        """Execute a saved workflow"""
        if name not in self.workflows:
            return {"error": f"Workflow '{name}' not found"}
        
        workflow = self.workflows[name]
        results = []
        
        for i, step in enumerate(workflow["steps"]):
            result = await self._execute_step(step, context)
            results.append({
                "step": i + 1,
                "action": step.get("action"),
                "result": result
            })
            
            # Pass result to next step's context
            if context is None:
                context = {}
            context[f"step_{i}_result"] = result
        
        return {
            "workflow": name,
            "completed": len(results),
            "results": results
        }
    
    async def _execute_step(self, step: Dict, context: Dict = None) -> Any:
        """Execute a single workflow step"""
        action = step.get("action")
        
        if action == "wait":
            await asyncio.sleep(step.get("duration", 1))
            return "Waited"
        elif action == "transform":
            # Transform data based on rules
            data = context.get("data", "")
            transformation = step.get("transformation", "upper")
            if transformation == "upper":
                return data.upper()
            elif transformation == "lower":
                return data.lower()
            return data
        elif action == "filter":
            # Filter data based on criteria
            data = context.get("data", [])
            criteria = step.get("criteria", {})
            # Simple filtering logic
            return [item for item in data if self._matches_criteria(item, criteria)]
        
        return f"Executed: {action}"
    
    def _matches_criteria(self, item: Any, criteria: Dict) -> bool:
        """Check if item matches criteria"""
        for key, value in criteria.items():
            if hasattr(item, key):
                if getattr(item, key) != value:
                    return False
            elif isinstance(item, dict):
                if item.get(key) != value:
                    return False
        return True


class CreativeAgent:
    """Generates creative content - stories, ideas, brainstorming"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def brainstorm(self, topic: str, count: int = 10) -> List[str]:
        """Generate creative ideas"""
        ideas = []
        
        if self.llm:
            prompt = f"Generate {count} creative ideas related to: {topic}"
            response = await self.llm.send_prompt(prompt)
            # Parse response into list
            ideas = response.split('\n')
        else:
            # Mock ideas
            ideas = [f"Idea {i+1} about {topic}" for i in range(count)]
        
        return ideas
    
    async def generate_story(self, prompt: str, style: str = "short") -> str:
        """Generate creative stories"""
        if self.llm:
            story_prompt = f"Write a {style} story about: {prompt}"
            return await self.llm.send_prompt(story_prompt)
        
        return f"Once upon a time, {prompt}..."
    
    async def create_analogy(self, concept: str) -> str:
        """Create analogies to explain concepts"""
        if self.llm:
            prompt = f"Create a creative analogy to explain: {concept}"
            return await self.llm.send_prompt(prompt)
        
        return f"{concept} is like a puzzle where each piece represents a different part."
