"""
Research Agent with Live Web Intelligence
Actively researches topics instead of just storing static text
"""
from typing import Dict, List, Optional, Any
import asyncio
import httpx
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib


@dataclass
class ResearchResult:
    query: str
    sources: List[Dict[str, Any]]
    facts: List[str]
    confidence: float
    timestamp: datetime
    knowledge_graph: Optional[Dict] = None


class WebSearchTool:
    """Interface for web search APIs - now with REAL search!"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # Import the real search provider
        from agent.core.search_providers import perform_web_search
        self.perform_search = perform_web_search
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web for current information
        Uses Tavily, Serper, ArXiv, or DuckDuckGo based on what's configured
        """
        try:
            # Use the real search provider
            search_results = await self.perform_search(query, max_results)
            
            # Format results for the research agent
            formatted_results = []
            for result in search_results.get("results", []):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:200],
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.5),
                    "source": result.get("source", "unknown")
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error, falling back to mock: {e}")
            # Fallback to mock if search fails
            return [
                {
                    "title": f"Result for {query}",
                    "url": "https://example.com",
                    "snippet": f"Information about {query}",
                    "content": f"Detailed content about {query}",
                    "score": 0.95,
                    "source": "mock"
                }
            ]
    
    async def close(self):
        pass  # No longer need to close client


class FactVerifier:
    """Cross-references information across multiple sources"""
    
    def __init__(self):
        self.trust_scores = {}
    
    def cross_reference(self, sources: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract facts that appear in multiple sources
        Higher confidence for facts mentioned more frequently
        """
        fact_counts = {}
        fact_sources = {}
        
        for source in sources:
            content = source.get("content", "")
            # Simple fact extraction - enhance with NLP
            sentences = content.split(". ")
            
            for sentence in sentences:
                if len(sentence) > 20:  # Basic filter
                    fact_hash = hashlib.md5(sentence.lower().encode()).hexdigest()
                    fact_counts[fact_hash] = fact_counts.get(fact_hash, 0) + 1
                    
                    if fact_hash not in fact_sources:
                        fact_sources[fact_hash] = {
                            "text": sentence,
                            "sources": []
                        }
                    fact_sources[fact_hash]["sources"].append(source["url"])
        
        # Return facts with confidence scores
        verified_facts = []
        for fact_hash, data in fact_sources.items():
            if fact_counts[fact_hash] > 1:  # Mentioned in multiple sources
                verified_facts.append({
                    "fact": data["text"],
                    "confidence": min(fact_counts[fact_hash] / len(sources), 1.0),
                    "sources": data["sources"]
                })
        
        return sorted(verified_facts, key=lambda x: x["confidence"], reverse=True)


class KnowledgeGraphBuilder:
    """Builds structured knowledge from research results"""
    
    def build(self, facts: List[Dict], query: str) -> Dict:
        """
        Create a knowledge graph from verified facts
        """
        graph = {
            "central_topic": query,
            "entities": [],
            "relationships": [],
            "key_facts": []
        }
        
        for fact_data in facts[:10]:  # Top 10 facts
            fact = fact_data["fact"]
            graph["key_facts"].append({
                "text": fact,
                "confidence": fact_data["confidence"]
            })
            
            # Extract entities (simplified - use NER in production)
            # This would use spaCy or similar for entity extraction
            words = fact.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    if word not in graph["entities"]:
                        graph["entities"].append(word)
        
        return graph


class ResearchAgent:
    """
    Autonomous agent that actively researches topics
    instead of just retrieving from static database
    """
    
    def __init__(self, llm_client=None):
        self.web_search = WebSearchTool()
        self.fact_verifier = FactVerifier()
        self.graph_builder = KnowledgeGraphBuilder()
        self.llm = llm_client
        self.research_cache = {}
    
    async def research_topic(
        self, 
        query: str, 
        depth: str = "standard"
    ) -> ResearchResult:
        """
        Actively research a topic using multiple sources
        
        Args:
            query: Topic to research
            depth: "quick" | "standard" | "deep"
        """
        # Check cache first
        cache_key = f"{query}:{depth}"
        if cache_key in self.research_cache:
            cached = self.research_cache[cache_key]
            if (datetime.now() - cached.timestamp).seconds < 3600:  # 1 hour cache
                return cached
        
        # Expand query if we have LLM
        search_queries = [query]
        if self.llm:
            # Generate related queries for comprehensive research
            expanded = await self._expand_query(query)
            search_queries.extend(expanded)
        
        # Search from multiple angles
        all_sources = []
        for q in search_queries[:3]:  # Limit to avoid rate limits
            results = await self.web_search.search(q)
            all_sources.extend(results)
        
        # Verify and cross-reference facts
        verified_facts = self.fact_verifier.cross_reference(all_sources)
        
        # Build knowledge graph
        knowledge_graph = self.graph_builder.build(verified_facts, query)
        
        # Calculate overall confidence
        avg_confidence = sum(f["confidence"] for f in verified_facts) / len(verified_facts) if verified_facts else 0
        
        result = ResearchResult(
            query=query,
            sources=all_sources,
            facts=[f["fact"] for f in verified_facts],
            confidence=avg_confidence,
            timestamp=datetime.now(),
            knowledge_graph=knowledge_graph
        )
        
        # Cache the result
        self.research_cache[cache_key] = result
        
        return result
    
    async def _expand_query(self, query: str) -> List[str]:
        """Use LLM to generate related search queries"""
        if not self.llm:
            return []
        
        prompt = f"""Given the research topic: "{query}"
        Generate 3 related search queries that would help gather comprehensive information.
        Return only the queries, one per line."""
        
        try:
            response = await self.llm.send_prompt(prompt, temperature=0.7)
            expanded = [q.strip() for q in response.split('\n') if q.strip()]
            return expanded[:3]
        except:
            return []
    
    async def fill_knowledge_gaps(
        self, 
        current_knowledge: List[str], 
        user_query: str
    ) -> ResearchResult:
        """
        Identify what's missing from current knowledge
        and research to fill the gaps
        """
        if not self.llm:
            # Fallback to basic research
            return await self.research_topic(user_query)
        
        # Ask LLM what's missing
        prompt = f"""User asked: {user_query}
        
Current knowledge base contains:
{chr(10).join(current_knowledge[:5])}

What key information is missing to fully answer this question?
List the top 3 missing pieces of information."""
        
        try:
            gaps = await self.llm.send_prompt(prompt)
            # Research the identified gaps
            return await self.research_topic(gaps)
        except:
            return await self.research_topic(user_query)
    
    async def close(self):
        """Clean up resources"""
        await self.web_search.close()


# Example usage
async def demo():
    agent = ResearchAgent()
    
    # Research a topic
    result = await agent.research_topic("Latest developments in RAG systems 2024")
    
    print(f"Found {len(result.facts)} verified facts")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Knowledge Graph: {len(result.knowledge_graph['entities'])} entities")
    
    # Show top facts
    for fact in result.facts[:3]:
        print(f"- {fact}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(demo())
