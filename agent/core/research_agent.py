"""
Research Agent with Live Web Intelligence
Actively researches topics instead of just storing static text
"""
from typing import Dict, List, Optional, Any
import asyncio
import httpx  # type: ignore[import-untyped]
import re
from agent.core.cache_manager import cache_manager # type: ignore
from agent.config import get_settings # type: ignore
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
import logging
import re
from agent.db.graph_store import GraphStore
from agent.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


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
        from agent.core.search_providers import perform_web_search  # type: ignore[import-untyped]
        self.perform_search = perform_web_search
    
    async def search(self, query: str, max_results: int = 5, depth: str = "standard") -> List[Dict]:
        """
        Search the web for current information
        Uses Tavily, Serper, ArXiv, or DuckDuckGo based on what's configured
        """
        try:
            # Use the real search provider with depth support
            search_results = await self.perform_search(query, max_results=max_results)
            
            # Format results for the research agent
            formatted_results = []
            for result in search_results.get("results", []):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:300], # Increased snippet size
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.5), # type: ignore
                    "source": result.get("source", "unknown")
                })
            if not formatted_results:
                logger.warning(f"No results found for query: {query}")
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return []
    
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
        Uses normalized text matching to catch overlapping concepts
        """
        fact_counts = {}
        fact_sources = {}
        
        # Heuristic stop words for normalization
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "by", "of"}
        
        for source in sources:
            content = source.get("content", "")
            if not content:
                continue
                
            # Basic sentence splitting
            sentences = re.split(r'[.!?]\s+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30 and len(sentence) < 400:  # Sensible range for a "fact"
                    # Normalize for matching: lower case, remove punctuation, strip stop words
                    normalized = re.sub(r'[^\w\s]', '', sentence.lower())
                    words = [w for w in normalized.split() if w not in stop_words]
                    match_key = " ".join(words[:15]) # Use first 15 meaningful words as key
                    
                    if not match_key:
                        continue
                        
                    fact_counts[match_key] = fact_counts.get(match_key, 0) + 1
                    if match_key not in fact_sources:
                        fact_sources[match_key] = {
                            "text": sentence,
                            "sources": set()
                        }
                    fact_sources[match_key]["sources"].add(source.get("url", ""))
        
        # Return facts with confidence scores
        verified_facts = []
        total_sources = len(set(s.get("url", "") for s in sources)) or 1
        
        for match_key, data in fact_sources.items():
            count = fact_counts[match_key]
            # Boost facts mentioned in multiple sources
            confidence = min(0.4 + (count / total_sources) * 0.6, 1.0) if count > 1 else 0.3
            
            verified_facts.append({
                "fact": data["text"],
                "confidence": confidence,
                "sources": list(data["sources"]),
                "frequency": count
            })
        
        logger.info(f"Verified {len(verified_facts)} facts from {len(sources)} sources")
        
        # Sort by confidence and frequency
        return sorted(verified_facts, key=lambda x: (x["confidence"], x["frequency"]), reverse=True)


class KnowledgeGraphBuilder:
    """Builds structured knowledge from research results"""
    
    async def build(self, facts: List[Dict], query: str, llm_client: Any = None) -> Dict:
        """
        Create a knowledge graph from verified facts.
        If LLM is available, uses it for high-quality extraction.
        """
        if llm_client:
            return await self._build_with_llm(facts, query, llm_client)
        return self._build_heuristic(facts, query)

    def _build_heuristic(self, facts: List[Dict], query: str) -> Dict:
        """Fallback heuristic-based graph building"""
        graph = {
            "central_topic": query,
            "entities": [],
            "relationships": [],
            "key_facts": []
        }
        
        topic_id = self._hash_id(query)
        entities: List[Dict[str, Any]] = graph["entities"]
        entities.append({"id": topic_id, "name": query, "type": "topic"})
        
        for fact_data in facts[:15]:
            fact = fact_data["fact"]
            graph["key_facts"].append({
                "text": fact,
                "confidence": fact_data["confidence"]
            })
            
            # Simple Entity Extraction (Capitalized words)
            extracted_entities = []
            words = re.findall(r'\b[A-Z][a-z]{2,}\b', fact)
            for word in words:
                if word.lower() in ["the", "this", "that", "with", "from", "they", "there", "their"]:
                    continue
                entity_id = self._hash_id(word)
                if not any(e["id"] == entity_id for e in entities):
                    entities.append({"id": entity_id, "name": word, "type": "entity"})
                extracted_entities.append(entity_id)
            
            relationships: List[Dict[str, Any]] = graph["relationships"]
            for eid in set(extracted_entities):
                if eid != topic_id:
                    relationships.append({
                        "source": topic_id,
                        "target": eid,
                        "relation": "relates_to",
                        "metadata": {"fact": fact[:100] + "..."}
                    })
        
        return graph

    async def _build_with_llm(self, facts: List[Dict], query: str, llm: Any) -> Dict:
        """Premium LLM-powered graph construction"""
        fact_text = "\n".join([f"- {f['fact']}" for f in facts[:10]])
        prompt = f"""Based on these facts about "{query}":
{fact_text}

Extract a knowledge graph in JSON format:
{{
  "entities": [{{ "id": "uuid", "name": "Name", "type": "Type" }}],
  "relationships": [{{ "source": "uuid", "target": "uuid", "relation": "Type" }}]
}}
Focus on core concepts and meaningful relationships like "developed_by", "component_of", "solved_by"."""

        try:
            response = await llm.send_prompt(prompt)
            # Find JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                graph_data = json.loads(match.group(0))
                graph_data["central_topic"] = query
                graph_data["key_facts"] = [{"text": f["fact"], "confidence": f["confidence"]} for f in facts[:10]]
                return graph_data
        except Exception as e:
            logger.error(f"LLM Graph Builder failed: {e}")
            
        return self._build_heuristic(facts, query)

    def _hash_id(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.lower().encode()).hexdigest()


class ResearchAgent:
    """
    Autonomous agent that actively researches topics
    instead of just retrieving from static database
    """
    
    def __init__(self, llm_client=None, db_path: str = "./data/mentorzero.db", vector_db_path: str = "./data/vectorstore"):
        self.web_search = WebSearchTool()
        self.fact_verifier = FactVerifier()
        self.graph_builder = KnowledgeGraphBuilder()
        self.graph_store = GraphStore(db_path)
        self.vector_store = VectorStore(vector_db_path)
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
        
        # 2. SEMANTIC REUSE & CROSS-SESSION KNOWLEDGE
        # Check if we already have a direct or high-similarity match for this query
        semantic_threshold = get_settings().semantic_cache_threshold
        past_knowledge = self.vector_store.query_related_knowledge(query, n_results=10)
        
        # Determine if we can skip/optimize based on current knowledge
        has_deep_knowledge = len(past_knowledge) > 5
        context_summary = "\n".join([f"- {k}" for k in past_knowledge]) if past_knowledge else "None."
        
        logger.info(f"Retrieved {len(past_knowledge)} related facts. Knowledge sufficient: {has_deep_knowledge}")
        
        # STORM PATTERN: Persona-Driven Query Expansion
        search_queries = [query]
        if self.llm and depth != "quick":
            from agent.config import get_settings
            settings = get_settings()
            niche = settings.niche_focus
            
            # Generate expert personas for 360-degree coverage
            personas = await self._generate_personas(query, niche)
            logger.info(f"Generated expert personas: {personas}")
            
            # Optimization: If we have past knowledge, ask personas to focus on GAPS
            persona_tasks = [self._expand_query_as_persona(query, p, context=context_summary if has_deep_knowledge else None) for p in personas]
            expanded_lists = await asyncio.gather(*persona_tasks)
            for sub_list in expanded_lists:
                search_queries.extend(sub_list)
        elif self.llm:
            # Fallback to simple expansion for quick mode
            expanded = await self._expand_query(query)
            search_queries.extend(expanded)
        
        # Search from multiple angles in parallel! (Major Optimization)
        num_queries = 2 if depth == "quick" else 4 if depth == "standard" else 6
        search_tasks = [self.web_search.search(q, depth=depth) for q in search_queries[:num_queries]]
        
        results_lists = await asyncio.gather(*search_tasks)
        all_sources = []
        for r_list in results_lists:
            all_sources.extend(r_list)
        
        # Verify and cross-reference facts
        verified_facts = self.fact_verifier.cross_reference(all_sources)
        
        # Build knowledge graph (LLM-aware)
        knowledge_graph = await self.graph_builder.build(verified_facts, query, self.llm)
        
        # Calculate overall confidence
        relevant_facts = verified_facts[:20]
        avg_confidence = sum(f["confidence"] for f in relevant_facts) / len(relevant_facts) if relevant_facts else 0
        
        # Persist to Graph Store
        self._persist_knowledge(knowledge_graph)
        
        # Persist to Vector Store (Persistent Memory)
        self.vector_store.add_facts(verified_facts, query) # usage of query as session_id for now
        
        # IMPROVISED: Self-Correction Loop (Gap Analysis)
        # If depth is deep and we have low confidence, try to fill gaps
        if depth == "deep" and avg_confidence < 0.7 and self.llm:
            logger.info(f"Research confidence low ({avg_confidence:.2f}). Triggering gap analysis...")
            gap_result = await self.fill_knowledge_gaps([f["fact"] for f in verified_facts[:10]], query)
            
            # Merge results
            all_sources.extend(gap_result.sources)
            # Re-verify with new sources
            verified_facts = self.fact_verifier.cross_reference(all_sources)
            # Re-build graph
            knowledge_graph = await self.graph_builder.build(verified_facts, query, self.llm)
            # Recalculate confidence
            relevant_facts = verified_facts[:20]
            avg_confidence = sum(f["confidence"] for f in relevant_facts) / len(relevant_facts) if relevant_facts else 0

        # IMPROVISED: Semantic Deduplication (concise insights)
        unique_facts = [f["fact"] for f in verified_facts[:30]]
        if self.llm and len(unique_facts) > 10:
             unique_facts = await self._semantic_deduplicate(unique_facts)

        # REFLECTION PASS: Self-Critique & Refinement
        if self.llm and depth == "deep":
            logger.info("Entering Reflection phase...")
            reflection = await self._reflect_on_results(unique_facts, query)
            if reflection.get("is_sufficient") is False and reflection.get("gap_query"):
                logger.info(f"Reflection identified critical gap: {reflection['gap_query']}")
                final_gap_res = await self.web_search.search(reflection["gap_query"], depth="standard")
                all_sources.extend(final_gap_res)
                # Final re-verify and merge
                verified_facts = self.fact_verifier.cross_reference(all_sources)
                unique_facts = await self._semantic_deduplicate([f["fact"] for f in verified_facts[:30]])
        
        result = ResearchResult(
            query=query,
            sources=all_sources,
            facts=unique_facts,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            knowledge_graph=knowledge_graph
        )
        
        # Cache the result
        self.research_cache[cache_key] = result
        
        return result

    def _persist_knowledge(self, graph: Dict):
        """Save extracted entities and relationships to the persistent store"""
        try:
            # Add nodes
            for entity in graph["entities"]:
                self.graph_store.add_node(
                    name=entity["name"],
                    node_type=entity["type"],
                    metadata=entity.get("metadata", {})
                )
            
            # Add edges
            for rel in graph["relationships"]:
                self.graph_store.add_edge(
                    source_id=rel["source"],
                    target_id=rel["target"],
                    relation=rel["relation"],
                    metadata=rel.get("metadata", {})
                )
            logger.info(f"Persisted {len(graph['entities'])} nodes and {len(graph['relationships'])} edges")
        except Exception as e:
            logger.error(f"Failed to persist knowledge: {e}")
    
    async def _expand_query(self, query: str, niche: str = "") -> List[str]:
        """Use LLM to generate related search queries with optional niche biasing"""
        if not self.llm:
            return []
        
        niche_clause = f" specifically within the niche of {niche}" if niche else ""
        prompt = f"""Given the research topic: "{query}"{niche_clause}
        Generate 3 related search queries that would help gather comprehensive information.
        Return only the queries, one per line."""
        
        try:
            response = await self.llm.send_prompt(prompt, temperature=0.7)
            expanded = [q.strip() for q in response.split('\n') if q.strip()]
            return expanded[:3]  # type: ignore
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
{chr(10).join(current_knowledge[:5])}  # type: ignore

What key information is missing to fully answer this question?
List the top 3 missing pieces of information."""
        
        try:
            gaps = await self.llm.send_prompt(prompt)
            # Research the identified gaps
            return await self.research_topic(gaps)
        except:
            return await self.research_topic(user_query)
    
    async def _generate_personas(self, query: str, niche: str = "") -> List[str]:
        """Generate 2-3 expert personas to look at the topic from different angles with memoization"""
        cache_key = f"personas:{query}:{niche}"
        cached = cache_manager.get(cache_key, ttl_hours=24 * 7) # 1 week TTL for personas
        if cached:
            logger.info(f"Using memoized personas for: {query}")
            return cached

        prompt = f"Given the research topic '{query}'{f' in the {niche} field' if niche else ''}, name 3 distinct expert personas who would have unique perspectives on this (e.g. 'Security Auditor', 'Market Analyst', 'History Professor'). Return as a simple comma-separated list."
        try:
            res = await self.llm.send_prompt(prompt)
            personas = [p.strip() for p in res.split(",")][:3]
            cache_manager.set(cache_key, personas, provider="llm_planning")
            return personas
        except:
            return ["General Researcher"]

    async def _expand_query_as_persona(self, query: str, persona: str, context: Optional[str] = None) -> List[str]:
        """Generate targeted queries from a specific expert persona's viewpoint"""
        context_clause = f"\n\nWe already know:\n{context}\nFocus ONLY on finding NEW or updated information." if context else ""
        prompt = f"As a {persona}, what are 2 highly specific search queries you would use to investigate '{query}'?{context_clause}\nReturn only the queries, one per line."
        try:
            res = await self.llm.send_prompt(prompt)
            return [q.strip() for q in res.split('\n') if q.strip()][:2]
        except:
            return []

    async def _reflect_on_results(self, facts: List[str], query: str) -> Dict:
        """Analyze current findings for hallucinations, bias, or missing critical info"""
        fact_bullet = "\n".join([f"- {f}" for f in facts[:10]])
        prompt = f"""Review the following facts about '{query}':
{fact_bullet}

Critique this research:
1. Is it sufficient to provide a comprehensive answer? (True/False)
2. Is there a critical missing piece of information?
3. If not sufficient, provide ONE highly targeted search query to fix it.

Return JSON only: {{"is_sufficient": bool, "critique": "string", "gap_query": "string"}}"""
        try:
            res = await self.llm.send_prompt(prompt)
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        return {"is_sufficient": True}

    async def _semantic_deduplicate(self, facts: List[str]) -> List[str]:
        """Use LLM to remove redundancy and consolidate facts into core insights"""
        if not facts or not self.llm:
            return facts
            
        prompt = f"""These research facts contain redundancies. Consolidate them into a clean, non-repetitive list of the most important unique insights (max 15).
        
Facts:
{chr(10).join(facts)}

Return only the consolidated list, one item per line."""
        
        try:
            response = await self.llm.send_prompt(prompt, temperature=0.3)
            deduplicated = [line.strip().strip("- ") for line in response.split('\n') if line.strip() and len(line) > 20]
            return deduplicated if deduplicated else facts
        except Exception as e:
            logger.error(f"Semantic deduplication failed: {e}")
            return facts

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
    if result.knowledge_graph:
        print(f"Knowledge Graph: {len(result.knowledge_graph['entities'])} entities")  # type: ignore
    
    # Show top facts
    for fact in result.facts[:3]:  # type: ignore
        print(f"- {fact}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(demo())
