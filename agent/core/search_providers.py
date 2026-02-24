import os
import httpx  # type: ignore
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class TavilySearch:
    """Tavily API for deep web search"""
    
    def __init__(self):
        self.api_key = os.getenv("MZ_TAVILY_API_KEY", "")
        self.base_url = "https://api.tavily.com"
        self.enabled = bool(self.api_key)
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.enabled:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": True,
                        "include_raw_content": True,
                        "search_depth": "advanced"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # Include the AI-generated answer if available
                    if data.get("answer"):
                        results.append({
                            "title": "AI Summary",
                            "content": data["answer"],
                            "url": "tavily-answer",
                            "score": 1.0
                        })
                    
                    # Add search results
                    for result in data.get("results", []):
                        results.append({
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0.5)
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            
        return []


class SerperSearch:
    """Serper API for Google search results"""
    
    def __init__(self):
        self.api_key = os.getenv("MZ_SERPER_API_KEY", "")
        self.base_url = "https://google.serper.dev"
        self.enabled = bool(self.api_key)
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.enabled:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "q": query,
                        "num": max_results
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # Organic search results
                    for item in data.get("organic", []):
                        results.append({
                            "title": item.get("title", ""),
                            "content": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "score": 0.8
                        })
                    
                    # Knowledge graph if available
                    if data.get("knowledgeGraph"):
                        kg = data["knowledgeGraph"]
                        results.insert(0, {
                            "title": kg.get("title", "Knowledge Graph"),
                            "content": kg.get("description", ""),
                            "url": kg.get("descriptionUrl", ""),
                            "score": 0.95
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            
        return []


class ArxivSearch:
    """ArXiv API for academic papers (no key needed)"""
    
    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"
        self.enabled = os.getenv("MZ_ARXIV_ENABLED", "true").lower() == "true"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.enabled:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    self.base_url,
                    params={
                        "search_query": f"all:{query}",
                        "max_results": max_results,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending"
                    }
                )
                
                if response.status_code == 200:
                    # Parse XML response (simplified)
                    import re
                    text = response.text
                    
                    results = []
                    entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
                    
                    for entry in entries:
                        title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                        summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                        id_match = re.search(r'<id>(.*?)</id>', entry)
                        
                        if title_match and summary_match:
                            title_text = re.sub(r'\s+', ' ', title_match.group(1).strip())
                            summary_text = re.sub(r'\s+', ' ', summary_match.group(1).strip())[:500] # type: ignore
                            paper_url = id_match.group(1) if id_match else ""
                            results.append({
                                "title": title_text,
                                "content": summary_text,
                                "url": paper_url,
                                "score": 0.9
                            })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            
        return []


class DuckDuckGoSearch:
    """DuckDuckGo search (no API key needed, but limited)"""
    
    def __init__(self):
        self.enabled = True  # Always enabled as fallback
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Note: This is a simplified implementation
        For production, use duckduckgo-search library
        """
        try:
            # Attempt to use duckduckgo-search library if installed
            try:
                from duckduckgo_search import DDGS # type: ignore
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(query, max_results=max_results)]
                    return [{
                        "title": r.get("title", ""),
                        "content": r.get("body", ""),
                        "url": r.get("href", ""),
                        "score": 0.7
                    } for r in results]
            except ImportError:
                logger.warning(f"DuckDuckGo search requested but 'duckduckgo-search' package not found. Install it for results without API keys.")
                return []
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            
        return []


class UnifiedSearchProvider:
    """Combines multiple search providers for comprehensive results"""
    
    def __init__(self):
        self.providers = {
            "tavily": TavilySearch(),
            "serper": SerperSearch(),
            "arxiv": ArxivSearch(),
            "duckduckgo": DuckDuckGoSearch()
        }
        
        # Check which providers are enabled
        self.enabled_providers = []
        if self.providers["tavily"].enabled:
            self.enabled_providers.append("tavily")
            logger.info("Tavily search enabled")
        if self.providers["serper"].enabled:
            self.enabled_providers.append("serper")
            logger.info("Serper search enabled")
        if self.providers["arxiv"].enabled:
            self.enabled_providers.append("arxiv")
            logger.info("ArXiv search enabled")
        
        # Always attempt to include DuckDuckGo if possible (as a permanent fallback/broad search)
        try:
            from duckduckgo_search import DDGS # type: ignore
            self.enabled_providers.append("duckduckgo")
            logger.info("DuckDuckGo search enabled")
        except ImportError:
            if not self.enabled_providers:
                logger.warning("No search providers available. Install 'duckduckgo-search' for a free fallback.")
    
    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search across all enabled providers
        Returns aggregated and deduplicated results
        """
        all_results = []
        sources_used = []
        
        # Run searches in parallel across enabled providers
        tasks = []
        for provider_name in self.enabled_providers:
            provider = self.providers[provider_name]
            tasks.append(provider.search(query, max_results))
            sources_used.append(provider_name)
        
        if tasks:
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            for provider_name, results in zip(self.enabled_providers, results_lists):
                if isinstance(results, Exception):
                    logger.error(f"Error in {provider_name}: {results}")
                    continue
                    
                if results and isinstance(results, list):  # type: ignore
                    for result in results:
                        if isinstance(result, dict):
                            result["source"] = provider_name
                            all_results.append(result)
        
        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Simple deduplication by title similarity
        seen_titles = set()
        unique_results = []
        for result in all_results:
            title_key = result.get("title", "").lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(result)
                if len(unique_results) >= max_results * 2:  # Keep more for diversity
                    break
        
        return {
            "query": query,
            "results": unique_results[:max_results],  # type: ignore
            "sources": sources_used,
            "timestamp": datetime.now().isoformat(),
            "total_found": len(all_results)
        }


# Global instance
search_provider = UnifiedSearchProvider()


async def perform_web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Main function to perform web search
    Uses all available search providers
    """
    return await search_provider.search(query, max_results)

