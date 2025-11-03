#!/usr/bin/env python3
"""
Validate novelty of seed ideas using Semantic Scholar API
With rate limiting protection and retry logic
"""

import json
import os
import time
import requests
from typing import Dict, List

def search_semantic_scholar(query: str, api_key: str = None, max_retries: int = 3) -> Dict:
    """Search Semantic Scholar for similar papers with retry logic"""
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    params = {
        "query": query[:500],  # Limit query length
        "limit": 20,
        "fields": "title,abstract,year,citationCount,authors"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 429:
                # Rate limited - wait longer
                wait_time = 5 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Search failed after {max_retries} attempts - {e}")
                return {"data": []}
            else:
                wait_time = 3 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
    
    return {"data": []}

def calculate_novelty_score(num_papers: int, citation_counts: List[int], max_year: int) -> float:
    """Calculate novelty score (1-10)"""
    
    if num_papers == 0:
        return 10.0
    elif num_papers < 5:
        paper_score = 9.0
    elif num_papers < 10:
        paper_score = 7.5
    elif num_papers < 20:
        paper_score = 6.0
    else:
        paper_score = 4.0
    
    # Citation adjustment
    if citation_counts:
        avg_citations = sum(citation_counts) / len(citation_counts)
        if avg_citations < 10:
            citation_adjustment = 0.5
        elif avg_citations < 50:
            citation_adjustment = 0.0
        elif avg_citations < 100:
            citation_adjustment = -0.5
        else:
            citation_adjustment = -1.0
    else:
        citation_adjustment = 0.0
    
    # Recency adjustment
    current_year = 2025
    if max_year and max_year >= current_year - 2:
        recency_adjustment = 0.5
    else:
        recency_adjustment = 0.0
    
    final_score = paper_score + citation_adjustment + recency_adjustment
    return max(1.0, min(10.0, final_score))

def validate_idea(idea: Dict, api_key: str = None) -> Dict:
    """Validate a single research idea"""
    
    title = idea.get("Title", "")
    name = idea.get("Name", "unknown")
    
    # Create SHORT search query (just title + key terms)
    # Semantic Scholar works better with shorter queries
    query = title
    
    print(f"\n{'='*70}")
    print(f"Idea: {name}")
    print(f"Title: {title[:60]}...")
    print(f"Searching Semantic Scholar...")
    
    # Search with rate limiting protection
    results = search_semantic_scholar(query, api_key)
    papers = results.get("data", [])
    
    # Extract metrics
    num_papers = len(papers)
    citation_counts = [p.get("citationCount", 0) for p in papers]
    years = [p.get("year", 0) for p in papers if p.get("year")]
    max_year = max(years) if years else None
    
    # Calculate novelty
    novelty_score = calculate_novelty_score(num_papers, citation_counts, max_year)
    
    # Find most similar paper
    most_similar = None
    if papers:
        most_similar = {
            "title": papers[0].get("title", "Unknown"),
            "year": papers[0].get("year", "Unknown"),
            "citations": papers[0].get("citationCount", 0)
        }
    
    result = {
        "name": name,
        "title": title,
        "novelty_score": round(novelty_score, 1),
        "similar_papers": num_papers,
        "avg_citations": round(sum(citation_counts) / len(citation_counts), 1) if citation_counts else 0,
        "most_recent_year": max_year,
        "most_similar_paper": most_similar,
        "validation_method": "semantic_scholar"
    }
    
    # Print summary
    print(f"✓ Found {num_papers} similar papers")
    if citation_counts:
        print(f"  Avg citations: {result['avg_citations']:.0f}")
    if max_year:
        print(f"  Most recent: {max_year}")
    print(f"  Novelty Score: {novelty_score:.1f}/10")
    
    if most_similar:
        print(f"  Most similar: \"{most_similar['title'][:50]}...\" ({most_similar['year']})")
    
    # Wait between requests (3s without API key, 1s with)
    wait_time = 1.0 if api_key else 3.0
    time.sleep(wait_time)
    
    return result

def main():
    print("="*70)
    print("SEMANTIC SCHOLAR NOVELTY VALIDATION")
    print("="*70)
    
    # Load seed ideas
    seed_file = "seed_ideas.json"
    if not os.path.exists(seed_file):
        print(f"Error: {seed_file} not found!")
        return
    
    with open(seed_file, 'r') as f:
        ideas = json.load(f)
    
    print(f"\nLoaded {len(ideas)} ideas from {seed_file}")
    
    # Get API key
    api_key = os.getenv("S2_API_KEY")
    if not api_key:
        print("\n⚠ No API key - using public API with 3s delays")
        print("Get free key: https://www.semanticscholar.org/product/api")
    else:
        print("\n✓ Using Semantic Scholar API key (faster)")
    
    # Validate each idea
    results = []
    for i, idea in enumerate(ideas, 1):
        print(f"\n[{i}/{len(ideas)}]")
        result = validate_idea(idea, api_key)
        results.append(result)
    
    # Calculate summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    novelty_scores = [r["novelty_score"] for r in results]
    similar_papers = [r["similar_papers"] for r in results]
    
    mean_novelty = sum(novelty_scores) / len(novelty_scores)
    std_novelty = (sum((x - mean_novelty)**2 for x in novelty_scores) / len(novelty_scores))**0.5
    
    print(f"\nTotal ideas validated: {len(results)}")
    print(f"\nNovelty Scores:")
    print(f"  Mean: {mean_novelty:.2f} ± {std_novelty:.2f}")
    print(f"  Range: [{min(novelty_scores):.1f}, {max(novelty_scores):.1f}]")
    print(f"  High Novel (≥8.0): {sum(1 for s in novelty_scores if s >= 8.0)}/{len(novelty_scores)}")
    
    print(f"\nSimilar Papers:")
    print(f"  Mean: {sum(similar_papers)/len(similar_papers):.1f}")
    print(f"  Median: {sorted(similar_papers)[len(similar_papers)//2]:.0f}")
    
    # Save results
    output_file = "novelty_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total_ideas": len(results),
                "mean_novelty": round(mean_novelty, 2),
                "std_novelty": round(std_novelty, 2),
                "high_novel_count": sum(1 for s in novelty_scores if s >= 8.0),
                "mean_similar_papers": round(sum(similar_papers)/len(similar_papers), 1)
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Show top 3
    print("\nTop 3 Most Novel Ideas:")
    top_3 = sorted(results, key=lambda x: x["novelty_score"], reverse=True)[:3]
    for i, idea in enumerate(top_3, 1):
        print(f"{i}. {idea['name']}: {idea['novelty_score']}/10 ({idea['similar_papers']} similar)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
