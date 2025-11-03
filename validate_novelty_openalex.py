#!/usr/bin/env python3
"""
Validate novelty using OpenAlex API (no key required!)
Polite pool: just need an email in User-Agent
"""

import json
import os
import time
import requests
from typing import Dict, List

def search_openalex(query: str, email: str = "researcher@example.com") -> Dict:
    """Search OpenAlex for similar papers"""
    
    url = "https://api.openalex.org/works"
    
    headers = {
        "User-Agent": f"mailto:{email}"  # Polite pool
    }
    
    params = {
        "search": query[:500],
        "per_page": 20,
        "sort": "relevance_score:desc"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Warning: Search failed - {e}")
        return {"results": []}

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

def validate_idea(idea: Dict, email: str) -> Dict:
    """Validate a single research idea"""
    
    title = idea.get("Title", "")
    name = idea.get("Name", "unknown")
    
    # Use just the title for searching
    query = title
    
    print(f"\n{'='*70}")
    print(f"Idea: {name}")
    print(f"Title: {title[:60]}...")
    print(f"Searching OpenAlex...")
    
    # Search OpenAlex
    results = search_openalex(query, email)
    papers = results.get("results", [])
    
    # Extract metrics
    num_papers = len(papers)
    citation_counts = [p.get("cited_by_count", 0) for p in papers]
    years = [p.get("publication_year", 0) for p in papers if p.get("publication_year")]
    max_year = max(years) if years else None
    
    # Calculate novelty
    novelty_score = calculate_novelty_score(num_papers, citation_counts, max_year)
    
    # Find most similar paper
    most_similar = None
    if papers:
        most_similar = {
            "title": papers[0].get("title", "Unknown"),
            "year": papers[0].get("publication_year", "Unknown"),
            "citations": papers[0].get("cited_by_count", 0)
        }
    
    result = {
        "name": name,
        "title": title,
        "novelty_score": round(novelty_score, 1),
        "similar_papers": num_papers,
        "avg_citations": round(sum(citation_counts) / len(citation_counts), 1) if citation_counts else 0,
        "most_recent_year": max_year,
        "most_similar_paper": most_similar,
        "validation_method": "openalex"
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
    
    # Be polite: 1 second between requests
    time.sleep(1.0)
    
    return result

def main():
    print("="*70)
    print("OPENALEX NOVELTY VALIDATION (No API Key Required!)")
    print("="*70)
    
    # Load seed ideas
    seed_file = "seed_ideas.json"
    if not os.path.exists(seed_file):
        print(f"Error: {seed_file} not found!")
        return
    
    with open(seed_file, 'r') as f:
        ideas = json.load(f)
    
    print(f"\nLoaded {len(ideas)} ideas from {seed_file}")
    
    # Get email for polite pool (or use default)
    email = os.getenv("OPENALEX_EMAIL", "researcher@example.com")
    print(f"Using email: {email}")
    print("(Set OPENALEX_EMAIL='your@email.com' for polite pool)")
    
    # Validate each idea
    results = []
    print(f"\nValidating {len(ideas)} ideas (1 second between requests)...")
    print("This will take about {len(ideas)} seconds.\n")
    
    for i, idea in enumerate(ideas, 1):
        print(f"\n[{i}/{len(ideas)}]")
        result = validate_idea(idea, email)
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
    print(f"  High Novel (≥8.0): {sum(1 for s in novelty_scores if s >= 8.0)}/{len(novelty_scores)} ({100*sum(1 for s in novelty_scores if s >= 8.0)/len(novelty_scores):.0f}%)")
    
    print(f"\nSimilar Papers:")
    print(f"  Mean: {sum(similar_papers)/len(similar_papers):.1f}")
    print(f"  Median: {sorted(similar_papers)[len(similar_papers)//2]:.0f}")
    print(f"  Range: [{min(similar_papers)}, {max(similar_papers)}]")
    
    # Save results
    output_file = "novelty_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "validation_method": "openalex",
            "summary": {
                "total_ideas": len(results),
                "mean_novelty": round(mean_novelty, 2),
                "std_novelty": round(std_novelty, 2),
                "high_novel_count": sum(1 for s in novelty_scores if s >= 8.0),
                "high_novel_percentage": round(100*sum(1 for s in novelty_scores if s >= 8.0)/len(novelty_scores), 1),
                "mean_similar_papers": round(sum(similar_papers)/len(similar_papers), 1)
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Show breakdown
    print("\nAll Ideas (sorted by novelty):")
    sorted_results = sorted(results, key=lambda x: x["novelty_score"], reverse=True)
    for i, idea in enumerate(sorted_results, 1):
        novelty_label = "★" if idea['novelty_score'] >= 8.0 else " "
        print(f"{novelty_label} {i}. {idea['name']}: {idea['novelty_score']}/10 ({idea['similar_papers']} papers)")
    
    print("\n" + "="*70)
    print("✓ Validation complete!")
    print("="*70)

if __name__ == "__main__":
    main()
