#!/usr/bin/env python3
"""
docinsight_client.py - Client for interacting with the DocInsight service (real or mock).
"""

class DocInsightClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def start_research(self, paper_id):
        # Placeholder: implement call to /start_research
        print(f"[Stub] Would call /start_research for paper_id={paper_id}")
        return {"status": "ok"}

    def get_results(self, paper_id):
        # Placeholder: implement call to /get_results
        print(f"[Stub] Would call /get_results for paper_id={paper_id}")
        return {"summary": "This is a stub summary."}
