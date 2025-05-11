#!/usr/bin/env python3
"""
docinsight_client.py - Client for interacting with the DocInsight service (real or mock).
"""

import requests

class DocInsightClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def start_research(self, query: str, force_index: list[str] | None = None) -> str:
        """
        Start a research job with a query and optional list of paper filenames to index.
        Returns the job_id string.
        """
        url = f"{self.base_url}/start_research"
        payload: dict = {"query": query}
        if force_index:
            payload["force_index"] = force_index
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"start_research failed: {resp.status_code} {resp.text}")
        data = resp.json()
        return data.get("job_id")

    def get_results(self, job_ids: list[str]) -> list[dict]:
        """
        Retrieve results for given job IDs. Returns a list of result dicts.
        """
        url = f"{self.base_url}/get_results"
        payload = {"job_ids": job_ids}
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"get_results failed: {resp.status_code} {resp.text}")
        return resp.json()
