#!/usr/bin/env python3
"""
docinsight_client.py - Client for interacting with the DocInsight service (real or mock).
"""

import requests
import os
import time

class DocInsightAPIError(Exception):
    pass

class DocInsightTimeoutError(Exception):
    pass

class DocInsightClient:
    def __init__(self, base_url=None):
        # Prefer BASE_SERVER_URL env (matches DocInsight .env), fallback to localhost:52020
        self.base_url = (base_url or os.getenv("BASE_SERVER_URL") or "http://localhost:52020").rstrip('/')

    def start_research(self, query: str, force_index: list[str] | None = None, timeout: float = 30.0) -> str:
        url = f"{self.base_url}/start_research"
        payload = {"query": query}
        if force_index:
            payload["force_index"] = force_index
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise DocInsightAPIError(f"start_research failed: {resp.status_code} {resp.text}")
            data = resp.json()
            job_id = data.get("job_id")
            if not job_id:
                raise DocInsightAPIError(f"No job_id returned: {data}")
            return job_id
        except requests.Timeout as e:
            raise DocInsightTimeoutError(f"start_research timed out: {e}")
        except requests.RequestException as e:
            raise DocInsightAPIError(f"start_research request error: {e}")

    def get_results(self, job_ids: list[str], timeout: float = 30.0) -> list[dict]:
        url = f"{self.base_url}/get_results"
        payload = {"job_ids": job_ids}
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise DocInsightAPIError(f"get_results failed: {resp.status_code} {resp.text}")
            return resp.json()
        except requests.Timeout as e:
            raise DocInsightTimeoutError(f"get_results timed out: {e}")
        except requests.RequestException as e:
            raise DocInsightAPIError(f"get_results request error: {e}")

    def wait_for_result(self, job_id: str, poll_interval: float = 3.0, timeout: float = 60.0) -> dict:
        """
        Polls get_results until the job is complete or timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                results = self.get_results([job_id], timeout=poll_interval)
            except DocInsightTimeoutError:
                continue
            except DocInsightAPIError:
                raise
            if results and isinstance(results, list):
                status = results[0].get("status")
                if status in ("completed", "done"):
                    return results[0]
                elif status == "error":
                    raise DocInsightAPIError(f"DocInsight job {job_id} failed: {results[0]}")
            time.sleep(poll_interval)
        raise DocInsightTimeoutError(f"DocInsight job {job_id} did not complete in {timeout} seconds.")
