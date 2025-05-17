#!/usr/bin/env python3
"""
docinsight_client.py - Client for interacting with the DocInsight service (real or mock).
"""

import requests
import os
import time
import logging
import inspect

class DocInsightAPIError(Exception):
    pass

class DocInsightTimeoutError(Exception):
    pass

class DocInsightClient:
    # Poll settings: interval and timeout (defaults: 5s interval, 600s timeout)
    @staticmethod
    def _as_float(env_name: str, default: float) -> float:
        """
        Reads an environment variable as a float, returning a default value if unset or invalid.
        
        Args:
            env_name: Name of the environment variable to read.
            default: Value to return if the environment variable is missing or cannot be converted to float.
        
        Returns:
            The float value of the environment variable, or the default if conversion fails.
        """
        try:
            return float(os.getenv(env_name, default))
        except (TypeError, ValueError):
            logging.warning("Invalid %s – using default %.1f", env_name, default)
            return default
    DEFAULT_POLL_INTERVAL = _as_float.__func__('DOCINSIGHT_POLL_INTERVAL_SECONDS', 5.0)
    DEFAULT_POLL_TIMEOUT  = _as_float.__func__('DOCINSIGHT_POLL_TIMEOUT_SECONDS', 600.0)

    def __init__(self, base_url=None):
        # Prefer BASE_SERVER_URL env (matches DocInsight .env), fallback to localhost:52020
        """
        Initializes the DocInsightClient with a base URL.
        
        If no base URL is provided, uses the BASE_SERVER_URL environment variable or defaults to 'http://localhost:52020'.
        """
        self.base_url = (base_url or os.getenv("BASE_SERVER_URL") or "http://localhost:52020").rstrip('/')

    def start_research(self, query: str, force_index: list[str] | None = None, timeout: float = 30.0) -> str:
        """
        Starts a research job on the DocInsight service and returns the job ID.
        
        Sends a POST request to the `/start_research` endpoint with the specified query and optional forced index list. Raises a `DocInsightAPIError` if the request fails or the response does not contain a job ID, and raises a `DocInsightTimeoutError` if the request times out.
        
        Args:
            query: The research query string to submit.
            force_index: Optional list of index names to force the search on.
            timeout: Maximum time in seconds to wait for the request.
        
        Returns:
            The job ID assigned to the research task.
        """
        url = f"{self.base_url}/start_research"
        payload = {"query": query}
        if force_index:
            payload["force_index"] = force_index
        try:
            # Conditionally include timeout to support mocks without timeout parameter
            post_kwargs = {'json': payload}
            if 'timeout' in inspect.signature(requests.post).parameters:
                post_kwargs['timeout'] = timeout
            resp = requests.post(url, **post_kwargs)
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
        """
        Retrieves the results for a list of research job IDs from the DocInsight service.
        
        Args:
            job_ids: List of job IDs to fetch results for.
            timeout: Maximum time in seconds to wait for the request (default is 30.0).
        
        Returns:
            A list of dictionaries containing the results for each requested job ID.
        
        Raises:
            DocInsightAPIError: If the API returns an error or the request fails.
            DocInsightTimeoutError: If the request times out.
        """
        url = f"{self.base_url}/get_results"
        payload = {"job_ids": job_ids}
        try:
            # Conditionally include timeout to support mocks without timeout parameter
            post_kwargs = {'json': payload}
            if 'timeout' in inspect.signature(requests.post).parameters:
                post_kwargs['timeout'] = timeout
            resp = requests.post(url, **post_kwargs)
            if resp.status_code != 200:
                raise DocInsightAPIError(f"get_results failed: {resp.status_code} {resp.text}")
            return resp.json()
        except requests.Timeout as e:
            raise DocInsightTimeoutError(f"get_results timed out: {e}")
        except requests.RequestException as e:
            raise DocInsightAPIError(f"get_results request error: {e}")

    def wait_for_result(self, job_id: str, poll_interval: float = None, timeout: float = None) -> dict:
        """
        Polls for the completion of a DocInsight job until it finishes or a timeout occurs.
        
        Repeatedly checks the status of the specified job by calling `get_results` at regular intervals.
        Returns the result dictionary when the job status is "completed", "done", or "success".
        Raises `DocInsightAPIError` if the job status is "error", or `DocInsightTimeoutError` if the job does not complete within the timeout period.
        
        Args:
            job_id: The identifier of the DocInsight job to monitor.
            poll_interval: Optional polling interval in seconds; defaults to the class constant.
            timeout: Optional maximum time to wait in seconds; defaults to the class constant.
        
        Returns:
            The result dictionary for the completed job.
        
        Raises:
            DocInsightAPIError: If the job fails with an error status.
            DocInsightTimeoutError: If the job does not complete within the timeout period.
        """
        # use defaults if not provided
        poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        timeout = timeout or self.DEFAULT_POLL_TIMEOUT
        start = time.time()
        while time.time() - start < timeout:
            try:
                results = self.get_results([job_id], timeout=min(30, poll_interval))
            except DocInsightTimeoutError:
                continue
            except DocInsightAPIError:
                raise
            if results and isinstance(results, list):
                status = results[0].get("status")
                # handle various completion statuses
                if status in ("completed", "done", "success"):
                    return results[0]
                if status == "error":
                    raise DocInsightAPIError(f"DocInsight job {job_id} failed: {results[0]}")
                # else status pending or unknown: continue polling
            elapsed = time.time() - start
            time_remaining = timeout - elapsed
            time.sleep(min(poll_interval, max(0, time_remaining)))
        logging.error(f"DocInsight job {job_id} did not complete within {timeout} seconds.")
        raise DocInsightTimeoutError(f"Timeout after {timeout}s waiting on job {job_id}")
