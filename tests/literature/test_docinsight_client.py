import requests
from cultivation.scripts.literature.docinsight_client import DocInsightClient

class DummyResponse:
    def __init__(self, status_code, data):
        """
        Initializes a DummyResponse object to simulate an HTTP response.
        
        Args:
            status_code: The HTTP status code to simulate.
            data: The response data to be returned by the dummy response.
        """
        self.status_code = status_code
        self._data = data
        self.text = str(data)
    def json(self):
        """
        Returns the JSON data associated with the response.
        
        Returns:
            The parsed JSON content stored in the response.
        """
        return self._data

def test_start_research(monkeypatch):
    """
    Tests that DocInsightClient.start_research sends the correct POST request and returns the expected job ID.
    """
    base = "http://mock-server"
    client = DocInsightClient(base_url=base)
    def fake_post(url, json):
        """
        Simulates a POST request to the /start_research endpoint for testing purposes.
        
        Asserts that the URL and JSON payload match the expected values and returns a dummy response containing a job ID.
        """
        assert url == f"{base}/start_research"
        assert json == {"query": "test query", "force_index": ["paper.pdf"]}
        return DummyResponse(200, {"job_id": "job-123"})
    monkeypatch.setattr(requests, "post", fake_post)
    job_id = client.start_research(query="test query", force_index=["paper.pdf"])
    assert job_id == "job-123"

def test_get_results(monkeypatch):
    """
    Tests that DocInsightClient.get_results sends the correct POST request and returns the expected job results.
    
    Verifies that the method constructs the proper URL and payload, and that the parsed response matches the expected data.
    """
    base = "http://mock-server"
    client = DocInsightClient(base_url=base)
    expected = [
        {
            "job_id": "job-123", "status": "done",
            "answer": "Mock summary.", "relevance": 0.9,
            "novelty": 0.75, "chunks": []
        }
    ]
    def fake_post(url, json):
        """
        Simulates a POST request to the /get_results endpoint for testing purposes.
        
        Asserts that the URL and JSON payload match the expected values and returns a dummy response with predefined data.
        """
        assert url == f"{base}/get_results"
        assert json == {"job_ids": ["job-123"]}
        return DummyResponse(200, expected)
    monkeypatch.setattr(requests, "post", fake_post)
    results = client.get_results(["job-123"])
    assert results == expected

### New tests for wait_for_result polling behavior ###
import pytest
from cultivation.scripts.literature.docinsight_client import DocInsightAPIError, DocInsightTimeoutError

def test_wait_for_result_immediate(monkeypatch):
    """
    Tests that wait_for_result returns the result immediately when the job status is "done".
    
    Simulates the scenario where the job completes on the first poll by mocking get_results to return a completed job result.
    """
    client = DocInsightClient(base_url="http://example")
    dummy = {"job_id": "j", "status": "done", "answer": "ok"}
    # get_results returns done on first call
    monkeypatch.setattr(client, "get_results", lambda job_ids, timeout: [dummy])
    result = client.wait_for_result("j", poll_interval=0.01, timeout=0.1)
    assert result == dummy

def test_wait_for_result_delayed_success(monkeypatch):
    client = DocInsightClient(base_url="http://example")
    calls = {"count": 0}
    # pending twice then done
    def fake_get(job_ids, timeout):
        """
        Simulates polling for job results, returning 'pending' status twice before returning 'done'.
        
        Args:
            job_ids: List of job IDs to query.
            timeout: Timeout value for the simulated request.
        
        Returns:
            A list containing a dictionary with job status, returning 'pending' on the first two calls and 'done' with an answer on the third call.
        """
        calls["count"] += 1
        if calls["count"] < 3:
            return [{"job_id": "j", "status": "pending"}]
        return [{"job_id": "j", "status": "done", "answer": "y"}]
    monkeypatch.setattr(client, "get_results", fake_get)
    result = client.wait_for_result("j", poll_interval=0.01, timeout=0.2)
    assert result.get("status") == "done"

def test_wait_for_result_error(monkeypatch):
    """
    Tests that wait_for_result raises DocInsightAPIError when the job status is "error".
    """
    client = DocInsightClient(base_url="http://example")
    dummy_err = {"job_id": "j", "status": "error"}
    monkeypatch.setattr(client, "get_results", lambda job_ids, timeout: [dummy_err])
    with pytest.raises(DocInsightAPIError):
        client.wait_for_result("j", poll_interval=0.01, timeout=0.1)

def test_wait_for_result_timeout(monkeypatch):
    """
    Tests that wait_for_result raises DocInsightTimeoutError when no results are returned within the timeout period.
    """
    client = DocInsightClient(base_url="http://example")
    # Always return empty (no results), forcing timeout
    monkeypatch.setattr(client, "get_results", lambda job_ids, timeout: [])
    with pytest.raises(DocInsightTimeoutError):
        client.wait_for_result("j", poll_interval=0.01, timeout=0.05)
