import requests
from cultivation.scripts.literature.docinsight_client import DocInsightClient

class DummyResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
        self.text = str(data)
    def json(self):
        return self._data

def test_start_research(monkeypatch):
    base = "http://mock-server"
    client = DocInsightClient(base_url=base)
    def fake_post(url, json):
        assert url == f"{base}/start_research"
        assert json == {"query": "test query", "force_index": ["paper.pdf"]}
        return DummyResponse(200, {"job_id": "job-123"})
    monkeypatch.setattr(requests, "post", fake_post)
    job_id = client.start_research(query="test query", force_index=["paper.pdf"])
    assert job_id == "job-123"

def test_get_results(monkeypatch):
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
        assert url == f"{base}/get_results"
        assert json == {"job_ids": ["job-123"]}
        return DummyResponse(200, expected)
    monkeypatch.setattr(requests, "post", fake_post)
    results = client.get_results(["job-123"])
    assert results == expected

### New tests for wait_for_result polling behavior ###
import pytest
from cultivation.scripts.literature.docinsight_client import DocInsightClient, DocInsightAPIError, DocInsightTimeoutError

def test_wait_for_result_immediate(monkeypatch):
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
        calls["count"] += 1
        if calls["count"] < 3:
            return [{"job_id": "j", "status": "pending"}]
        return [{"job_id": "j", "status": "done", "answer": "y"}]
    monkeypatch.setattr(client, "get_results", fake_get)
    result = client.wait_for_result("j", poll_interval=0.01, timeout=0.2)
    assert result.get("status") == "done"

def test_wait_for_result_error(monkeypatch):
    client = DocInsightClient(base_url="http://example")
    dummy_err = {"job_id": "j", "status": "error"}
    monkeypatch.setattr(client, "get_results", lambda job_ids, timeout: [dummy_err])
    with pytest.raises(DocInsightAPIError):
        client.wait_for_result("j", poll_interval=0.01, timeout=0.1)

def test_wait_for_result_timeout(monkeypatch):
    client = DocInsightClient(base_url="http://example")
    # Always return empty (no results), forcing timeout
    monkeypatch.setattr(client, "get_results", lambda job_ids, timeout: [])
    with pytest.raises(DocInsightTimeoutError):
        client.wait_for_result("j", poll_interval=0.01, timeout=0.05)
