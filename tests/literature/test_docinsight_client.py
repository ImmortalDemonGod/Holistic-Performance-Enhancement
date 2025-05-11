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
