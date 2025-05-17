import json
import pytest
from pathlib import Path
from datetime import datetime

import requests

from cultivation.scripts.literature.fetch_arxiv_batch import (
    load_state, save_state, get_new_ids_for_query, main as batch_main
)

class DummyResp:
    def __init__(self, content):
        """
        Initializes a DummyResp instance with the given response content.
        
        Args:
            content: The content to be returned as the HTTP response body.
        """
        self.content = content
        self.status_code = 200
    def raise_for_status(self):
        """
        Simulates a successful HTTP response by performing no action when called.
        """
        pass


def test_load_and_save_state(tmp_path):
    state_file = tmp_path / 'state.json'
    state = {'foo': 'bar'}
    save_state(state_file, state)
    loaded = load_state(state_file)
    assert loaded == state


def test_load_state_empty(tmp_path):
    """
    Tests that loading state from a nonexistent file returns an empty dictionary.
    """
    state_file = tmp_path / 'no.json'
    loaded = load_state(state_file)
    assert loaded == {}


def test_get_new_ids_for_query(monkeypatch):
    """
    Tests that get_new_ids_for_query correctly extracts arXiv IDs from a mocked XML response.
    
    Replaces requests.get with a mock returning a sample Atom feed, then asserts that the
    function returns the expected list of IDs.
    """
    sample = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
    <entry><id>http://arxiv.org/abs/1.2</id></entry>
    <entry><id>http://arxiv.org/abs/3.4</id></entry></feed>'''
    monkeypatch.setattr(requests, 'get', lambda url: DummyResp(sample))
    ids = get_new_ids_for_query('test', datetime(2020,1,1))
    assert ids == ['1.2', '3.4']


def test_main_invokes_fetch_arxiv_paper(monkeypatch, tmp_path):
    # Stub get_new_ids_for_query and fetch_arxiv_paper
    """
    Tests that the batch_main function invokes fetch_arxiv_paper for each new arXiv ID
    and updates the state file accordingly.
    
    This test monkeypatches dependencies to simulate fetching IDs and recording calls,
    sets up command-line arguments, runs the batch process, and verifies correct
    function invocation and state persistence.
    """
    monkeypatch.setattr(
        'cultivation.scripts.literature.fetch_arxiv_batch.get_new_ids_for_query',
        lambda q, since: ['1.2', '5.6']
    )
    called = []
    monkeypatch.setattr(
        'cultivation.scripts.literature.fetch_arxiv_batch.fetch_arxiv_paper',
        lambda aid: called.append(aid) or True
    )
    state_file = tmp_path / 'state.json'
    monkeypatch.setattr('sys.argv', [
        'prog', '--queries', 'foo', '--state-file', str(state_file)
    ])
    batch_main()
    assert called == ['1.2', '5.6']
    # state file updated
    state = json.loads(state_file.read_text())
    assert 'foo' in state
