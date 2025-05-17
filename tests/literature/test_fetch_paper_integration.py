import threading
import time
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from tests.mocks.docinsight_mock import app as mock_app
import cultivation.scripts.literature.fetch_paper as fetch_module

# Constants for test directories
BASE_TEST_LIT_DIR = Path("test_integration_literature_output")
PDF_TEST_DIR = BASE_TEST_LIT_DIR / "pdf"
METADATA_TEST_DIR = BASE_TEST_LIT_DIR / "metadata"
NOTES_TEST_DIR = BASE_TEST_LIT_DIR / "notes"

@pytest.fixture(scope="session")
def live_mock_docinsight_server():
    """
    Pytest fixture that starts a mock DocInsight server in a background thread.
    
    Dynamically allocates a free TCP port, launches the mock Flask server, waits for it
    to become responsive, and yields the server's base URL for use in tests.
    """
    host = "127.0.0.1"
    # Dynamically allocate port
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    # Event to signal server is ready
    import threading
    server_ready = threading.Event()
    def run_server():
        """
        Runs the mock Flask server for testing purposes.
        
        Configures the Flask application for testing and starts it on the specified host and port.
        """
        mock_app.config['TESTING'] = True
        mock_app.run(host=host, port=port, debug=False, use_reloader=False)
        server_ready.set()
    thread = threading.Thread(
        target=run_server
    )
    thread.daemon = True
    thread.start()
    # Wait for server to start or timeout
    import requests
    from requests.exceptions import ConnectionError
    import time
    start_time = time.time()
    while time.time() - start_time < 5:
        try:
            requests.get(f"http://{host}:{port}/health")
            break
        except ConnectionError:
            time.sleep(0.1)
    yield f"http://{host}:{port}"
    # Clean shutdown if available (add mock_app.shutdown() if implemented)

@pytest.fixture
def setup_dirs(monkeypatch, tmp_path):
    # Override DocInsight and output directories via env var
    """
    Sets up temporary environment variables and directories for test output.
    
    Overrides the DocInsight API URL and output directory environment variables to use
    a local mock server and a temporary directory for the duration of the test.
    
    Yields:
        The temporary directory path for use in tests.
    """
    monkeypatch.setenv('DOCINSIGHT_API_URL', 'http://127.0.0.1:8008')
    monkeypatch.setenv('LIT_DIR_OVERRIDE', str(tmp_path))
    yield tmp_path

@patch('requests.get')
def test_fetch_arxiv_with_live_mock(
    mock_requests_get, live_mock_docinsight_server, setup_dirs, caplog
):
    # Sample arXiv XML
    """
    Tests end-to-end fetching and processing of an arXiv paper using a live mock DocInsight server.
    
    This integration test verifies that `fetch_arxiv_paper` downloads the paper PDF and metadata,
    processes them with the mock DocInsight API, and creates the expected metadata JSON and note
    markdown files with correct content in the overridden directories.
    """
    SAMPLE_XML = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        '<entry><id>http://arxiv.org/abs/2310.04822</id>'
        '<updated>2023-10-10T15:00:00Z</updated>'
        '<published>2023-10-09T12:00:00Z</published>'
        '<title>Title</title><summary>Abstract</summary>'
        '<author><name>A One</name></author>'
        '<arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/>'
        '</entry></feed>'
    )
    # Mock PDF and API responses
    pdf_resp = MagicMock()
    pdf_resp.status_code = 200
    pdf_resp.content = b"PDF"
    pdf_resp.raise_for_status = lambda: None
    api_resp = MagicMock()
    api_resp.status_code = 200
    api_resp.content = SAMPLE_XML.encode('utf-8')
    api_resp.raise_for_status = lambda: None
    mock_requests_get.side_effect = [pdf_resp, api_resp]

    arxiv_id = "2310.04822"
    result = fetch_module.fetch_arxiv_paper(arxiv_id)
    assert result is True

    # Verify metadata JSON
    meta_file = setup_dirs / 'metadata' / f"{arxiv_id}.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta.get('docinsight_summary', '').startswith("Mock summary")
    assert meta.get('docinsight_novelty') == 0.75

    # Verify note contents
    note_file = setup_dirs / 'notes' / f"{arxiv_id}.md"
    assert note_file.exists()
    note_text = note_file.read_text()
    assert "## DocInsight Summary" in note_text
    assert "Novelty Score" in note_text
