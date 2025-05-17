# tests/literature/test_fetch_paper.py
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import datetime
import json
import requests  # Added missing import for requests
# NOTE: If file system mocking becomes more complex, consider pyfakefs for more robust isolation.

from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper, main as fetch_paper_main
from cultivation.scripts.literature.docinsight_client import DocInsightClient, DocInsightAPIError
import cultivation.scripts.literature.fetch_paper as fp

BASE_TEST_LIT_DIR = Path("test_literature_output")
PDF_TEST_DIR = BASE_TEST_LIT_DIR / "pdf"
METADATA_TEST_DIR = BASE_TEST_LIT_DIR / "metadata"
NOTES_TEST_DIR = BASE_TEST_LIT_DIR / "notes"

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2310.04822v1</id>
    <updated>2023-10-10T15:00:00Z</updated>
    <published>2023-10-09T12:00:00Z</published>
    <title>A Test Paper Title for Cultivation</title>
    <summary>This is the abstract of the test paper. It is very insightful.</summary>
    <author><name>Author One</name></author>
    <author><name>Author Two</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

@pytest.fixture
def mock_arxiv_paths(monkeypatch):
    """
    Pytest fixture that monkeypatches arXiv-related directory paths for isolated testing.
    
    Temporarily overrides the PDF, metadata, and notes directories in the target module to use test-specific paths. Cleans up these directories before and after each test to ensure a clean test environment.
    """
    monkeypatch.setattr("cultivation.scripts.literature.fetch_paper.PDF_DIR", PDF_TEST_DIR)
    monkeypatch.setattr("cultivation.scripts.literature.fetch_paper.METADATA_DIR", METADATA_TEST_DIR)
    monkeypatch.setattr("cultivation.scripts.literature.fetch_paper.NOTES_DIR", NOTES_TEST_DIR)
    for p_dir in [PDF_TEST_DIR, METADATA_TEST_DIR, NOTES_TEST_DIR, BASE_TEST_LIT_DIR]:
        if p_dir.exists():
            for f in p_dir.glob("*"):
                f.unlink()
            if not any(p_dir.iterdir()):
                p_dir.rmdir()
    yield
    for p_dir in [PDF_TEST_DIR, METADATA_TEST_DIR, NOTES_TEST_DIR, BASE_TEST_LIT_DIR]:
        if p_dir.exists():
            for f in p_dir.glob("*"):
                f.unlink()
            if not any(p_dir.iterdir()):
                p_dir.rmdir()

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch.object(Path, 'mkdir')
def test_fetch_arxiv_paper_success(
    mock_mkdir, mock_open, mock_get, mock_arxiv_paths, monkeypatch
):
    """
    Tests that fetch_arxiv_paper successfully downloads the PDF and metadata, writes files,
    and interacts with DocInsightClient as expected.
    
    Simulates successful HTTP responses for PDF and metadata, verifies file creation,
    and asserts correct calls to DocInsightClient methods.
    """
    arxiv_id = "2310.04822"
    cleaned_arxiv_id = "2310.04822"

    mock_pdf_response = MagicMock()
    mock_pdf_response.status_code = 200
    mock_pdf_response.content = b"dummy pdf content"
    mock_pdf_response.raise_for_status = MagicMock()

    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.content = SAMPLE_ARXIV_XML.encode('utf-8')
    mock_api_response.raise_for_status = MagicMock()

    mock_get.side_effect = [mock_pdf_response, mock_api_response]

    # Monkeypatch DocInsightClient methods
    mock_start = MagicMock(return_value='job1')
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', mock_start)
    mock_wait = MagicMock(return_value={'answer':'ok','novelty':0.5})
    monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', mock_wait)

    result = fetch_arxiv_paper(arxiv_id)

    assert result is True
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)
    mock_get.assert_any_call(f"https://arxiv.org/pdf/{cleaned_arxiv_id}.pdf", timeout=30)
    mock_get.assert_any_call(f"http://export.arxiv.org/api/query?id_list={cleaned_arxiv_id}", timeout=30)

    pdf_path = PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf"
    mock_open.assert_any_call(pdf_path, 'wb')
    handle_pdf = mock_open()
    handle_pdf.write.assert_any_call(b"dummy pdf content")

    metadata_path = METADATA_TEST_DIR / f"{cleaned_arxiv_id}.json"
    mock_open.assert_any_call(metadata_path, 'w')
    found_json_write = False
    for call_args in mock_open.call_args_list:
        if call_args[0][0] == metadata_path:
            found_json_write = True
            break
    assert found_json_write

    note_path = NOTES_TEST_DIR / f"{cleaned_arxiv_id}.md"
    mock_open.assert_any_call(note_path, 'w')
    # Ensure start_research called with new query and force_index
    mock_start.assert_called_once()
    call_kwargs = mock_start.call_args.kwargs
    assert 'query' in call_kwargs and 'force_index' in call_kwargs
    assert call_kwargs['query'].startswith("Summarize abstract and key contributions of")
    assert str((PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf").resolve()) in call_kwargs['force_index']

@patch('requests.get')
@patch.object(Path, 'mkdir')
def test_fetch_arxiv_paper_pdf_download_fails(
    mock_mkdir, mock_get, mock_arxiv_paths, caplog, monkeypatch
):
    arxiv_id = "2310.00000"

    mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found for url")

    # Monkeypatch DocInsightClient methods
    mock_start = MagicMock(return_value='job1')
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', mock_start)
    mock_wait = MagicMock(return_value={'answer':'ok','novelty':0.5})
    monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', mock_wait)

    result = fetch_arxiv_paper(arxiv_id)
    assert result is False
    assert f"Failed to download PDF for {arxiv_id}" in caplog.text

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch.object(Path, 'mkdir')
def test_fetch_arxiv_paper_metadata_fetch_fails(
    mock_mkdir, mock_open, mock_get, mock_arxiv_paths, caplog, monkeypatch
):
    arxiv_id = "2310.04822"
    mock_pdf_response = MagicMock()
    mock_pdf_response.status_code = 200
    mock_pdf_response.content = b"dummy pdf content"
    mock_get.side_effect = [
        mock_pdf_response,
        requests.exceptions.ConnectionError("API connection failed")
    ]

    # Monkeypatch DocInsightClient methods
    mock_start = MagicMock(return_value='job1')
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', mock_start)
    mock_wait = MagicMock(return_value={'answer':'ok','novelty':0.5})
    monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', mock_wait)

    result = fetch_arxiv_paper(arxiv_id)
    assert result is False
    assert f"Failed to fetch metadata for {arxiv_id}" in caplog.text

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch.object(Path, 'mkdir')
def test_fetch_arxiv_paper_files_already_exist(
    mock_mkdir, mock_open, mock_get, mock_arxiv_paths, monkeypatch
):
    """
    Tests that fetch_arxiv_paper does not re-download files if they already exist, but still
    invokes DocInsightClient to start research and update metadata as needed.
    
    Ensures no HTTP requests are made for existing files, and verifies correct arguments are
    passed to DocInsightClient methods even when files are present.
    """
    arxiv_id = "2310.04822"
    cleaned_arxiv_id = "2310.04822"
    pdf_path = PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf"
    metadata_path = METADATA_TEST_DIR / f"{cleaned_arxiv_id}.json"
    note_path = NOTES_TEST_DIR / f"{cleaned_arxiv_id}.md"
    mock_metadata_content = json.dumps({
        "id": cleaned_arxiv_id, "title": "Existing Title", "authors": ["Existing Author"],
        "year": 2022, "abstract": "Existing abstract", "source": "arXiv",
        "pdf_link": "link", "abs_link": "link", "primary_category": "cs.LG",
        "tags": [], "imported_at": datetime.datetime.utcnow().isoformat() + "Z"
    })
    m = mock_open()
    def custom_open_side_effect(path_arg, mode='r'):
        """
        Provides a side effect function for mocking open() calls with custom behavior.
        
        Returns a mock file object that reads predefined metadata content when opening
        the metadata file in read mode; otherwise, returns a generic mock file object.
        """
        if path_arg == metadata_path and mode == 'r':
            return mock_open(read_data=mock_metadata_content).return_value
        return mock_open().return_value
    m.side_effect = custom_open_side_effect
    with patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', m):
        # Monkeypatch DocInsightClient methods
        mock_start = MagicMock(return_value='job1')
        monkeypatch.setattr(fp.DocInsightClient, 'start_research', mock_start)
        mock_wait = MagicMock(return_value={'answer':'ok','novelty':0.5})
        monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', mock_wait)

        result = fetch_arxiv_paper(arxiv_id)
    assert result is True
    mock_get.assert_not_called()
    # Ensure start_research called with new query and force_index even if files exist
    mock_start.assert_called_once()
    call_kwargs = mock_start.call_args.kwargs
    assert 'query' in call_kwargs and 'force_index' in call_kwargs
    assert call_kwargs['query'].startswith("Summarize abstract and key contributions of")
    assert str((PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf").resolve()) in call_kwargs['force_index']

@patch('cultivation.scripts.literature.fetch_paper.fetch_arxiv_paper')
def test_main_function_calls_fetch_arxiv_paper(mock_fetch_arxiv, capsys, monkeypatch):
    test_arxiv_id = "1234.56789"
    monkeypatch.setattr("sys.argv", ["fetch_paper.py", "--arxiv_id", test_arxiv_id])
    fetch_paper_main()
    mock_fetch_arxiv.assert_called_once_with(test_arxiv_id, force_redownload=False)

@patch('cultivation.scripts.literature.fetch_paper.fetch_arxiv_paper')
def test_main_function_no_args(mock_fetch_arxiv, capsys, monkeypatch):
    """
    Tests that the main CLI function exits with an error when no arguments are provided.
    
    Asserts that `fetch_arxiv_paper` is not called and a `SystemExit` is raised.
    """
    monkeypatch.setattr("sys.argv", ["fetch_paper.py"])
    with pytest.raises(SystemExit):
        fetch_paper_main()
    mock_fetch_arxiv.assert_not_called()

# --- New Tests for Phase 1.1.1 and 1.1.3 ---
import pytest
from jsonschema import ValidationError
from cultivation.scripts.literature.docinsight_client import DocInsightTimeoutError

@pytest.fixture(autouse=True)
def reset_dirs(tmp_path, monkeypatch):
    # Override directories to use tmp_path
    """
    Configures test directories for PDFs, metadata, and notes to use a temporary path.
    
    Overrides the default storage directories in the target module with subdirectories under
    the provided pytest `tmp_path` fixture, ensuring test isolation. Returns the base temporary path.
    """
    pdf = tmp_path / 'pdf'
    meta = tmp_path / 'metadata'
    notes = tmp_path / 'notes'
    # Do not pre-create dirs; let code or tests create them
    monkeypatch.setattr('cultivation.scripts.literature.fetch_paper.PDF_DIR', pdf)
    monkeypatch.setattr('cultivation.scripts.literature.fetch_paper.METADATA_DIR', meta)
    monkeypatch.setattr('cultivation.scripts.literature.fetch_paper.NOTES_DIR', notes)
    return tmp_path

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch.object(Path, 'mkdir')
@patch('cultivation.scripts.literature.fetch_paper.validate')
def test_schema_validation_failure(
    mock_validate, mock_mkdir, mock_open, mock_get, reset_dirs, caplog
):
    """
    Tests that fetch_arxiv_paper returns False and logs an error when metadata schema validation fails.
    
    Simulates a ValidationError during metadata validation and verifies that no metadata JSON file is created.
    """
    from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper
    # Force validate() to raise ValidationError
    mock_validate.side_effect = ValidationError('fail')
    caplog.set_level('ERROR')
    # Stub PDF and API requests to succeed before validation error
    mock_pdf = MagicMock(status_code=200, content=b'x'); mock_pdf.raise_for_status=MagicMock()
    api_xml = SAMPLE_ARXIV_XML.encode('utf-8')
    mock_api = MagicMock(status_code=200, content=api_xml); mock_api.raise_for_status=MagicMock()
    mock_get.side_effect = [mock_pdf, mock_api]
    result = fetch_arxiv_paper('0000.00000')
    assert result is False
    assert 'Metadata schema validation failed' in caplog.text
    # Metadata file should not be created
    assert not any(reset_dirs.glob('metadata/*.json'))

from unittest.mock import patch, MagicMock
@patch('requests.get')
def test_docinsight_polling_success(mock_get, reset_dirs, monkeypatch):
    """
    Tests that fetch_arxiv_paper successfully fetches a paper, processes metadata, and updates
    the metadata JSON with DocInsight summary and novelty score when DocInsight polling succeeds.
    """
    from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper
    # Mock GET for PDF and XML
    pdf_resp = MagicMock(status_code=200, content=b'x')
    xml = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1111.1111v1</id><updated>2025-01-01T00:00:00Z</updated><published>2025-01-01T00:00:00Z</published><title>Test</title><summary>Abs</summary><author><name>A</name></author><arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/></entry></feed>'''
    api_resp = MagicMock(status_code=200, content=xml)
    mock_get.side_effect = [pdf_resp, api_resp]
    # Monkeypatch DocInsightClient methods
    mock_start = MagicMock(return_value='job1')
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', mock_start)
    mock_wait = MagicMock(return_value={'answer':'ok','novelty':0.5})
    monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', mock_wait)

    result = fetch_arxiv_paper('1111.1111')
    assert result is True
    # Verify metadata JSON updated with summary/novelty
    import json, os
    meta_files = list((reset_dirs/'metadata').glob('*.json'))
    data = json.loads(meta_files[0].read_text())
    assert data['docinsight_summary']=='ok'
    assert data['docinsight_novelty']==0.5

@patch('requests.get')
def test_docinsight_polling_timeout(mock_get, reset_dirs, caplog, monkeypatch):
    """
    Tests that fetch_arxiv_paper handles DocInsight polling timeouts gracefully.
    
    Simulates a scenario where the DocInsightClient's wait_for_result method raises a
    DocInsightTimeoutError. Verifies that the function logs a warning, returns True,
    and writes the job ID but not the summary to the metadata JSON file.
    """
    from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper
    # Mock GET
    pdf_resp = MagicMock(status_code=200, content=b'x')
    api_xml = SAMPLE_ARXIV_XML.encode('utf-8')
    api_resp = MagicMock(status_code=200, content=api_xml)
    mock_get.side_effect = [pdf_resp, api_resp]
    # Monkeypatch DocInsightClient methods
    mock_start = MagicMock(return_value='job2')
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', mock_start)
    mock_wait = MagicMock(side_effect=DocInsightTimeoutError('timeout'))
    monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', mock_wait)
    caplog.set_level('WARNING')
    result = fetch_arxiv_paper('2222.2222')
    assert result is True
    assert 'timed out waiting for result' in caplog.text
    # Metadata should have job_id but no summary
    import json
    data = json.loads((reset_dirs/'metadata'/ '2222.2222.json').read_text())
    assert data.get('docinsight_summary') is None
    assert data['docinsight_job_id']=='job2'

@patch('cultivation.scripts.literature.fetch_paper.validate')
@patch('requests.get')
def test_schema_validation_skip(mock_get, mock_validate, reset_dirs, caplog):
    """
    Tests that fetch_arxiv_paper skips metadata schema validation and continues when jsonschema.validate raises a non-ValidationError.
    
    Simulates a generic exception during schema validation and verifies that the function logs a warning and returns True.
    """
    # Stub validate to raise generic error
    mock_validate.side_effect = RuntimeError('skip schema')
    # Stub PDF and API responses
    pdf_resp = MagicMock(status_code=200, content=b'x'); pdf_resp.raise_for_status=MagicMock()
    api_resp = MagicMock(status_code=200, content=SAMPLE_ARXIV_XML.encode()); api_resp.raise_for_status=MagicMock()
    mock_get.side_effect = [pdf_resp, api_resp]
    caplog.set_level('WARNING')
    result = fetch_arxiv_paper('9999.9999')
    assert result is True
    assert 'Skipping metadata schema validation' in caplog.text

@patch('requests.get')
def test_docinsight_api_error(mock_get, reset_dirs, caplog, monkeypatch):
    """
    Tests that fetch_arxiv_paper logs an error and returns True if DocInsightClient.start_research raises a DocInsightAPIError.
    """
    # Stub PDF and API responses
    pdf_resp = MagicMock(status_code=200, content=b'x'); pdf_resp.raise_for_status=MagicMock()
    api_resp = MagicMock(status_code=200, content=SAMPLE_ARXIV_XML.encode()); api_resp.raise_for_status=MagicMock()
    mock_get.side_effect = [pdf_resp, api_resp]
    # Stub client.start_research to raise error
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', MagicMock(side_effect=DocInsightAPIError('fail')))
    caplog.set_level('ERROR')
    result = fetch_arxiv_paper('8888.8888')
    assert result is True
    assert 'Failed to submit DocInsight job' in caplog.text

@patch('requests.get')
def test_force_redownload(mock_get, tmp_path, caplog, monkeypatch):
    """
    Tests that setting force_redownload=True causes existing PDF, metadata, and notes files
    to be overwritten with newly downloaded content.
    """
    # Override dirs
    monkeypatch.setenv('LIT_DIR_OVERRIDE', str(tmp_path))
    from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper
    # Create existing files
    pdf = tmp_path / 'pdf'; meta = tmp_path / 'metadata'; notes = tmp_path / 'notes'
    for d in (pdf, meta, notes): d.mkdir()
    arxiv_id = '7777.7777'
    (pdf / f"{arxiv_id}.pdf").write_bytes(b'old')
    (meta / f"{arxiv_id}.json").write_text('{}')
    (notes / f"{arxiv_id}.md").write_text('old')
    # Stub downloads
    pdf_resp = MagicMock(status_code=200, content=b'new'); pdf_resp.raise_for_status=MagicMock()
    api_resp = MagicMock(status_code=200, content=SAMPLE_ARXIV_XML.encode()); api_resp.raise_for_status=MagicMock()
    mock_get.side_effect = [pdf_resp, api_resp]
    # Stub DocInsightClient
    monkeypatch.setenv('DOCINSIGHT_API_URL', '')
    monkeypatch.setattr(fp.DocInsightClient, 'start_research', MagicMock(return_value='job'))
    monkeypatch.setattr(fp.DocInsightClient, 'wait_for_result', MagicMock(return_value={}))
    # Run with force_redownload
    caplog.set_level('INFO')
    result = fetch_arxiv_paper(arxiv_id, force_redownload=True)
    assert result is True
    # Verify new PDF content
    assert (pdf / f"{arxiv_id}.pdf").read_bytes() == b'new'

### CLI tests for main() ###
def test_main_no_args(monkeypatch):
    # Running without any flags should exit with error
    monkeypatch.setattr("sys.argv", ["fetch_paper.py"])
    with pytest.raises(SystemExit):
        fp.main()

def test_main_success(monkeypatch, caplog):
    # Simulate successful fetch_arxiv_paper
    monkeypatch.setattr(fp, 'fetch_arxiv_paper', lambda arxiv_id, force_redownload=False: True)
    monkeypatch.setenv('LIT_DIR_OVERRIDE', '')
    monkeypatch.setattr("sys.argv", ["fetch_paper.py", "--arxiv_id", "1234.5678"])
    caplog.set_level('INFO')
    fp.main()
    assert 'Successfully processed arXiv ID: 1234.5678' in caplog.text

def test_main_failure(monkeypatch, caplog):
    # Simulate failed fetch_arxiv_paper
    monkeypatch.setattr(fp, 'fetch_arxiv_paper', lambda arxiv_id, force_redownload=False: False)
    monkeypatch.setattr("sys.argv", ["fetch_paper.py", "--arxiv_id", "0000.0000"])
    caplog.set_level('ERROR')
    fp.main()
    assert 'Failed to process arXiv ID: 0000.0000' in caplog.text

def test_main_with_force_redownload_flag(monkeypatch, caplog):
    # Ensure force-redownload flag forwards to fetch_arxiv_paper
    """
    Tests that the --force-redownload CLI flag is correctly passed to fetch_arxiv_paper.
    
    Verifies that when the main function is run with the --force-redownload flag, the
    fetch_arxiv_paper function receives force_redownload=True and the correct arXiv ID.
    """
    called = {}
    def fake_fetch(arxiv_id, force_redownload=False):
        """
        Simulates fetching an arXiv paper and records the provided arguments.
        
        Args:
            arxiv_id: The arXiv identifier of the paper to fetch.
            force_redownload: If True, simulates forcing a redownload.
        
        Returns:
            True, indicating the simulated fetch was successful.
        """
        called['id'] = arxiv_id
        called['force'] = force_redownload
        return True
    monkeypatch.setattr(fp, 'fetch_arxiv_paper', fake_fetch)
    monkeypatch.setenv('LIT_DIR_OVERRIDE', '')
    monkeypatch.setattr("sys.argv", ["fetch_paper.py", "--arxiv_id", "9999.9999", "--force-redownload"])
    caplog.set_level('INFO')
    fp.main()
    assert called.get('id') == '9999.9999'
    assert called.get('force') is True
