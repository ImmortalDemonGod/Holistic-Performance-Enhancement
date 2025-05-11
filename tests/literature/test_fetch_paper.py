# tests/literature/test_fetch_paper.py
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import datetime
import json
import requests  # Added missing import for requests
# NOTE: If file system mocking becomes more complex, consider pyfakefs for more robust isolation.

from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper, main as fetch_paper_main
from cultivation.scripts.literature.docinsight_client import DocInsightClient

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
@patch.object(DocInsightClient, 'start_research')
def test_fetch_arxiv_paper_success(
    mock_start_research, mock_mkdir, mock_file_open, mock_requests_get, mock_arxiv_paths
):
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

    mock_requests_get.side_effect = [mock_pdf_response, mock_api_response]

    result = fetch_arxiv_paper(arxiv_id)

    assert result is True
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)
    mock_requests_get.assert_any_call(f"https://arxiv.org/pdf/{cleaned_arxiv_id}.pdf", timeout=30)
    mock_requests_get.assert_any_call(f"http://export.arxiv.org/api/query?id_list={cleaned_arxiv_id}", timeout=30)

    pdf_path = PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf"
    mock_file_open.assert_any_call(pdf_path, 'wb')
    handle_pdf = mock_file_open()
    handle_pdf.write.assert_any_call(b"dummy pdf content")

    metadata_path = METADATA_TEST_DIR / f"{cleaned_arxiv_id}.json"
    mock_file_open.assert_any_call(metadata_path, 'w')
    found_json_write = False
    for call_args in mock_file_open.call_args_list:
        if call_args[0][0] == metadata_path:
            found_json_write = True
            break
    assert found_json_write

    note_path = NOTES_TEST_DIR / f"{cleaned_arxiv_id}.md"
    mock_file_open.assert_any_call(note_path, 'w')
    # Ensure start_research called with new query and force_index
    mock_start_research.assert_called_once()
    call_kwargs = mock_start_research.call_args.kwargs
    assert 'query' in call_kwargs and 'force_index' in call_kwargs
    assert cleaned_arxiv_id in call_kwargs['query']
    assert str((PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf").resolve()) in call_kwargs['force_index']

@patch('requests.get')
@patch.object(Path, 'mkdir')
def test_fetch_arxiv_paper_pdf_download_fails(mock_mkdir, mock_requests_get, caplog, mock_arxiv_paths):
    arxiv_id = "2310.00000"
    mock_requests_get.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found for url")
    result = fetch_arxiv_paper(arxiv_id)
    assert result is False
    assert f"Failed to download PDF for {arxiv_id}" in caplog.text

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch.object(Path, 'mkdir')
def test_fetch_arxiv_paper_metadata_fetch_fails(mock_mkdir, mock_file_open, mock_requests_get, caplog, mock_arxiv_paths):
    arxiv_id = "2310.04822"
    mock_pdf_response = MagicMock()
    mock_pdf_response.status_code = 200
    mock_pdf_response.content = b"dummy pdf content"
    mock_requests_get.side_effect = [
        mock_pdf_response,
        requests.exceptions.ConnectionError("API connection failed")
    ]
    result = fetch_arxiv_paper(arxiv_id)
    assert result is False
    assert f"Failed to fetch metadata for {arxiv_id}" in caplog.text

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch.object(Path, 'mkdir')
@patch.object(DocInsightClient, 'start_research')
def test_fetch_arxiv_paper_files_already_exist(
    mock_start_research, mock_mkdir, mock_file_open, mock_requests_get, mock_arxiv_paths
):
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
        if path_arg == metadata_path and mode == 'r':
            return mock_open(read_data=mock_metadata_content).return_value
        return mock_open().return_value
    m.side_effect = custom_open_side_effect
    with patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', m):
        result = fetch_arxiv_paper(arxiv_id)
    assert result is True
    mock_requests_get.assert_not_called()
    # Ensure start_research called with new query and force_index even if files exist
    mock_start_research.assert_called_once()
    call_kwargs = mock_start_research.call_args.kwargs
    assert 'query' in call_kwargs and 'force_index' in call_kwargs
    assert cleaned_arxiv_id in call_kwargs['query']
    assert str((PDF_TEST_DIR / f"{cleaned_arxiv_id}.pdf").resolve()) in call_kwargs['force_index']

@patch('cultivation.scripts.literature.fetch_paper.fetch_arxiv_paper')
def test_main_function_calls_fetch_arxiv_paper(mock_fetch_arxiv, capsys, monkeypatch):
    test_arxiv_id = "1234.56789"
    monkeypatch.setattr("sys.argv", ["fetch_paper.py", "--arxiv_id", test_arxiv_id])
    fetch_paper_main()
    mock_fetch_arxiv.assert_called_once_with(test_arxiv_id)

@patch('cultivation.scripts.literature.fetch_paper.fetch_arxiv_paper')
def test_main_function_no_args(mock_fetch_arxiv, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["fetch_paper.py"])
    with pytest.raises(SystemExit):
        fetch_paper_main()
    mock_fetch_arxiv.assert_not_called()
