#!/usr/bin/env python3
"""
docinsight_mock.py - Mock server for DocInsight API endpoints for testing purposes.
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return 'OK'

# In-memory job store
jobs = {}

@app.route('/start_research', methods=['POST'])
def start_research():
    data = request.get_json() or {}
    job_id = f"job_{len(jobs) + 1}"
    # Simulate immediate completion
    jobs[job_id] = {
        'status': 'done',
        'answer': f"Mock summary for paper based on '{data.get('query')}'.",
        'novelty': 0.75
    }
    return jsonify({'job_id': job_id}), 200

@app.route('/get_results', methods=['POST'])
def get_results():
    data = request.get_json() or {}
    job_ids = data.get('job_ids', [])
    results = []
    for jid in job_ids:
        rec = jobs.get(jid)
        if rec:
            res = rec.copy()
            res['job_id'] = jid
            results.append(res)
        else:
            results.append({'job_id': jid, 'status': 'error'})
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8008)
