#!/usr/bin/env python3
"""
plot_reading_metrics.py - Fetches reading telemetry from the FastAPI endpoint and visualizes reading path and time spent per page.
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

def fetch_metrics(arxiv_id, session_id=None, event_type=None):
    params = {}
    if session_id:
        params['session_id'] = session_id
    if event_type:
        params['event_type'] = event_type
    url = f"{API_BASE}/metrics/{arxiv_id}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def parse_events(events):
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Flatten payload
    payload_df = pd.json_normalize(df['payload'])
    df = pd.concat([df.drop('payload', axis=1), payload_df], axis=1)
    return df

def plot_reading_path(df, arxiv_id):
    if 'page_num' not in df:
        print("No page_num in events; nothing to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['page_num'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Page Number')
    plt.title(f'Reading Path for {arxiv_id}')
    plt.tight_layout()
    plt.show()

def plot_time_on_page(df, arxiv_id):
    if 'page_num' not in df:
        print("No page_num in events; cannot compute time per page.")
        return
    df = df.sort_values('timestamp')
    # Estimate time on page by difference between consecutive view_area_update events
    df_page = df[df['event_type'] == 'view_area_update'].copy()
    df_page['next_time'] = df_page['timestamp'].shift(-1)
    df_page['duration'] = (df_page['next_time'] - df_page['timestamp']).dt.total_seconds()
    time_per_page = df_page.groupby('page_num')['duration'].sum().fillna(0)
    total_time = time_per_page.sum()
    unique_pages = time_per_page.index.nunique()
    print(f"Total estimated reading time: {total_time:.1f} seconds\nPages visited: {sorted(time_per_page.index.tolist())}")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=time_per_page.index, y=time_per_page.values, hue=time_per_page.index, palette='viridis', legend=False)
    plt.xlabel('Page Number')
    plt.ylabel('Estimated Time Spent (s)')
    plt.title(f'Time Spent per Page for {arxiv_id}\nTotal time: {total_time:.1f}s, Unique pages: {unique_pages}')
    plt.tight_layout()
    plt.show()
    # Heatmap overlay if enough data
    if unique_pages > 1 and len(df_page) > unique_pages:
        pivot = df_page.pivot_table(index='page_num', values='duration', aggfunc='sum').fillna(0)
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Time (s)'})
        plt.title(f'Heatmap: Time Spent per Page ({arxiv_id})')
        plt.xlabel('Page Number')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

from rapidfuzz import process, fuzz
import numpy as np
import os


def cluster_similar_texts(texts, threshold=90):
    """Cluster texts using fuzzy matching (rapidfuzz) with a similarity threshold."""
    clusters = []
    used = set()
    for i, t in enumerate(texts):
        if i in used:
            continue
        group = [t]
        used.add(i)
        for j in range(i+1, len(texts)):
            if j in used:
                continue
            score = fuzz.ratio(t, texts[j])
            if score >= threshold:
                group.append(texts[j])
                used.add(j)
        clusters.append(group)
    # Use the longest text in each cluster as the canonical flashcard
    canonical = [max(group, key=len) for group in clusters]
    return canonical, clusters


def assign_page_numbers(sel_df, full_df):
    # For each selection, find the latest view_area_update event before it (by timestamp)
    view_df = full_df[full_df['event_type'] == 'view_area_update'][['timestamp', 'page_num']].copy()
    view_df = view_df.sort_values('timestamp')
    sel_df = sel_df.sort_values('timestamp')
    sel_df['page_num'] = np.nan
    view_idx = 0
    view_times = view_df['timestamp'].tolist()
    view_pages = view_df['page_num'].tolist()
    for i, row in sel_df.iterrows():
        while view_idx + 1 < len(view_times) and view_times[view_idx + 1] <= row['timestamp']:
            view_idx += 1
        if view_times:
            sel_df.at[i, 'page_num'] = view_pages[view_idx]
    return sel_df


def analyze_text_selections(df):
    # Filter for text_selected events
    sel_df = df[df['event_type'] == 'text_selected'].copy()
    if sel_df.empty or 'selected_text' not in sel_df:
        print("No text selections found.")
        return
    # Clean and dedupe selected text
    sel_df['clean_text'] = sel_df['selected_text'].str.strip().replace({r'\s+': ' '}, regex=True)
    # Remove single-character and whitespace-only selections
    sel_df = sel_df[sel_df['clean_text'].str.len() > 1]
    # Assign page numbers
    sel_df = assign_page_numbers(sel_df, df)
    # Cluster similar texts
    canonical, clusters = cluster_similar_texts(sel_df['clean_text'].tolist(), threshold=90)
    # Map canonical to page numbers (most common page in cluster)
    cluster_pages = []
    for group in clusters:
        pages = sel_df[sel_df['clean_text'].isin(group)]['page_num'].dropna().astype(int)
        page = int(pages.mode().iloc[0]) if not pages.empty else None
        cluster_pages.append(page)
    # Count frequencies for canonical selections
    canonical_freq = [sum(sel_df['clean_text'].isin(group)) for group in clusters]
    print(f"\nTotal text selections: {len(sel_df)}")
    print(f"Unique selections after clustering: {len(canonical)}")
    print("\nMost frequently selected text (clustered):")
    for txt, freq in sorted(zip(canonical, canonical_freq), key=lambda x: -x[1])[:5]:
        print(f"{freq} | {txt[:120]}{'...' if len(txt)>120 else ''}")
    # Timeline plot (clustered)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,3))
    plt.plot(sel_df['timestamp'], range(1, len(sel_df)+1), marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Selections')
    plt.title('Timeline of Text Selections')
    plt.tight_layout()
    plt.show()
    # Bar plot of top canonical selections
    top_n = 10
    plt.figure(figsize=(12, max(4, top_n//2)))
    import matplotlib
    matplotlib.rcParams.update({'font.size': 10})
    y_labels = [f"Pg {p}: {txt[:60]}{'...' if len(txt)>60 else ''}" for txt, p in zip(canonical, cluster_pages)]
    sorted_idx = np.argsort(canonical_freq)[::-1][:top_n]
    plt.barh([y_labels[i] for i in sorted_idx], [canonical_freq[i] for i in sorted_idx])
    plt.xlabel('Selection Count')
    plt.title(f'Top {top_n} Most Frequently Selected Text (Clustered)')
    plt.tight_layout()
    plt.show()
    # Output deduped selections for flashcards
    print("\nDeduped selected text for flashcards (clustered):")
    for i, (txt, p) in enumerate(zip(canonical, cluster_pages), 1):
        print(f"{i}. [Pg {p}] {txt[:120]}{'...' if len(txt)>120 else ''}")
    # Export to file
    out_path = os.path.join(os.path.dirname(__file__), 'anki_flashcards.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for txt, p in zip(canonical, cluster_pages):
            f.write(f"[Pg {p}] {txt}\n")
    print(f"\nExported {len(canonical)} flashcard candidates to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot reading metrics for a given arxiv_id.')
    parser.add_argument('arxiv_id', help='arXiv ID of the paper')
    parser.add_argument('--session_id', type=int, default=None, help='Session ID (optional)')
    args = parser.parse_args()

    events = fetch_metrics(args.arxiv_id, args.session_id)
    df = parse_events(events)
    if df.empty:
        print("No events found for this paper/session.")
        return
    plot_reading_path(df, args.arxiv_id)
    plot_time_on_page(df, args.arxiv_id)
    analyze_text_selections(df)

if __name__ == '__main__':
    main()
