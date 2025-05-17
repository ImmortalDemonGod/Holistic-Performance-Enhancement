#!/usr/bin/env python3
"""
Convert a structured Markdown workout log with YAML frontmatter into a single YAML file.
"""
import argparse
import yaml

def parse_markdown(md_path):
    with open(md_path, 'r') as f:
        lines = f.read().splitlines()
    # Extract frontmatter
    if not lines or lines[0].strip() != '---':
        raise ValueError('No YAML frontmatter found')
    try:
        end_idx = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == '---')
    except StopIteration:
        raise ValueError('Missing closing frontmatter marker "---"')
    fm_text = '\n'.join(lines[1:end_idx])
    fm = yaml.safe_load(fm_text)
    # Sanitize session_notes: join lines if multiline
    if 'session_notes' in fm and isinstance(fm['session_notes'], str):
        if '\n' in fm['session_notes']:
            print('[WARNING] session_notes contained line breaks; joining into a single line.')
            fm['session_notes'] = ' '.join(fm['session_notes'].splitlines())
    # Extract exercises list
    rest = lines[end_idx+1:]
    start = next((i for i, line in enumerate(rest) if line.strip().lower().startswith('## exercises')), None)
    exercises = []
    if start is not None:
        raw_exer_lines = rest[start+1:]
        # Collect YAML list until next header or end
        filtered = []
        for line in raw_exer_lines:
            if not line.strip():
                continue
            if line.startswith('#'):
                break
            filtered.append(line)
        exer_text = '\n'.join(filtered)
        try:
            exercises = yaml.safe_load(exer_text)
            if not isinstance(exercises, list):
                print(f"Warning: Exercises section did not parse as a list. Got {type(exercises)}")
                exercises = []
        except yaml.YAMLError as e:
            print(f"Error parsing exercises section: {e}")
            exercises = []
    fm['exercises'] = exercises
    return fm

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Convert Markdown workout log to YAML')
    p.add_argument('md_file', help='Path to the .md file')
    p.add_argument('-o', '--output', help='Output YAML path (defaults to md_file.yaml)')
    args = p.parse_args()

    import os.path
    if not os.path.exists(args.md_file):
        print(f"Error: Input file '{args.md_file}' does not exist")
        exit(1)

    data = parse_markdown(args.md_file)
    # Handle files with or without extensions
    out = args.output
    if not out:
        if '.' in os.path.basename(args.md_file):
            out = args.md_file.rsplit('.', 1)[0] + '.yaml'
        else:
            out = args.md_file + '.yaml'
    with open(out, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Wrote YAML to {out}')
