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
    end_idx = next(i for i, l in enumerate(lines[1:], 1) if l.strip() == '---')
    fm_text = '\n'.join(lines[1:end_idx])
    fm = yaml.safe_load(fm_text)
    # Extract exercises list
    rest = lines[end_idx+1:]
    start = next((i for i, l in enumerate(rest) if l.strip().lower().startswith('## exercises')), None)
    exercises = []
    if start is not None:
        exer_lines = rest[start+1:]
        exer_text = '\n'.join(exer_lines)
        exercises = yaml.safe_load(exer_text)
    fm['exercises'] = exercises
    return fm

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Convert Markdown workout log to YAML')
    p.add_argument('md_file', help='Path to the .md file')
    p.add_argument('-o', '--output', help='Output YAML path (defaults to md_file.yaml)')
    args = p.parse_args()

    data = parse_markdown(args.md_file)
    out = args.output or args.md_file.rsplit('.', 1)[0] + '.yaml'
    with open(out, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Wrote YAML to {out}')
