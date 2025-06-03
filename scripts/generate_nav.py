import sys
from pathlib import Path

def format_nav(items, indent=2):
    """Format nav as YAML with correct indentation."""
    lines = []
    for label, value in items:
        pad = ' ' * indent
        if isinstance(value, str):
            lines.append(f"{pad}- {label}: {value}")
        elif isinstance(value, list) and value:
            lines.append(f"{pad}- {label}:")
            lines.extend(format_nav(value, indent + 2))
    return lines

def nav_tree(path, rel_base):
    """Recursively build nav as (label, value) tuples. Skip empty groups."""
    items = []
    # Files first, sorted
    files = sorted([f for f in path.iterdir() if f.is_file() and f.suffix == '.md'])
    for f in files:
        label = f.stem.replace('_', ' ').title()
        items.append((label, str(f.relative_to(rel_base))))
    # Dirs next, sorted
    dirs = sorted([d for d in path.iterdir() if d.is_dir()])
    for d in dirs:
        child_items = nav_tree(d, rel_base)
        if child_items:  # Only include non-empty groups
            label = d.name.replace('_', ' ').title()
            items.append((label, child_items))
    return items

import subprocess
from ruamel.yaml import YAML

def nav_to_yaml_structure(nav_items):
    # Recursively convert nav_items to the YAML structure expected by mkdocs
    nav_struct = []
    for label, value in nav_items:
        if isinstance(value, str):
            nav_struct.append({label: value})
        elif isinstance(value, list) and value:
            nav_struct.append({label: nav_to_yaml_structure(value)})
    return nav_struct

if __name__ == '__main__':
    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('cultivation/docs')
    rel_base = docs_dir
    nav_items = nav_tree(docs_dir, rel_base)
    # Remove any nav items that would duplicate Home or Index
    filtered_nav_items = [item for item in nav_items if not (
        (isinstance(item[1], str) and item[1].lower() == 'index.md') or
        (isinstance(item[1], str) and item[0].strip().lower() == 'index')
    )]
    nav_struct = [{'Home': 'index.md'}] + nav_to_yaml_structure(filtered_nav_items)

    mkdocs_path = Path('mkdocs.yml')
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        with mkdocs_path.open('r') as f:
            data = yaml.load(f)
    except Exception as e:
        # Fallback: clean up duplicate navs using regex, keep only the last one
        print('[WARN] YAML load failed (likely duplicate nav keys). Attempting to clean up...')
        text = mkdocs_path.read_text()
        import re
        nav_blocks = list(re.finditer(r'^nav:\n[\s\S]*?(?=^\w|^plugins:|\Z)', text, re.MULTILINE))
        if len(nav_blocks) > 1:
            # Remove all but the last nav block
            last_nav = nav_blocks[-1]
            cleaned = text[:nav_blocks[0].start()] + text[last_nav.start():]
            mkdocs_path.write_text(cleaned)
            print(f'[INFO] Removed {len(nav_blocks)-1} duplicate nav section(s). Retrying YAML load...')
        else:
            print('[ERROR] Could not find duplicate navs, aborting.')
            raise e
        with mkdocs_path.open('r') as f:
            data = yaml.load(f)
    data['nav'] = nav_struct
    with mkdocs_path.open('w') as f:
        yaml.dump(data, f)
    print('[INFO] mkdocs.yml nav section replaced (ruamel.yaml, no duplicates).')

    # Run mkdocs build --strict
    print('[INFO] Running mkdocs build --strict...')
    result = subprocess.run(['mkdocs', 'build', '--strict'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode == 0:
        print('[SUCCESS] mkdocs build completed successfully.')
    else:
        print('[ERROR] mkdocs build failed.')
