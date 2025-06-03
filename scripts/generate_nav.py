"""
Script to automatically generate MkDocs navigation structure and table of contents.

This script scans the documentation directory to build a hierarchical navigation tree,
updates mkdocs.yml with the generated structure, and creates an auto-generated ToC
in index.md.
"""

import sys
import subprocess
import re
from pathlib import Path
from ruamel.yaml import YAML

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

def nav_tree(path, relative_base):
    """
    Recursively build navigation tree as (label, value) tuples.

    Args:
        path: Directory path to scan
        relative_base: Base path for calculating relative paths
    
    Returns:
        List of (label, value) tuples where value is either a file path or nested list
    """
    items = []
    # Files first, sorted
    files = sorted([file for file in path.iterdir() if file.is_file() and file.suffix == '.md'])
    for file in files:
        label = file.stem.replace('_', ' ').title()
        items.append((label, str(file.relative_to(relative_base))))
    # Dirs next, sorted
    dirs = sorted([directory for directory in path.iterdir() if directory.is_dir()])
    for directory in dirs:
        child_items = nav_tree(directory, relative_base)
        if child_items:  # Only include non-empty groups
            label = directory.name.replace('_', ' ').title()
            items.append((label, child_items))
    return items


def nav_to_yaml_structure(navigation_items):
    """
    Convert navigation items to YAML structure expected by MkDocs.

    Args:
        navigation_items: List of (label, value) tuples
        
    Returns:
        List of dictionaries suitable for MkDocs nav configuration
    """
    nav_structure = []
    for item_label, item_value in navigation_items:
        if isinstance(item_value, str):
            nav_structure.append({item_label: item_value})
        elif isinstance(item_value, list) and item_value:
            nav_structure.append({item_label: nav_to_yaml_structure(item_value)})
    return nav_structure

def nav_to_markdown(items, indent=0, top_level=True):
    """
    Convert navigation items to markdown table of contents format.

    Args:
        items: List of (label, value) tuples
        indent: Current indentation level
        top_level: Whether this is the top level of the hierarchy
    
    Returns:
        List of markdown formatted strings
    """
    md = []
    pad = '  ' * indent
    for label, value in items:
        if isinstance(value, str):
            # File: always show with 📄
            md.append(f"{pad}- [📄 {label}]({value})")
        elif isinstance(value, list) and value:
            # Folder: bold and 📁 if top-level, regular 📁 if nested
            if top_level:
                md.append(f"{pad}- **📁 {label}**")
            else:
                md.append(f"{pad}- 📁 {label}")
            md.extend(nav_to_markdown(value, indent + 1, top_level=False))
    return md

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
    except (yaml.YAMLError, FileNotFoundError, PermissionError) as e:
        # Fallback: clean up duplicate navs using regex, keep only the last one
        print('[WARN] YAML load failed (likely duplicate nav keys). Attempting to clean up...')
        text = mkdocs_path.read_text(encoding='utf-8')
        import re
        nav_blocks = list(re.finditer(r'^nav:\n[\s\S]*?(?=^\w|^plugins:|\Z)', text, re.MULTILINE))
        if len(nav_blocks) > 1:
            # Remove all but the last nav block
            last_nav = nav_blocks[-1]
            cleaned = text[:nav_blocks[0].start()] + text[last_nav.start():]
            mkdocs_path.write_text(cleaned, encoding='utf-8')
            print(f'[INFO] Removed {len(nav_blocks)-1} duplicate nav section(s). Retrying YAML load...')
        else:
            print('[ERROR] Could not find duplicate navs, aborting.')
            raise e
        with mkdocs_path.open('r') as f:
            data = yaml.load(f)
    data['nav'] = nav_struct
    with mkdocs_path.open('w', encoding='utf-8') as f:
        yaml.dump(data, f)
    print('[INFO] mkdocs.yml nav section replaced (ruamel.yaml, no duplicates).')

    # --- Markdown ToC generation ---
    toc_md = ['<!-- AUTO-TOC-START -->',
              '# Cultivation Documentation',
              '',
              '> **Comprehensive Table of Contents (Auto-generated)**',
              '',
              'Below is a hierarchical overview of all documentation sections, subfolders, and key files. Click any link to jump directly to that document.',
              '',
              '---',
              '']
    toc_md += nav_to_markdown(filtered_nav_items, indent=0)
    toc_md.append('<!-- AUTO-TOC-END -->')
    toc_md_str = '\n'.join(toc_md) + '\n'

    index_path = docs_dir / 'index.md'
    import re
    if index_path.exists():
        index_text = index_path.read_text()
        # Always remove everything except the auto-TOC block
        if '<!-- AUTO-TOC-START -->' in index_text and '<!-- AUTO-TOC-END -->' in index_text:
            # Replace everything between and including the markers
            new_index = re.sub(r'(?s)^.*?<!-- AUTO-TOC-START -->(.|\n)*?<!-- AUTO-TOC-END -->.*$', toc_md_str, index_text)
        else:
            # Overwrite entire file with ToC
            new_index = toc_md_str
        index_path.write_text(new_index)
        print('[INFO] index.md auto-ToC replaced (idempotent).')
    else:
        index_path.write_text(toc_md_str)
        print('[INFO] index.md created with auto-ToC.')

    # Run mkdocs build --strict
    print('[INFO] Running mkdocs build --strict...')
    result = subprocess.run(['mkdocs', 'build', '--strict'], capture_output=True, text=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode == 0:
        print('[SUCCESS] mkdocs build completed successfully.')
    else:
        print('[ERROR] mkdocs build failed.')
