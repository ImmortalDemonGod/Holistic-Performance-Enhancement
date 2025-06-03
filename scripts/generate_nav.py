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
    """
    Formats a list of navigation items into YAML lines with proper indentation.
    
    Args:
        items: A list of (label, value) tuples representing navigation entries. Values may be strings (file paths) or nested lists for sub-navigation.
        indent: The number of spaces to use for indentation at the current level.
    
    Returns:
        A list of YAML-formatted strings representing the navigation structure.
    """
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
    """
    Recursively generates a navigation tree of Markdown files and directories.
    
    Scans the given directory for Markdown files and subdirectories, creating a list of (label, value) tuples where labels are human-readable names and values are relative file paths or nested navigation groups. Empty directories are excluded from the result.
    
    Args:
        path: The root directory to scan.
        rel_base: The base directory for computing relative paths.
    
    Returns:
        A list of (label, value) tuples representing the navigation structure.

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


def nav_to_yaml_structure(nav_items):
    # Recursively convert nav_items to the YAML structure expected by mkdocs
    """
    Converts navigation tuples into a nested YAML structure for MkDocs.
    
    Recursively processes a list of (label, value) tuples, producing a list of dictionaries mapping labels to file paths or nested navigation structures as required by the MkDocs 'nav' configuration.
    """
    nav_struct = []
    for label, value in nav_items:
        if isinstance(value, str):
            nav_struct.append({label: value})
        elif isinstance(value, list) and value:
            nav_struct.append({label: nav_to_yaml_structure(value)})
    return nav_struct

def nav_to_markdown(items, indent=0, top_level=True):
    """
    Converts a navigation tree into a Markdown-formatted table of contents.
    
    Args:
        items: List of (label, value) tuples representing files and directories.
        indent: Current indentation level for nested items.
        top_level: Whether the current items are at the top level of the tree.
    
    Returns:
        A list of Markdown strings representing the hierarchical table of contents, with icons for files and folders.

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
