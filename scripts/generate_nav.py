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

if __name__ == '__main__':
    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('cultivation/docs')
    rel_base = docs_dir
    print('nav:')
    print('  - Home: index.md')
    nav_items = nav_tree(docs_dir, rel_base)
    print('\n'.join(format_nav(nav_items, indent=2)))
