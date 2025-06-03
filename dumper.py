import os
import sys
from pathlib import Path
from datetime import datetime
import pathspec
import argparse

# Optional: Import tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def load_ignore_patterns(ignore_files=[".gitignore", ".dockerignore", ".gcloudignore"]):
    """Load and combine patterns from multiple ignore files"""
    patterns = []
    for ignore_file in ignore_files:
        if os.path.exists(ignore_file):
            with open(ignore_file, "r") as f:
                raw_patterns = f.read().splitlines()
                # Strip leading './' and empty lines, skip comments
                file_patterns = [
                    pattern[2:] if pattern.startswith("./") else pattern
                    for pattern in raw_patterns
                    if pattern and not pattern.startswith("#")
                ]
                patterns.extend(file_patterns)
        else:
            print(f"No {ignore_file} found. Skipping...")

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def build_file_tree(root_path, spec):
    """
    Traverse the directory and build a list of files and directories, respecting .gitignore.
    """
    file_tree = []
    for path, dirs, files in os.walk(root_path, topdown=True):
        # Compute relative path
        rel_path = os.path.relpath(path, root_path)
        if rel_path == ".":
            rel_path = ""
        else:
            # Convert to POSIX path for consistency
            posix_rel_path = Path(rel_path).as_posix()
            file_tree.append(posix_rel_path + "/")

        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(rel_path, d))]

        for file in files:
            file_rel_path = Path(rel_path, file).as_posix()
            if not spec.match_file(file_rel_path):
                file_tree.append(file_rel_path)
            else:
                print(f"Ignoring file: {file_rel_path}")
    return file_tree


def generate_file_tree_structure(file_tree):
    """
    Generate a hierarchical structure (dictionary) from the file tree list.
    """
    from collections import defaultdict

    tree = lambda: defaultdict(tree)
    root = tree()

    for path in file_tree:
        parts = path.split("/")
        current = root
        for part in parts:
            if part:  # Avoid empty strings from trailing '/'
                current = current[part]

    return root


def format_file_tree(d, prefix=""):
    """
    Convert the hierarchical file tree dictionary into a formatted string.
    """
    lines = []
    sorted_keys = sorted(d.keys())
    for i, key in enumerate(sorted_keys):
        connector = "└── " if i == len(sorted_keys) - 1 else "├── "
        lines.append(prefix + connector + key)
        if d[key]:
            extension = "    " if i == len(sorted_keys) - 1 else "│   "
            lines.extend(format_file_tree(d[key], prefix + extension))
    return lines


def collect_files(file_tree, extensions):
    """
    Collect files with specified extensions.
    """
    return [f for f in file_tree if any(f.endswith(ext) for ext in extensions)]


def write_code_dump(
    file_tree,
    files,
    root_path,
    dump_path="./bak/my_agent_project.md",
    header_content=None,
):
    """
    Write the header, directory tree, and the contents of the specified files to the dump file.
    Also, accumulate the total number of characters for token estimation.
    """
    dump_dir = Path(dump_path).parent
    dump_dir.mkdir(parents=True, exist_ok=True)

    tree_structure = generate_file_tree_structure(file_tree)
    formatted_tree = format_file_tree(tree_structure)

    total_chars = 0  # Initialize character count

    with open(dump_path, "w", encoding="utf-8") as dump_file:
        # Write the Header (if provided)
        if header_content:
            header = "===== User Header =====\n"
            dump_file.write(header)
            total_chars += len(header)

            dump_file.write(header_content + "\n\n")
            total_chars += (
                len(header_content) + 2
            )  # Adding length of two newline characters

        # Write the File Tree
        tree_header = "===== Project File Tree =====\n"
        dump_file.write(tree_header)
        total_chars += len(tree_header)

        tree_content = "\n".join(formatted_tree) + "\n\n"
        dump_file.write(tree_content)
        total_chars += len(tree_content)

        # Write the Contents of Each File
        code_header = "===== Code and Configuration Files =====\n\n"
        dump_file.write(code_header)
        total_chars += len(code_header)

        for file in files:
            file_path = Path(root_path) / file
            header = f"===== {file} =====\n"
            dump_file.write(header)
            total_chars += len(header)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                dump_file.write(content)
                dump_file.write("\n\n")
                total_chars += (
                    len(content) + 2
                )  # Adding length of two newline characters
            except Exception as e:
                error_message = f"Failed to read {file}: {e}\n\n"
                dump_file.write(error_message)
                total_chars += len(error_message)
                print(f"Failed to read {file}: {e}", file=sys.stderr)

    return total_chars


def estimate_tokens(total_chars):
    """
    Estimate the number of tokens based on total characters.
    - Simple Estimation: 1 token ≈ 4 characters
    - Accurate Estimation: Using tiktoken if available
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding for GPT-4 and GPT-3.5-turbo
            encoding = tiktoken.get_encoding("cl100k_base")
            # To simulate the total characters, create a dummy string of 'A's with length total_chars
            dummy_text = "A" * total_chars
            tokens = len(encoding.encode(dummy_text))
        except Exception as e:
            print(f"Error using tiktoken for token estimation: {e}", file=sys.stderr)
            print("Falling back to simple character-based estimation.", file=sys.stderr)
            tokens = total_chars / 4
    else:
        # Simple approximation
        tokens = total_chars / 4
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate a Python project into a single dump file while respecting .gitignore."
    )
    parser.add_argument(
        "header_file",
        nargs="?",
        default=None,
        help="Path to a markdown file containing a header message to include in the dump.",
    )
    parser.add_argument(
        "--extensions", default=".py,.md,.txt", help="File extensions to process"
    )
    args = parser.parse_args()

    target_extensions = args.extensions.split(",")
    root_path = "."

    # Use combined ignore patterns
    spec = load_ignore_patterns()
    file_tree = build_file_tree(root_path, spec)

    collected_files = collect_files(file_tree, target_extensions)

    # Step 4: Define dump_path with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = f"./bak/my_agent_project_{timestamp}.md"

    # Step 5: Read Header Content (if provided)
    header_content = None
    if args.header_file:
        header_path = Path(args.header_file)
        if header_path.is_file():
            try:
                with open(header_path, "r", encoding="utf-8") as header_file:
                    header_content = header_file.read()
            except Exception as e:
                print(
                    f"Failed to read header file '{header_path}': {e}", file=sys.stderr
                )
        else:
            print(
                f"Header file '{header_path}' does not exist or is not a file.",
                file=sys.stderr,
            )

    # Step 6: Write to code.dump (including header and file tree)
    total_chars = write_code_dump(
        file_tree,
        collected_files,
        root_path,
        dump_path=dump_path,
        header_content=header_content,
    )

    # Step 7: Estimate tokens
    tokens = estimate_tokens(total_chars)

    print(
        f"\nProject file tree and selected files have been consolidated into {dump_path}"
    )
    print(f"Estimated Token Count: {int(tokens)} tokens")

    if not TIKTOKEN_AVAILABLE:
        print(
            "\nFor a more accurate token count, consider installing the 'tiktoken' library:"
        )
        print("    pip install tiktoken")
        print("Then, modify the script to enable accurate token estimation.")


if __name__ == "__main__":
    main()
