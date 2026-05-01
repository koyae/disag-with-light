import pathlib
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Recursively delete files by name pattern.")
    parser.add_argument("pattern", help="The name pattern to match (e.g., 'test' or '*rewire*')")
    parser.add_argument("--root", default=".", help="Root directory to start search (default: current)")
    parser.add_argument("--wet", action="store_true", help="Actually delete the files. Otherwise, just lists them.")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    # Ensure pattern has wildcards if not provided
    search_str = args.pattern if "*" in args.pattern else f"*{args.pattern}*"

    print(f"{'!! EXECUTING DELETION !!' if args.wet else '-- DRY RUN --'}")
    print(f"Searching for: {search_str} in {root.absolute()}\n")

    files_found = list(root.rglob(search_str))
    files_to_delete = [f for f in files_found if f.is_file()]

    if not files_to_delete:
        print("No matching files found.")
        return

    for f in files_to_delete:
        if args.wet:
            try:
                f.unlink()
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
        else:
            print(f"Would delete: {f}")

    if not args.wet:
        print(f"\nFound {len(files_to_delete)} files. Run with '--wet' to delete them.")
    else:
        print(f"\nCleanup complete. {len(files_to_delete)} files processed.")

if __name__ == "__main__":
    main()
