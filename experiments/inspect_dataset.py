import json
from collections import Counter
from pathlib import Path, PurePosixPath


dataset_path = Path("data/repository_files.jsonl")

records = []

with dataset_path.open("r", encoding="utf-8") as dataset_file:
    for line in dataset_file:
        if line.strip():
            records.append(json.loads(line))


file_types = Counter()
total_size = 0
empty_files = 0

for record in records:
    path = PurePosixPath(record["file_path"])

    if path.name == "Dockerfile":
        file_type = "Dockerfile"
    else:
        file_type = path.suffix.lower() or "no extension"

    file_types[file_type] += 1
    total_size += record["size"]

    if not record["content"].strip():
        empty_files += 1


largest_files = sorted(
    records,
    key=lambda record: record["size"],
    reverse=True,
)[:10]


print(f"Total files: {len(records)}")
print(f"Total size: {total_size / 1024:.2f} KB")
print(f"Empty files: {empty_files}")

print("\nFiles by type:")

for file_type, count in file_types.most_common():
    print(f"  {file_type}: {count}")

print("\nLargest files:")

for record in largest_files:
    print(
        f"  {record['size'] / 1024:.2f} KB"
        f"  {record['file_path']}"
    )