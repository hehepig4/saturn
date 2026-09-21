#!/usr/bin/env python3
"""Remap train_query docids to a new id_map and convert to BIRDIE train format.

Maps generated query text to the semantic ID of its source table and converts
the question field to the text field expected by BIRDIE's training dataset.
"""
import argparse
import json
import sys


def load_id_map(path):
    """Return dict: tableID -> semantic_id (str)."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {str(item["tableID"]): str(item["semantic_id"]) for item in raw}
    return {str(k): str(v) for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser(description="Remap train_query text_id to new docids")
    ap.add_argument("--train_query", required=True, help="Path to train_query.json (jsonl)")
    ap.add_argument("--id_map", required=True, help="Path to new id_map.json")
    ap.add_argument("--output", required=True, help="Output birdie_train.json (jsonl)")
    args = ap.parse_args()

    id_map = load_id_map(args.id_map)
    n_in = n_out = n_skip = 0
    with open(args.train_query) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            table_id = row.get("tableId", row.get("tableID"))
            text = row.get("question", row.get("text", ""))
            new_id = id_map.get(str(table_id))
            if new_id is None or not text:
                n_skip += 1
                continue
            out = {"text_id": new_id, "text": text, "tableId": table_id}
            if "origin_id" in row:
                out["origin_id"] = row["origin_id"]
            fout.write(json.dumps(out) + "\n")
            n_out += 1

    print(f"remap: in={n_in} out={n_out} skipped={n_skip} tables_in_map={len(id_map)}")
    if n_out == 0:
        sys.exit("ERROR: no rows remapped - check tableId/id_map alignment")


if __name__ == "__main__":
    main()
