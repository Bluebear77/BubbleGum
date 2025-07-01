import os
import json
import ijson

# === Configuration ===
input_folder = "."
output_folder = "."
MAX_BYTES = 20_000_000  # 20 MB limit per slice

# List of large files to process
large_files = [
    "tabfact_train_92283.json",
]

# === Helper: estimate JSON-encoded byte size of one entry ===
def get_json_size(obj):
    return len(json.dumps(obj, ensure_ascii=False).encode('utf-8'))

# === Main slicing function ===
def slice_json_stream_by_size(filename):
    input_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0]

    print(f"\n📦 Processing (size-aware slicing): {filename}")
    
    with open(input_path, "rb") as f:
        objects = ijson.items(f, "item")
        batch = []
        batch_size = 0
        part_num = 1
        count = 0

        for obj in objects:
            obj_size = get_json_size(obj)
            # Write batch if adding this object would exceed limit
            if batch_size + obj_size > MAX_BYTES and batch:
                save_name = f"{base_name}_part{part_num}.json"
                save_path = os.path.join(output_folder, save_name)
                with open(save_path, "w", encoding="utf-8") as out_f:
                    json.dump(batch, out_f, ensure_ascii=False, indent=2)
                print(f"  ✅ Saved: {save_path} ({len(batch)} items, ~{batch_size/1e6:.2f} MB)")
                batch = []
                batch_size = 0
                part_num += 1

            batch.append(obj)
            batch_size += obj_size
            count += 1

        # Save any remaining entries
        if batch:
            save_name = f"{base_name}_part{part_num}.json"
            save_path = os.path.join(output_folder, save_name)
            with open(save_path, "w", encoding="utf-8") as out_f:
                json.dump(batch, out_f, ensure_ascii=False, indent=2)
            print(f"  ✅ Saved: {save_path} ({len(batch)} items, ~{batch_size/1e6:.2f} MB)")

    print(f"🎉 Done slicing '{filename}' into {part_num} parts ({count} total entries).")

# === Main runner ===
if __name__ == "__main__":
    for file in large_files:
        slice_json_stream_by_size(file)
