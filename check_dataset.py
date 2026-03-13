import os
import numpy as np


def check_dataset():
    """Check dataset structure and shapes"""
    base_dir = "app/dataset/train/pose"

    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return

    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    print(f"\n📁 Found {len(classes)} classes")
    print(f"   First 10: {classes[:10]}")

    total_files = 0
    shapes = {}

    for cls in classes[:5]:  # Check first 5 classes
        cls_path = os.path.join(base_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
        total_files += len(files)

        print(f"\n📂 {cls}: {len(files)} files")

        # Check first 3 files
        for file in files[:3]:
            file_path = os.path.join(cls_path, file)
            data = np.load(file_path)
            shape_str = str(data.shape)
            shapes[shape_str] = shapes.get(shape_str, 0) + 1
            print(f"  📄 {file}: shape {data.shape}")

    print(f"\n📊 Total files checked: {total_files}")
    print("\n📊 Shape distribution:")
    for shape, count in shapes.items():
        print(f"  {shape}: {count} files")


if __name__ == "__main__":
    check_dataset()