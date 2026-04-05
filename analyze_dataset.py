# analyze_dataset.py
import json
import os
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(dataset_dir="person_dataset"):
    features_dir = f"{dataset_dir}/features"
    if not os.path.exists(features_dir):
        print("数据集不存在")
        return
    stats = {
        'total_samples': 0,
        'persons': set(),
        'angles': [],
        'confidence': []
    }
    for filename in os.listdir(features_dir):
        if filename.endswith('.json'):
            with open(f"{features_dir}/{filename}", 'r') as f:
                data = json.load(f)
            stats['total_samples'] += 1
            stats['persons'].add(data['person_id'])
            angle = data['features']['face_angle'][0]
            stats['angles'].append(angle)
            if 'metadata' in data and 'confidence' in data['metadata']:
                stats['confidence'].append(data['metadata']['confidence'])
    print("="*50)
    print("📊 数据集统计信息")
    print("="*50)
    print(f"总样本数: {stats['total_samples']}")
    print(f"不同人物数: {len(stats['persons'])}")
    print("\n角度分布:")
    angle_counts = Counter(stats['angles'])
    for angle, count in angle_counts.items():
        percentage = count / stats['total_samples'] * 100
        print(f"  - {angle}: {count} ({percentage:.1f}%)")
    if stats['confidence']:
        avg_conf = sum(stats['confidence']) / len(stats['confidence'])
        print(f"\n平均置信度: {avg_conf:.3f}")
    print("="*50)
    if stats['angles']:
        plt.figure(figsize=(10,6))
        angles = list(angle_counts.keys())
        counts = list(angle_counts.values())
        plt.bar(angles, counts)
        plt.title('人物角度分布')
        plt.xlabel('角度')
        plt.ylabel('样本数')
        plt.xticks(rotation=45)
        for i, v in enumerate(counts):
            plt.text(i, v+0.5, str(v), ha='center')
        plt.tight_layout()
        plt.savefig(f"{dataset_dir}/angle_distribution.png")
        print(f"\n📈 图表已保存: {dataset_dir}/angle_distribution.png")

if __name__ == "__main__":
    analyze_dataset()