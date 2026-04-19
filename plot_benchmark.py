import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data from benchmark
models = ['FaceNet', 'MobileFaceNet', 'EfficientNet-Lite0']
params = [27.91, 2.39, 4.03]
latency = [13.65, 29.61, 145.05]
fps = [73.3, 33.8, 6.9]
size_mb = [106.47, 9.11, 15.36]
eff_score = [380.94, 70.70, 584.17]

sns.set_theme(style="whitegrid")

# 1. Bar Graph: Number of Parameters
plt.figure(figsize=(8, 5))
bars = plt.bar(models, params, color=['#e74c3c', '#2ecc71', '#3498db'])
plt.title('Модель параметры (Миллионы)', fontsize=14, pad=15)
plt.ylabel('Параметры (M)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}M', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('graph_parameters.png', dpi=300)
plt.close()

# 2. Bar Graph: FPS (Throughput)
plt.figure(figsize=(8, 5))
bars = plt.bar(models, fps, color=['#e74c3c', '#2ecc71', '#3498db'])
plt.title('Пропускная способность (FPS)', fontsize=14, pad=15)
plt.ylabel('FPS')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('graph_fps.png', dpi=300)
plt.close()

# 3. Bar Graph: Energy Efficiency Score
plt.figure(figsize=(8, 5))
bars = plt.bar(models, eff_score, color=['#e74c3c', '#2ecc71', '#3498db'])
plt.title('Оценка энергоэффективности (Меньше = Лучше)', fontsize=14, pad=15)
plt.ylabel('Score (Params x Latency)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{yval}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('graph_efficiency.png', dpi=300)
plt.close()

# 4. Line Graph: Size vs FPS Trade-off
plt.figure(figsize=(8, 5))
plt.plot(models, latency, marker='o', linestyle='-', linewidth=2, markersize=8, color='#9b59b6', label='Задержка (ms)')
plt.plot(models, size_mb, marker='s', linestyle='--', linewidth=2, markersize=8, color='#f39c12', label='Размер (MB)')
plt.title('Размер vs Задержка', fontsize=14, pad=15)
plt.ylabel('MB / ms')
plt.legend()
for i in range(len(models)):
    plt.text(i, latency[i] + 5, f'{latency[i]}ms', ha='center', color='#9b59b6', fontweight='bold')
    plt.text(i, size_mb[i] - 10, f'{size_mb[i]}MB', ha='center', color='#f39c12', fontweight='bold')
plt.tight_layout()
plt.savefig('graph_line_tradeoff.png', dpi=300)
plt.close()

print("Graphs generated successfully.")
