"""
Generate comprehensive comparison charts for all model architectures.
Produces: accuracy comparison, F1 score comparison, confusion matrices,
training curves, model complexity comparison, per-breed performance radar chart.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ========== Style Setup ==========
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'efficientnet': '#2196F3',
    'resnet18': '#FF9800',
    'resnet34': '#9C27B0',
    'baseline_v1': '#F44336',
    'cow': '#4CAF50',
    'buffalo': '#FF5722',
}
BREED_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Load Data ==========
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_json(path):
    with open(os.path.join(BASE, path), 'r') as f:
        return json.load(f)

# EfficientNet V2 results (best models)
cow_eval = load_json('results/evaluation_v2/enhanced_metrics.json')
buffalo_eval = load_json('results/buffalo_evaluation/enhanced_metrics.json')
cow_history = load_json('models/classification/cow_classifier_v2/history.json')
buffalo_history = load_json('models/classification/buffalo_classifier_v1/history.json')

# ResNet18 results
resnet18_cow = load_json('models/classification/resnet18_cow_v1/metrics.json')
resnet18_buffalo = load_json('models/classification/resnet18_buffalo_v1/metrics.json')

# Baseline V1 (from docs - EfficientNet before data expansion)
baseline_v1 = {
    'accuracy': 75.65,
    'per_class': {'gir': 91.11, 'red_sindhi': 30.0, 'sahiwal': 80.0},
    'f1_macro': 0.72  # approximate
}

print("=" * 60)
print("GENERATING COMPARISON CHARTS")
print("=" * 60)

# ========== Chart 1: Overall Accuracy Bar Chart ==========
print("\n[1/8] Overall Accuracy Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

models = ['EfficientNet-B0\nBaseline V1\n(947 imgs)', 
          'ResNet18\nCow V1\n(3394 imgs)',
          'EfficientNet-B0\nCow V2 ⭐\n(6788 imgs)',
          'ResNet18\nBuffalo\n(6 breeds)',
          'EfficientNet-B0\nBuffalo V1 ⭐\n(686 imgs, 3 breeds)']
accuracies = [75.65, 87.28, 98.85, 38.37, 95.96]
colors = [COLORS['baseline_v1'], COLORS['resnet18'], COLORS['efficientnet'], 
          COLORS['resnet34'], COLORS['efficientnet']]

bars = ax.bar(models, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.65)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Accuracy Comparison — Cow & Buffalo Breed Classification', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 110)
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% Target')
ax.legend(fontsize=12)
ax.tick_params(axis='x', labelsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_overall_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 01_overall_accuracy_comparison.png")

# ========== Chart 2: Per-Breed Accuracy (Cows) ==========
print("[2/8] Per-Breed Cow Accuracy...")

fig, ax = plt.subplots(figsize=(12, 7))

breeds = ['Gir', 'Red Sindhi', 'Sahiwal']
baseline_accs = [91.11, 30.0, 80.0]
resnet18_accs = [93.16, 56.47, 93.64]
effnet_v2_accs = [99.72, 95.60, 99.31]

x = np.arange(len(breeds))
width = 0.25

b1 = ax.bar(x - width, baseline_accs, width, label='EfficientNet Baseline V1 (947 imgs)', 
            color=COLORS['baseline_v1'], edgecolor='white', linewidth=1.5)
b2 = ax.bar(x, resnet18_accs, width, label='ResNet18 (3394 imgs)', 
            color=COLORS['resnet18'], edgecolor='white', linewidth=1.5)
b3 = ax.bar(x + width, effnet_v2_accs, width, label='EfficientNet V2 ⭐ (6788 imgs)', 
            color=COLORS['efficientnet'], edgecolor='white', linewidth=1.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Per-Breed Accuracy — Cow Classification', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(breeds, fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=11, loc='upper left')
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_per_breed_cow_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 02_per_breed_cow_accuracy.png")

# ========== Chart 3: F1 Score Comparison ==========
print("[3/8] F1 Score Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Cow F1 Scores
cow_breeds = ['Gir', 'Red Sindhi', 'Sahiwal', 'Macro Avg']
effnet_f1_cow = [99.3, 96.5, 99.3, 98.4]
resnet18_f1_cow = [91.7, 69.1, 88.9, 83.2]
baseline_f1_cow = [93.0, 28.0, 78.0, 66.3]  # approximate from v1

x = np.arange(len(cow_breeds))
width = 0.25

axes[0].bar(x - width, baseline_f1_cow, width, label='Baseline V1', color=COLORS['baseline_v1'], edgecolor='white')
axes[0].bar(x, resnet18_f1_cow, width, label='ResNet18', color=COLORS['resnet18'], edgecolor='white')
axes[0].bar(x + width, effnet_f1_cow, width, label='EfficientNet V2 ⭐', color=COLORS['efficientnet'], edgecolor='white')

for i, (b, r, e) in enumerate(zip(baseline_f1_cow, resnet18_f1_cow, effnet_f1_cow)):
    axes[0].text(i - width, b + 1, f'{b:.0f}', ha='center', fontsize=9, fontweight='bold')
    axes[0].text(i, r + 1, f'{r:.0f}', ha='center', fontsize=9, fontweight='bold')
    axes[0].text(i + width, e + 1, f'{e:.0f}', ha='center', fontsize=9, fontweight='bold')

axes[0].set_title('F1 Score — Cow Breeds', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cow_breeds, fontsize=11)
axes[0].set_ylim(0, 115)
axes[0].set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

# Buffalo F1 Scores
buffalo_breeds = ['Jaffarabadi', 'Mehsana', 'Murrah', 'Macro Avg']
effnet_f1_buf = [98.3, 91.3, 96.8, 95.5]

bars = axes[1].bar(buffalo_breeds, effnet_f1_buf, color=COLORS['efficientnet'], 
                   edgecolor='white', linewidth=1.5, width=0.5)
for bar, f1 in zip(bars, effnet_f1_buf):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{f1:.1f}%', ha='center', fontsize=11, fontweight='bold')

axes[1].set_title('F1 Score — Buffalo Breeds (EfficientNet V1)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 115)
axes[1].set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold')

plt.suptitle('F1 Score Comparison Across All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_f1_score_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 03_f1_score_comparison.png")

# ========== Chart 4: Precision / Recall / F1 Heatmap ==========
print("[4/8] Precision-Recall-F1 Heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Cow heatmap
cow_data = np.array([
    [98.9, 99.7, 99.3],   # Gir
    [97.4, 95.6, 96.5],   # Red Sindhi
    [99.3, 99.3, 99.3],   # Sahiwal
])
sns.heatmap(cow_data, annot=True, fmt='.1f', cmap='YlGn', vmin=90, vmax=100,
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=['Gir', 'Red Sindhi', 'Sahiwal'],
            ax=axes[0], annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=2, linecolor='white')
axes[0].set_title('Cow V2 — EfficientNet-B0 (98.85%)', fontsize=13, fontweight='bold')

# Buffalo heatmap
buf_data = np.array([
    [96.7, 100.0, 98.3],  # Jaffarabadi
    [95.5,  87.5, 91.3],  # Mehsana
    [95.7,  97.8, 96.8],  # Murrah
])
sns.heatmap(buf_data, annot=True, fmt='.1f', cmap='YlGn', vmin=85, vmax=100,
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=['Jaffarabadi', 'Mehsana', 'Murrah'],
            ax=axes[1], annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=2, linecolor='white')
axes[1].set_title('Buffalo V1 — EfficientNet-B0 (95.96%)', fontsize=13, fontweight='bold')

plt.suptitle('Precision / Recall / F1-Score Heatmaps', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_precision_recall_f1_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 04_precision_recall_f1_heatmap.png")

# ========== Chart 5: Confusion Matrices ==========
print("[5/8] Confusion Matrices...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cow confusion matrix
cow_cm = np.array(cow_eval['confusion_matrix'])
sns.heatmap(cow_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Gir', 'Red Sindhi', 'Sahiwal'],
            yticklabels=['Gir', 'Red Sindhi', 'Sahiwal'],
            ax=axes[0], annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white')
axes[0].set_title('Cow V2 — Confusion Matrix (n=953)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)

# Buffalo confusion matrix
buf_cm = np.array(buffalo_eval['confusion_matrix'])
sns.heatmap(buf_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Jaffarabadi', 'Mehsana', 'Murrah'],
            yticklabels=['Jaffarabadi', 'Mehsana', 'Murrah'],
            ax=axes[1], annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white')
axes[1].set_title('Buffalo V1 — Confusion Matrix (n=99)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)

plt.suptitle('Confusion Matrices — Best Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 05_confusion_matrices.png")

# ========== Chart 6: Training Curves ==========
print("[6/8] Training Curves...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Cow V2 - Loss
epochs_cow = range(1, len(cow_history['train_loss']) + 1)
axes[0, 0].plot(epochs_cow, cow_history['train_loss'], 'b-', linewidth=2, label='Train Loss')
axes[0, 0].plot(epochs_cow, cow_history['val_loss'], 'r-', linewidth=2, label='Val Loss')
axes[0, 0].set_title('Cow V2 (EfficientNet) — Loss', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Cow V2 - Accuracy
axes[0, 1].plot(epochs_cow, cow_history['train_acc'], 'b-', linewidth=2, label='Train Acc')
axes[0, 1].plot(epochs_cow, cow_history['val_acc'], 'r-', linewidth=2, label='Val Acc')
axes[0, 1].axhline(y=98.85, color='green', linestyle='--', alpha=0.7, label='Test Acc: 98.85%')
axes[0, 1].set_title('Cow V2 (EfficientNet) — Accuracy', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend(fontsize=11)
axes[0, 1].set_ylim(50, 101)
axes[0, 1].grid(True, alpha=0.3)

# Buffalo V1 - Loss
epochs_buf = range(1, len(buffalo_history['train_loss']) + 1)
axes[1, 0].plot(epochs_buf, buffalo_history['train_loss'], 'b-', linewidth=2, label='Train Loss')
axes[1, 0].plot(epochs_buf, buffalo_history['val_loss'], 'r-', linewidth=2, label='Val Loss')
axes[1, 0].set_title('Buffalo V1 (EfficientNet) — Loss', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Buffalo V1 - Accuracy
axes[1, 1].plot(epochs_buf, buffalo_history['train_acc'], 'b-', linewidth=2, label='Train Acc')
axes[1, 1].plot(epochs_buf, buffalo_history['val_acc'], 'r-', linewidth=2, label='Val Acc')
axes[1, 1].axhline(y=95.96, color='green', linestyle='--', alpha=0.7, label='Test Acc: 95.96%')
axes[1, 1].set_title('Buffalo V1 (EfficientNet) — Accuracy', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].legend(fontsize=11)
axes[1, 1].set_ylim(40, 101)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Training Curves — Best Models (EfficientNet-B0 + Custom Head)', 
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 06_training_curves.png")

# ========== Chart 7: Model Architecture Comparison ==========
print("[7/8] Model Architecture Comparison Infographic...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Table data
table_data = [
    ['Architecture', 'Type', 'Params\n(M)', 'Cow\nAccuracy', 'Buffalo\nAccuracy', 'Cow\nF1', 'Buffalo\nF1', 'MCC'],
    ['EfficientNet-B0\nBaseline V1', 'timm\npretrained', '~5.3', '75.65%', '—', '~0.72', '—', '—'],
    ['ResNet18', 'torchvision\npretrained', '~11.2', '87.28%', '38.37%*', '0.833', '0.337*', '—'],
    ['EfficientNet-B0\nV2 (Best) ⭐', 'timm\ncustom head', '~5.3', '98.85%', '—', '0.984', '—', '0.981'],
    ['EfficientNet-B0\nBuffalo ⭐', 'timm\ncustom head', '~5.3', '—', '95.96%', '—', '0.955', '0.937'],
    ['Custom\nEfficientNet', 'from-scratch\nimplementation', '~5.3', 'Ref.\nOnly', 'Ref.\nOnly', '—', '—', '—'],
    ['ResNet34', 'torchvision\npretrained', '~21.3', 'Trained', 'Trained', '—', '—', '—'],
]

colors_table = [['#E3F2FD'] * 8]  # header
row_colors = ['#FFF3E0', '#FFF3E0', '#E8F5E9', '#E8F5E9', '#F3E5F5', '#F3E5F5']
for c in row_colors:
    colors_table.append([c] * 8)

table = ax.table(cellText=table_data, cellColours=colors_table,
                 cellLoc='center', loc='center',
                 colWidths=[0.15, 0.12, 0.08, 0.10, 0.10, 0.09, 0.09, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Bold header
for j in range(8):
    table[0, j].set_text_props(fontweight='bold', fontsize=11)
    table[0, j].set_facecolor('#1976D2')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=10)

# Highlight best rows
for j in range(8):
    table[3, j].set_edgecolor('#4CAF50')
    table[3, j].set_linewidth(2)
    table[4, j].set_edgecolor('#4CAF50')
    table[4, j].set_linewidth(2)

ax.set_title('Model Architecture Comparison\n* ResNet18 Buffalo was trained on 6 breeds (not comparable to 3-breed EfficientNet)',
             fontsize=14, fontweight='bold', pad=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_model_architecture_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 07_model_architecture_comparison.png")

# ========== Chart 8: Improvement Journey ==========
print("[8/8] Improvement Journey...")

fig, ax = plt.subplots(figsize=(14, 7))

# Data points showing V1 -> ResNet18 -> V2 journey for each breed
stages = ['Baseline V1\n(947 imgs)', 'ResNet18\n(3394 imgs)', 'EfficientNet V2\n(6788 imgs)']
x = [0, 1, 2]

gir_journey = [91.11, 93.16, 99.72]
red_sindhi_journey = [30.0, 56.47, 95.60]
sahiwal_journey = [80.0, 93.64, 99.31]

ax.plot(x, gir_journey, 'o-', linewidth=3, markersize=12, label='Gir', color='#2196F3')
ax.plot(x, red_sindhi_journey, 's-', linewidth=3, markersize=12, label='Red Sindhi', color='#F44336')
ax.plot(x, sahiwal_journey, '^-', linewidth=3, markersize=12, label='Sahiwal', color='#4CAF50')

# Add value labels
for journey, color in [(gir_journey, '#2196F3'), (red_sindhi_journey, '#F44336'), (sahiwal_journey, '#4CAF50')]:
    for i, val in enumerate(journey):
        ax.annotate(f'{val:.1f}%', (i, val), textcoords="offset points", 
                    xytext=(0, 15), ha='center', fontsize=12, fontweight='bold', color=color)

# Highlight huge Red Sindhi improvement
ax.annotate('', xy=(2, 95.6), xytext=(0, 30),
            arrowprops=dict(arrowstyle='->', color='#F44336', lw=2, ls='dashed'))
ax.text(1, 25, '+65.6% improvement! 🚀', fontsize=13, ha='center', 
        color='#F44336', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#F44336'))

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy Improvement Journey — Cow Breeds\nFrom Baseline to Production-Ready', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 115)
ax.legend(fontsize=13, loc='center right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_improvement_journey.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved 08_improvement_journey.png")

# ========== Summary ==========
print("\n" + "=" * 60)
print("ALL CHARTS GENERATED!")
print("=" * 60)
print(f"\nSaved {8} charts to: {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  📊 {f} ({size_kb:.0f} KB)")
