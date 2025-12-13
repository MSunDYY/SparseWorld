import matplotlib.pyplot as plt
import seaborn as sns

# 数据
labels = ["Image Backbone", "Range-Adaptive Perception",
          "State-Conditional Forecasting", "Forecasting", "Others"]
times = [0.026, 0.035, 0.026, 0.015, 0.017]

# 高级淡雅配色：pastel + 白色分割线 MLP
colors = sns.color_palette("pastel", len(times))

plt.figure(figsize=(6, 6))
plt.pie(
    times,

    colors=colors,
    autopct=lambda p: f'{p:.1f}%\n({p*sum(times)/100:.3f}s)',
    startangle=90,
    counterclock=False,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}  # 白色分隔线让视觉更干净
)

plt.title("Module Time Consumption", fontsize=14, fontweight='bold')
# plt.tight_layout()
plt.show()
