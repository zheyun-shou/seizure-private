import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

upscale_factor = 2  # 放大倍数

# 你的 EEG 通道名称顺序（图中19个灰色圆圈）
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 
            'P4', 'T6', 'O1', 'O2']

# 示例相关性值（0到1之间的值）——用你的真实数据替换它
correlations = np.random.rand(19)  # <== 你需要换成真实的 19 个值

# 通道在图片中的相对坐标 (手动定义，根据图中位置大致估计)
channel_coords = {
    'Fp1': (153, 58), 'Fp2': (231, 58),
    'F7': (93, 102), 'F3': (142, 111), 'Fz': (192, 111), 'F4': (242, 111), 'F8': (290, 102),
    'T3': (70, 173), 'C3': (131, 173), 'Cz': (192, 173), 'C4': (253, 173), 'T4': (313, 173),
    'T5': (93, 243), 'P3': (142, 234), 'Pz': (192, 234), 'P4': (242, 234), 'T6': (290, 243),
    'O1': (153, 288), 'O2': (231, 288)
}

# double the values for the channel_coords
# to match the original image size
for key in channel_coords:
    x, y = channel_coords[key]
    channel_coords[key] = (x * upscale_factor, y * upscale_factor)

model_name = '0422_en_mini_datasize0.4'
relevance_dict = {
"FP1": 0.0644905967332404,
"F3": 0.05991598743354701,
"C3": 0.050682509422660574,
"P3": 0.05557462781217641,
"O1": 0.05472645175963871,
"F7": 0.05618738442080273,
"T3": 0.04834609770616016,
"T5": 0.05001799055668087,
"FZ": 0.05003133009499767,
"CZ": 0.05926354927770297,
"PZ": 0.05908585406101323,
"FP2": 0.0499864524612309,
"F4": 0.052887606243530075,
"C4": 0.053015564283829954,
"P4": 0.04996811925677207,
"O2": 0.05484919054371901,
"F8": 0.04685527896606,
"T4": 0.04152751604504264,
"T6": 0.055534578352987654,
} 

# 加载背景图
img = Image.open('./figures/image.png')

# 设置画布
fig, ax = plt.subplots(figsize=(9, 6.86))
ax.imshow(img)
ax.axis('off')

# 自定义 colormap 和 normalization
cmap = cm.viridis
# norm = mcolors.Normalize(vmin=min(relevance_dict.values()), vmax=max(relevance_dict.values()))
norm = mcolors.Normalize(vmin=0, vmax=0.1)


for ch_key in channel_coords:
    rel_key = ch_key.upper()
    if rel_key in relevance_dict:
        value = relevance_dict[rel_key]
        x, y = channel_coords[ch_key]
        color = cmap(norm(value))
        
        # 画圆圈
        circle = patches.Circle((x, y), radius=32, color=color, ec='black', linewidth=3)
        ax.add_patch(circle)
        
        # 添加通道名，居中显示
        ax.text(x, y+2*upscale_factor, ch_key, color='black', fontsize=18, ha='center', va='center')

# 添加 colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('relevance', fontsize=12)

plt.tight_layout()

plt.savefig(f'./figures/channel_relevance/channel_relevance_{model_name}_abs.png', dpi=300)
plt.close()





# print the 8 most relevant channels
sorted_relevance = sorted(relevance_dict.items(), key=lambda x: x[1], reverse=True)
print("Top 8 most relevant channels:") 
for ch, val in sorted_relevance[:8]:
    print(f"{ch}: {val:.4f}")
# print the 8 least relevant channels
print("\nTop 8 least relevant channels:")
for ch, val in sorted_relevance[-8:]:
    print(f"{ch}: {val:.4f}")