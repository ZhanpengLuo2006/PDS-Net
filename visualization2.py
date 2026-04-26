import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import yaml

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TIMM_FX_DISABLED", "1")

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__dict__.update(
        {
            "init": lambda *args, **kwargs: None,
            "log": lambda *args, **kwargs: None,
            "finish": lambda *args, **kwargs: None,
            "login": lambda *args, **kwargs: None,
            "watch": lambda *args, **kwargs: None,
            "run": None,
            "config": {},
            "__all__": [],
        }
    )
    sys.modules["wandb"] = wandb_stub

ROOT = Path(__file__).resolve().parent
YOLO_ROOT = ROOT / "yolov13-main"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ultralytics import YOLO

warnings.filterwarnings('ignore')

TITLE_FONT_SIZE = 30


# ============================================================
# 全局样式
# ============================================================
def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 30,
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.labelsize': 32,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 15,
        'lines.linewidth': 3.5,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'axes.axisbelow': False,
    })


COLORS = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED',
          '#DB2777', '#0891B2', '#4F46E5', '#EA580C', '#16A34A',
          '#64748B', '#BE185D', '#0D9488', '#B45309']

# 类别名称替换表
NAME_REMAP = {
    'Rice False Smut': 'leaf_scaled',
}


def remap_names(names):
    return [NAME_REMAP.get(n, n) for n in names]


def build_local_data_yaml(src_yaml: Path) -> Path:
    with src_yaml.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    dataset_root = ROOT / 'dataset'
    data['path'] = str(dataset_root)
    data['train'] = 'train/images'
    data['val'] = 'val/images'
    data['test'] = 'test/images'

    fd, temp_path = tempfile.mkstemp(prefix='cropper_vis2_data_', suffix='.yaml')
    os.close(fd)
    temp_yaml = Path(temp_path)
    temp_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return temp_yaml


def format_axes(ax):
    ax.tick_params(axis='both', which='both',
                   top=False, right=False,
                   bottom=True, left=True,
                   direction='out', labelsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.grid(False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xticks_ok = xticks[(xticks > 0) & (xticks <= xlim[1])]
    yticks_ok = yticks[(yticks >= 0) & (yticks <= ylim[1])]

    if len(xticks_ok) > 0:
        ax.set_xticks(xticks_ok)
    if len(yticks_ok) > 0:
        ax.set_yticks(yticks_ok)
        labels = ['0.0' if v == 0 else f'{v:.1f}' for v in yticks_ok]
        ax.set_yticklabels(labels)

    x_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1.0
    y_range = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1.0
    pad = 0.06

    ax.set_xlim(xlim[0], xlim[1] + x_range * pad)
    ax.set_ylim(ylim[0], ylim[1] + y_range * pad)

    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])
    ax.spines['left'].set_bounds(ylim[0], ylim[1])
    ax.spines['left'].set_zorder(100)
    ax.spines['bottom'].set_zorder(100)

    arrow_kw = dict(arrowstyle='->', color='#333333', lw=2.0,
                    shrinkA=0, shrinkB=0, mutation_scale=15)
    ax.annotate('', xy=(xlim[1] + x_range * pad, ylim[0]),
                xytext=(xlim[1], ylim[0]),
                arrowprops=arrow_kw, annotation_clip=False, zorder=100)
    ax.annotate('', xy=(xlim[0], ylim[1] + y_range * pad),
                xytext=(xlim[0], ylim[1]),
                arrowprops=arrow_kw, annotation_clip=False, zorder=100)


# ============================================================
# 从模型验证提取曲线数据
# ============================================================
def extract_curves(model_path, data_yaml):
    print('Loading model and running validation...')
    model = YOLO(model_path)
    results = model.val(data=data_yaml, plots=False, verbose=True)

    names = model.names
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)
    nc = len(class_names)
    print(f'Classes: {nc} -> {class_names}')

    metric = results.box
    curves = {}
    px = np.array(metric.px)

    p_curve = r_curve = None

    if hasattr(metric, 'p_curve') and metric.p_curve is not None:
        p_curve = np.array(metric.p_curve)
    if hasattr(metric, 'r_curve') and metric.r_curve is not None:
        r_curve = np.array(metric.r_curve)

    if p_curve is None and hasattr(metric, 'curves_results'):
        cr = metric.curves_results
        if len(cr) >= 2:
            if isinstance(cr[0], (list, tuple)) and len(cr[0]) == 2:
                px = np.array(cr[0][0]); p_curve = np.array(cr[0][1])
            else:
                p_curve = np.array(cr[0])
            if isinstance(cr[1], (list, tuple)) and len(cr[1]) == 2:
                r_curve = np.array(cr[1][1])
            else:
                r_curve = np.array(cr[1])

    def fix_shape(arr, nc_, n_conf):
        if arr is None:
            return None
        arr = np.array(arr)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.shape[0] == n_conf and arr.shape[1] == nc_:
            return arr.T
        return arr

    n_conf = len(px)
    p_curve = fix_shape(p_curve, nc, n_conf)
    r_curve = fix_shape(r_curve, nc, n_conf)

    if p_curve is not None:
        curves['P'] = {
            'x': px, 'y_per_class': p_curve, 'y_all': p_curve.mean(axis=0),
            'class_names': class_names,
            'x_label': 'Confidence', 'y_label': 'Precision',
            'title': 'Precision-Confidence Curve',
            'fname': 'P_Confidence.png',
        }
    if r_curve is not None:
        curves['R'] = {
            'x': px, 'y_per_class': r_curve, 'y_all': r_curve.mean(axis=0),
            'class_names': class_names,
            'x_label': 'Confidence', 'y_label': 'Recall',
            'title': 'Recall-Confidence Curve',
            'fname': 'R_Confidence.png',
        }

    print(f'Extracted curves: {list(curves.keys())}')
    return curves


# ============================================================
# 绘制单张 Confidence 曲线图（正方形，无图例）
# ============================================================
def plot_confidence_curve(cd, save_path):
    set_plot_style()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')

    y_cls = cd['y_per_class']
    nc    = y_cls.shape[0]

    # 每个类别单独一条线，不加 label
    for i in range(nc):
        ax.plot(cd['x'], y_cls[i], color=COLORS[i % len(COLORS)],
                lw=2.5, alpha=0.7, zorder=1)

    # 所有类别均值线
    x_a, y_a = cd['x'], cd['y_all']
    ax.fill_between(x_a, y_a, alpha=0.08, color='#1a1a1a', zorder=1)
    ax.plot(x_a, y_a, color='#1a1a1a', lw=4.5, zorder=10)

    ax.set_xlabel(cd['x_label'], fontweight='bold')
    ax.set_ylabel(cd['y_label'], fontweight='bold')
    ax.set_title(cd['title'], fontweight='bold', pad=10, fontsize=TITLE_FONT_SIZE)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    format_axes(ax)
    # 无图例

    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'Saved: {save_path}')
    plt.close()


# ============================================================
# 单独生成图例图片（正方形）
# ============================================================
def plot_legend(cd, save_path):
    set_plot_style()

    disp_names = remap_names(cd['class_names'])

    handles = []
    for i, name in enumerate(disp_names):
        handles.append(Line2D([0], [1], color=COLORS[i % len(COLORS)],
                              lw=5, label=name))
    handles.append(Line2D([0], [1], color='#1a1a1a', lw=6, label='All classes'))

    ncol = len(handles)

    fig, ax = plt.subplots(figsize=(max(14, 2.8 * len(handles)), 2.4))
    fig.patch.set_facecolor('white')
    ax.set_visible(False)

    fig.legend(
        handles       = handles,
        loc           = 'center',
        ncol          = ncol,
        frameon       = False,
        fontsize      = 22,
        handlelength  = 2.5,
        handleheight  = 1.2,
        handletextpad = 0.6,
        labelspacing  = 0.7,
        columnspacing = 1.4,
        borderpad     = 1.0,
    )

    fig.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor='white', pad_inches=0.08)
    print(f'Saved legend: {save_path}')
    plt.close()


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':

    # ============================================
    # 配置路径
    # ============================================
    TRAIN_DIR  = ROOT / r'experient results\ablation\DSC3k2_PPA+DySample'
    MODEL_PATH = TRAIN_DIR / r'weights\best.pt'
    DATA_YAML  = ROOT / r'dataset\data.yaml'
    OUTPUT_DIR = ROOT / r'trian_img'
    # ============================================

    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('P-Confidence & R-Confidence Curve Plotter')
    print(f'Model:  {MODEL_PATH}')
    print(f'Data:   {DATA_YAML}')
    print(f'Output: {save_dir}')
    print('=' * 55)

    temp_data_yaml = build_local_data_yaml(Path(DATA_YAML))
    try:
        curves = extract_curves(str(MODEL_PATH), str(temp_data_yaml))

        legend_saved = False
        for key in ['P', 'R']:
            if key in curves:
                plot_confidence_curve(
                    curves[key],
                    save_dir / curves[key]['fname']
                )
                # 只生成一次图例（用 P 曲线的类别信息）
                if not legend_saved:
                    plot_legend(curves[key], save_dir / 'legend.png')
                    legend_saved = True
            else:
                print(f'[WARN] {key} curve not found')
    finally:
        temp_data_yaml.unlink(missing_ok=True)

    print('\nDone!')
    for f in sorted(save_dir.glob('*.png')):
        print(f'  {f.name}')
