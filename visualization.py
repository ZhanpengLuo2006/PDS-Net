import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

TITLE_FONT_SIZE = 30

# ── 全局字体 / 风格 ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     1.2,
    'axes.labelsize':     13,
    'axes.titlesize':     TITLE_FONT_SIZE,
    'axes.titleweight':   'bold',
    'axes.titlepad':      10,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'lines.linewidth':    1.8,
    'lines.markersize':   5,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.5,
    'grid.alpha':         0.40,
    'figure.dpi':         600,
    'savefig.dpi':        1200,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.12,
})

# ── 配色 / 线型 / marker ────────────────────────────────────────────────────
MODEL_STYLES = {
    'PDS-Net':          {'color': '#C0392B', 'linestyle': '-',  'marker': 'o'},
    'YOLOv13n':         {'color': '#2980B9', 'linestyle': '--', 'marker': 's'},
    'YOLOv6n':         {'color': '#27AE60', 'linestyle': '-.', 'marker': '^'},
    'YOLOv8n':          {'color': '#8E44AD', 'linestyle': ':',  'marker': 'D'},
    'YOLOv11n':         {'color': '#E67E22', 'linestyle': '-',  'marker': 'v'},
    'YOLOv12n':         {'color': '#16A085', 'linestyle': '--', 'marker': 'P'},
    'RT-DETR-l':        {'color': '#7F8C8D', 'linestyle': '-.', 'marker': 'X'},
    'RT-DETR-ResNet50': {'color': '#2C3E50', 'linestyle': ':',  'marker': '*'},
}
_FALLBACK_PALETTE = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
_FALLBACK_LS      = ['-', '--', '-.', ':']
_FALLBACK_MARKERS = ['o', 's', '^', 'D', 'v']


def get_style(name: str, idx: int) -> dict:
    if name in MODEL_STYLES:
        return MODEL_STYLES[name]
    return {
        'color':     _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)],
        'linestyle': _FALLBACK_LS[idx % len(_FALLBACK_LS)],
        'marker':    _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)],
    }


# ── 列名兼容（YOLO & RT-DETR）──────────────────────────────────────────────
_COL_CANDIDATES: dict = {
    'train_box':  ['train/box_loss', 'train/box_om', 'train/giou_loss'],
    'train_cls':  ['train/cls_loss'],
    'train_dfl':  ['train/dfl_loss', 'train/l1_loss'],
    'val_box':    ['val/box_loss',   'val/box_om',   'val/giou_loss'],
    'val_cls':    ['val/cls_loss'],
    'val_dfl':    ['val/dfl_loss',   'val/l1_loss'],
    'precision':  ['metrics/precision(B)'],
    'recall':     ['metrics/recall(B)'],
    'map50':      ['metrics/mAP50(B)'],
    'map5095':    ['metrics/mAP50-95(B)'],
    'epoch':      ['epoch', 'Epoch'],
}


def get_col(df: pd.DataFrame, key: str):
    cols_clean = {c.strip(): c for c in df.columns}
    for candidate in _COL_CANDIDATES.get(key, [key]):
        if candidate.strip() in cols_clean:
            return cols_clean[candidate.strip()]
    return None


# ── EMA 平滑 ───────────────────────────────────────────────────────────────
def ema_smooth(y: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """alpha 越大平滑越强（接近 1 = 很平滑）"""
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * y[i]
    return out


# ── 单条曲线绘制 ───────────────────────────────────────────────────────────
def draw_curve(ax, x, y_raw, name, style, smooth_alpha=0.0):
    y = ema_smooth(y_raw.astype(float), smooth_alpha) if smooth_alpha > 0 else y_raw.astype(float)
    is_hero = (name == 'PDS-Net')
    lw     = 2.2 if is_hero else 1.5
    zorder = 6   if is_hero else 3
    me     = max(1, len(x) // 15)

    if smooth_alpha > 0:
        ax.plot(x, y_raw, color=style['color'], alpha=0.12, linewidth=0.7, zorder=1)

    ax.plot(
        x, y,
        color           = style['color'],
        linestyle       = style['linestyle'],
        linewidth       = lw,
        marker          = style['marker'],
        markersize      = 5.5 if is_hero else 4.5,
        markevery       = me,
        markerfacecolor = 'white',
        markeredgewidth = 1.4 if is_hero else 1.0,
        markeredgecolor = style['color'],
        zorder          = zorder,
        solid_capstyle  = 'round',
        solid_joinstyle = 'round',
    )


# ── 坐标轴美化 ─────────────────────────────────────────────────────────────
def polish_ax(ax, xlabel='Epoch', ylabel='', title=''):
    ax.set_xlabel(xlabel, labelpad=7, color='#1a1a2e')
    ax.set_ylabel(ylabel, labelpad=7, color='#1a1a2e')
    ax.set_title(title, color='#1a1a2e', fontsize=TITLE_FONT_SIZE)
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, axis='both', color='#DDDDDD', linestyle='--',
            linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#444444')
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#AAAAAA')
        ax.spines[spine].set_linewidth(0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))


# ── 正方形画布 ─────────────────────────────────────────────────────────────
def square_fig(size=5.2):
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor('white')
    return fig, ax


# ── 主函数 ────────────────────────────────────────────────────────────────
def plot_multi_model(
    models:       dict,
    save_dir:     str,
    smooth_alpha: float = 0.0,
    fig_size:     float = 5.2,
    dpi:          int   = 1200,
):
    os.makedirs(save_dir, exist_ok=True)

    # 加载 CSV
    print('\n[INFO] Loading data...')
    dfs: dict = {}
    for name, path in models.items():
        if not os.path.exists(path):
            print(f'  [MISSING] {name} -> {path}')
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            dfs[name] = df
            print(f'  [OK] {name:<22s}  {len(df)} epochs')
        except Exception as e:
            print(f'  [ERROR] {name}: {e}')

    if not dfs:
        print('No valid data found. Exit.')
        return

    styles = {name: get_style(name, i) for i, name in enumerate(dfs)}

    # 10 张图配置: (文件名, metric_key, y轴标签, 标题, y轴是否限 0-1)
    plots = [
        ('01_train_box_loss.png',  'train_box', 'Loss',           'Train Box / GIoU Loss',  False),
        ('02_train_cls_loss.png',  'train_cls', 'Loss',           'Train Cls Loss',          False),
        ('03_train_dfl_loss.png',  'train_dfl', 'Loss',           'Train DFL / L1 Loss',     False),
        ('04_val_box_loss.png',    'val_box',   'Loss',           'Val Box / GIoU Loss',     False),
        ('05_val_cls_loss.png',    'val_cls',   'Loss',           'Val Cls Loss',            False),
        ('06_val_dfl_loss.png',    'val_dfl',   'Loss',           'Val DFL / L1 Loss',       False),
        ('07_precision.png',       'precision', 'Precision',      'Precision',               True),
        ('08_recall.png',          'recall',    'Recall',         'Recall',                  True),
        ('09_mAP50.png',           'map50',     'mAP@0.5',        'mAP@0.5',                 True),
        ('10_mAP50-95.png',        'map5095',   'mAP@0.5:0.95',  'mAP@0.5:0.95',            True),
    ]

    print(f'\n[INFO] Save dir: {save_dir}')
    print('=' * 55)

    for fname, key, ylabel, title, clip01 in plots:
        fig, ax = square_fig(fig_size)
        drawn = 0

        for name, df in dfs.items():
            epoch_col = get_col(df, 'epoch')
            ycol      = get_col(df, key)
            if epoch_col is None or ycol is None:
                continue

            mask = df[ycol].notna()
            x    = df.loc[mask, epoch_col].values
            y    = df.loc[mask, ycol].values
            if len(x) < 2:
                continue

            draw_curve(ax, x, y, name, styles[name], smooth_alpha)
            drawn += 1

        polish_ax(ax, ylabel=ylabel, title=title)
        if clip01:
            ax.set_ylim(bottom=0, top=1.05)
        if drawn == 0:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#BBBBBB')

        out = os.path.join(save_dir, fname)
        fig.savefig(out, dpi=dpi, bbox_inches='tight',
                    facecolor='white', pad_inches=0.12)
        plt.close(fig)
        print(f'  [OK] {fname}')

    _save_legend(styles, save_dir, dpi, fig_size)
    _print_summary(dfs)
    print(f'\nDone. Generated {len(plots)} charts + 1 legend.')


# ── 单独图例 ──────────────────────────────────────────────────────────────
def _save_legend(styles: dict, save_dir: str, dpi=1200, fig_size=5.2):
    """将图例单独保存为正方形图片"""
    handles = [
        Line2D([0], [0],
               label           = name,
               color           = s['color'],
               linestyle       = s['linestyle'],
               linewidth       = 2.2,
               marker          = s['marker'],
               markerfacecolor = 'white',
               markeredgewidth = 1.4,
               markeredgecolor = s['color'])
        for name, s in styles.items()
    ]

    if not handles:
        return

    n    = len(handles)
    ncol = (n + 1) // 2   # 2 行，列数自适应（横向排列）

    fig = plt.figure(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor('white')

    leg = fig.legend(
        handles       = handles,
        loc           = 'center',
        ncol          = ncol,
        frameon       = False,
        fontsize      = 12,
        handlelength  = 3.0,
        handleheight  = 1.4,
        handletextpad = 0.7,
        labelspacing  = 0.9,
        columnspacing = 1.6,
        borderpad     = 1.2,
    )
    fig.canvas.draw()

    out = os.path.join(save_dir, 'legend_models.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight',
                bbox_extra_artists=[leg], facecolor='white', pad_inches=0.15)
    plt.close(fig)
    print('  [OK] legend_models.png')


# ── 汇总表 ───────────────────────────────────────────────────────────────
def _print_summary(dfs: dict):
    metrics = [
        ('Precision',     'precision', 'max'),
        ('Recall',        'recall',    'max'),
        ('mAP@0.5',       'map50',     'max'),
        ('mAP@0.5:0.95',  'map5095',   'max'),
    ]
    sep = '=' * 90
    print('\n' + sep)
    print('Model metric summary')
    print(sep)
    header = f"  {'Model':<22s}"
    for m, _, _ in metrics:
        header += f" | {m:>14s}"
    print(header)
    print('  ' + '-' * 80)
    for name, df in dfs.items():
        row = f'  {name:<22s}'
        for _, key, mode in metrics:
            col = get_col(df, key)
            if col:
                valid = df[col].dropna()
                if len(valid) > 0:
                    best = valid.max() if mode == 'max' else valid.min()
                    row += f' | {best:>14.4f}'
                else:
                    row += f" | {'N/A':>14s}"
            else:
                row += f" | {'N/A':>14s}"
        print(row)
    print(sep)


# ── 入口 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    models = {
        'PDS-Net':          r'H:\Project\cropper\experient results\ablation\DSC3k2_PPA+DySample\results.csv',
        'YOLOv13n':         r'H:\Project\cropper\experient results\baseline\13n2\results.csv',
        'YOLOv6n':         r'H:\Project\cropper\experient results\baseline\6n3\results.csv',
        'YOLOv8n':          r'H:\Project\cropper\experient results\baseline\8n\results.csv',
        'YOLOv11n':         r'H:\Project\cropper\experient results\baseline\11n\results.csv',
        'YOLOv12n':         r'H:\Project\cropper\experient results\baseline\12n\results.csv',
        'RT-DETR-l':        r'H:\Project\cropper\experient results\baseline\rtdetr-l\results.csv',
        'RT-DETR-ResNet50': r'H:\Project\cropper\experient results\baseline\rtdetr-resnet50\results.csv',
    }

    save_dir = r'H:\Project\cropper\comparison_charts'
    plot_multi_model(models, save_dir)
