#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, pathlib, types, runpy

def ensure_repo_on_path():
    repo = os.path.expanduser('~/yolov5')
    if not os.path.isdir(repo):
        raise SystemExit(f"[ERR] 找不到 YOLOv5 目錄：{repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo

def patch_pathlib_only():
    # 讓舊 .pt 權重引用 pathlib._local 不會炸，及 WindowsPath 在 Linux 可用
    sys.modules.setdefault('pathlib._local', pathlib)
    if os.name != "nt" and hasattr(pathlib, "WindowsPath"):
        try:
            pathlib.WindowsPath = pathlib.PosixPath
        except Exception:
            pass

def stub_optional_packages():
    """
    將 YOLOv5 export 不需要、但會被 import 的套件做最小 stub，
    讓程式能跑過 import 階段。
    """
    import types, sys

    def ensure_stub(fullname: str):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        sys.modules[fullname] = mod
        return mod

    # ---- pandas: 提供 options 與 DataFrame/Series 的最小介面 ----
    pd = ensure_stub('pandas')
    try:
        # options.display.*
        display_ns = types.SimpleNamespace(
            max_columns=10,
            max_rows=60,
            max_colwidth=50
        )
        pd.options = types.SimpleNamespace(display=display_ns)

        # 最小 DataFrame/Series，避免一般屬性/方法呼叫報錯
        class _DF:
            def __init__(self, *a, **k): pass
            def to_csv(self, *a, **k): pass
            def head(self, *a, **k): return self
            def __repr__(self): return "<pd.DataFrame stub>"

        pd.DataFrame = _DF
        pd.Series = _DF
    except Exception:
        pass

    # ---- matplotlib: 提供 pyplot 幾個無作用方法 ----
    mpl = ensure_stub('matplotlib')
    mpl_colors = ensure_stub('matplotlib.colors')
    mpl_ticker = ensure_stub('matplotlib.ticker')
    mpl_transforms = ensure_stub('matplotlib.transforms')

    plt = ensure_stub('matplotlib.pyplot')
    def _noop(*a, **k): pass
    plt.figure = _noop
    plt.plot = _noop
    plt.imshow = _noop
    plt.savefig = _noop
    plt.close = _noop
    plt.title = _noop
    plt.subplot = _noop
    plt.axis = _noop

    # ---- 其他常見可選依賴：做成空模組即可 ----
    for name in [
        'seaborn', 'numexpr', 'bottleneck', 'scipy',
        'albumentations', 'pycocotools', 'thop',
        'cycler', 'pyparsing'
    ]:
        ensure_stub(name)

    # ---- torchvision 巢狀 ----
    ensure_stub('torchvision')
    ensure_stub('torchvision.transforms')
    ensure_stub('torchvision.transforms.functional')



def main():
    import argparse
    ap = argparse.ArgumentParser(description="YOLOv5 .pt -> .onnx 轉檔（Pi / NumPy 2.x 安全版）")
    ap.add_argument('--weights', required=True, help='輸入 .pt，例如 ~/best_lincense.pt')
    ap.add_argument('--imgsz', type=int, default=640, help='輸出 ONNX 輸入尺寸（方形）')
    ap.add_argument('--opset', type=int, default=12, help='ONNX opset（12/13/17…）')
    ap.add_argument('--dynamic', action='store_true', help='動態尺寸')
    ap.add_argument('--simplify', action='store_true', help='若裝了 onnxsim 則簡化圖')
    args = ap.parse_args()

    w = os.path.expanduser(args.weights)
    if not os.path.isfile(w):
        raise SystemExit(f"[ERR] 找不到權重：{w}")

    try:
        import onnx  # 必要
        _ = onnx  # 靜音檢查器
    except Exception:
        raise SystemExit("[ERR] 尚未安裝 onnx，請先：pip install onnx")

    # 1) 先打補丁與塞假模組
    patch_pathlib_only()
    stub_optional_packages()

    # 2) 把本機 yolov5 放到 sys.path 最前
    repo = ensure_repo_on_path()

    # 3) 組好參數，直接跑 yolov5/export.py
    os.environ.setdefault('YOLOV5_AUTOINSTALL', '0')
    os.environ.setdefault('ULTRALYTICS_NO_AUTOINSTALL', '1')

    argv = [
        os.path.join(repo, 'export.py'),
        '--weights', w,
        '--include', 'onnx',
        '--imgsz', str(args.imgsz),
        '--opset', str(args.opset),
    ]
    if args.dynamic:
        argv += ['--dynamic']
    if args.simplify:
        argv += ['--simplify']

    print("[info] 呼叫 yolov5/export.py：", " ".join(argv))
    sys.argv = argv
    runpy.run_path(os.path.join(repo, 'export.py'), run_name='__main__')
    print("\n[OK] 轉檔完成（.onnx 會與 .pt 同資料夾）。")

if __name__ == '__main__':
    main()
