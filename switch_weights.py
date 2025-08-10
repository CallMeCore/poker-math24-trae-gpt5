#!/usr/bin/env python3
import sys
import shutil
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "models"
DEFAULT = ASSETS / "playing_cards_yolov8.pt"

CANDIDATES = {
    "tuned": ASSETS / "yolov8m_tuned.pt",
    "synthetic": ASSETS / "yolov8m_synthetic.pt",
    "coco": ASSETS / "yolov8m.pt",
}

def md5sum(p: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def status():
    if not DEFAULT.exists():
        print(f"[status] {DEFAULT} 不存在")
        return
    default_md5 = md5sum(DEFAULT)
    print(f"[status] 当前默认权重: {DEFAULT.name} (md5={default_md5})")
    matched = []
    for name, p in CANDIDATES.items():
        if p.exists() and md5sum(p) == default_md5:
            matched.append(name)
    if matched:
        print(f"[status] 匹配到候选版本: {', '.join(matched)}")
    else:
        print("[status] 未与任何候选版本匹配（可能是其他权重文件）")

def switch(target: str):
    target = target.lower()
    if target not in CANDIDATES:
        print(f"不支持的目标: {target}，可选值: {', '.join(CANDIDATES.keys())}")
        sys.exit(2)
    src = CANDIDATES[target]
    if not src.exists():
        print(f"源文件不存在: {src}")
        sys.exit(3)
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, DEFAULT)
    print(f"已切换为: {target} -> {DEFAULT}")

def usage():
    print("用法:")
    print("  python switch_weights.py tuned       切到微调版（推荐）")
    print("  python switch_weights.py synthetic   切到合成数据版")
    print("  python switch_weights.py coco        切到通用COCO版")
    print("  python switch_weights.py status      查看当前默认权重匹配情况")

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help", "help"):
        usage()
        return
    cmd = sys.argv[1].lower()
    if cmd == "status":
        status()
    else:
        switch(cmd)

if __name__ == "__main__":
    main()