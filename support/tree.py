#!/usr/bin/env python3
"""
tree_listing.py  ── 打印文件夹的树状结构

用法：
    python tree_listing.py <folder_path>   # 递归打印整个文件结构
    python tree_listing.py <folder_path> -i  # 连同隐藏文件一起打印
    python tree_listing.py <folder_path> -d 3  # 限制最大深度为3层
"""

import os
import argparse
from pathlib import Path


def print_tree(root: Path, prefix: str = "", include_hidden: bool = False, max_depth: int = None, current_depth: int = 0, max_items_per_folder: int = None) -> None:
    """
    递归地打印文件/文件夹结构。

    Args:
        root (Path): 当前遍历到的目录路径
        prefix (str): 前缀字符串，用于显示缩进
        include_hidden (bool): 是否包含以「.」开头的隐藏文件
        max_depth (int): 最大递归深度，None表示无限制
        current_depth (int): 当前递归深度
        max_items_per_folder (int): 每个文件夹最多显示的条目数，None表示无限制
    """
    # 如果达到最大深度，不再递归
    if max_depth is not None and current_depth >= max_depth:
        return
    
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        print(f"{prefix}|-- [权限不足] {root.name}/")
        return

    # 过滤隐藏文件
    if not include_hidden:
        entries = [entry for entry in entries if not entry.name.startswith(".")]
    
    # 分离文件和目录
    dirs = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    
    # 合并文件和目录，先显示文件再显示目录
    all_entries = files + dirs
    total_items = len(all_entries)
    
    # 确定要显示的条目数量
    if max_items_per_folder is not None and total_items > max_items_per_folder:
        items_to_show = all_entries[:max_items_per_folder]
        remaining_count = total_items - max_items_per_folder
        show_ellipsis = True
    else:
        items_to_show = all_entries
        remaining_count = 0
        show_ellipsis = False
    
    # 打印要显示的条目
    for entry in items_to_show:
        if entry.is_file():
            print(f"{prefix}|-- {entry.name}")
        else:
            print(f"{prefix}|-- {entry.name}/")
            # 递归处理目录（只有在没达到最大深度时才递归）
            if max_depth is None or current_depth < max_depth - 1:
                print_tree(entry, prefix + "\t", include_hidden, max_depth, current_depth + 1, max_items_per_folder)
    
    # 如果有省略的条目，显示省略信息
    if show_ellipsis:
        print(f"{prefix}|-- ... (还有 {remaining_count} 个条目)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="打印指定文件夹的树状结构（模仿 Linux tree 命令的最简版）"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="要遍历的根文件夹路径",
    )
    parser.add_argument(
        "-i",
        "--include-hidden",
        action="store_true",
        help="同时显示隐藏文件（以 . 开头的文件/文件夹）",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=None,
        help="最大递归深度（默认无限制）",
    )
    parser.add_argument(
        "-n",
        "--max-items",
        type=int,
        default=None,
        help="每个文件夹最多显示的条目数（默认无限制）",
    )

    args = parser.parse_args()

    root = args.folder.expanduser().resolve()
    if not root.exists():
        parser.error(f"路径不存在: {root}")
    if not root.is_dir():
        parser.error(f"路径不是文件夹: {root}")
    
    if args.max_depth is not None and args.max_depth < 1:
        parser.error("最大深度必须大于等于1")
    
    if args.max_items is not None and args.max_items < 1:
        parser.error("每个文件夹最多显示条目数必须大于等于1")

    print(f"|-- {root.name}/")
    print_tree(root, prefix="\t", include_hidden=args.include_hidden, max_depth=args.max_depth, current_depth=0, max_items_per_folder=args.max_items)


if __name__ == "__main__":
    main()
