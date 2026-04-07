import os

import os
import random

def get_paired_files_by_prefix(gt_dir: str, hazy_dir: str, target_prefixes: list) -> dict:
    """
    按指定前缀筛选GT和hazy目录中的成对文件
    :param gt_dir: GT目录路径
    :param hazy_dir: hazy目录路径
    :param target_prefixes: 目标前缀列表（如["moderate", "thin", "thick"]）
    :return: 字典 {前缀: 成对文件前缀列表}
    """
    # 初始化结果字典
    paired_files = {prefix: [] for prefix in target_prefixes}
    
    # 获取GT目录所有文件的完整前缀（含分类前缀，如moderate_001）
    gt_all_prefixes = {}
    for filename in os.listdir(gt_dir):
        if os.path.isfile(os.path.join(gt_dir, filename)):
            # 提取文件完整前缀（去后缀）
            full_prefix = os.path.splitext(filename)[0]
            # 匹配目标分类前缀（如moderate_001 → 匹配moderate）
            for target in target_prefixes:
                if full_prefix.startswith(f"{target}_"):
                    gt_all_prefixes[full_prefix] = target
                    break
    
    # 筛选hazy目录中对应的成对文件
    for full_prefix, target in gt_all_prefixes.items():
        # 检查hazy目录是否有相同完整前缀的文件
        has_match = False
        for filename in os.listdir(hazy_dir):
            if os.path.isfile(os.path.join(hazy_dir, filename)):
                if os.path.splitext(filename)[0] == full_prefix:
                    has_match = True
                    break
        if has_match:
            paired_files[target].append(full_prefix)
    
    return paired_files

def delete_paired_files_by_prefix(gt_dir: str, hazy_dir: str, 
                                 target_prefixes: list, delete_num_per_prefix: int = 5):
    """
    按指定前缀删除成对文件，每个前缀删除指定数量
    :param gt_dir: GT目录路径
    :param hazy_dir: hazy目录路径
    :param target_prefixes: 目标前缀列表
    :param delete_num_per_prefix: 每个前缀删除的文件对数
    """
    # ========== 安全检查 ==========
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT目录不存在：{gt_dir}")
    if not os.path.isdir(hazy_dir):
        raise FileNotFoundError(f"hazy目录不存在：{hazy_dir}")
    
    # ========== 获取按前缀分类的成对文件 ==========
    paired_files = get_paired_files_by_prefix(gt_dir, hazy_dir, target_prefixes)
    
    # ========== 检查每个前缀的文件数量 ==========
    for prefix in target_prefixes:
        count = len(paired_files[prefix])
        if count < delete_num_per_prefix:
            raise ValueError(f"前缀「{prefix}」的成对文件仅{count}个，不足要删除的{delete_num_per_prefix}个！")
    
    # ========== 执行删除 ==========
    total_deleted = 0
    print(f"开始按前缀删除文件（每个前缀删{delete_num_per_prefix}对）：\n")
    
    for prefix in target_prefixes:
        # 随机选择要删除的文件前缀
        delete_list = random.sample(paired_files[prefix], delete_num_per_prefix)
        print(f"【{prefix}】随机选中的删除列表：{delete_list}")
        
        # 删除该前缀下的文件
        deleted_count = 0
        for full_prefix in delete_list:
            # 删除GT目录文件
            for filename in os.listdir(gt_dir):
                if os.path.splitext(filename)[0] == full_prefix:
                    file_path = os.path.join(gt_dir, filename)
                    os.remove(file_path)
                    print(f"已删除GT文件：{file_path}")
            
            # 删除hazy目录文件
            for filename in os.listdir(hazy_dir):
                if os.path.splitext(filename)[0] == full_prefix:
                    file_path = os.path.join(hazy_dir, filename)
                    os.remove(file_path)
                    print(f"已删除hazy文件：{file_path}")
            
            deleted_count += 1
        
        print(f"【{prefix}】已删除 {deleted_count} 对文件\n")
        total_deleted += deleted_count
    
    # ========== 输出最终结果 ==========
    print("===== 删除完成 =====")
    print(f"总计删除成对文件数：{total_deleted}")
    print(f"GT目录剩余文件数：{len([f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))])}")
    print(f"hazy目录剩余文件数：{len([f for f in os.listdir(hazy_dir) if os.path.isfile(os.path.join(hazy_dir, f))])}")

if __name__ == "__main__":
    # 配置项（按需修改）
    GT_DIR = "/home/yueyinlei/paper/datasets/Haze1k_ddm3/train/GT"
    HAZY_DIR = "/home/yueyinlei/paper/datasets/Haze1k_ddm3/train/hazy"
    TARGET_PREFIXES = ["moderate", "thin", "thick"]  # 要删除的分类前缀
    DELETE_NUM_PER_PREFIX = 15  # 每个前缀删除15对
    
    # 强制备份提示
    print("⚠️  重要提示：请先备份文件！执行以下命令备份：")
    print(f"cp -r ./train ./train_backup\n")
    
    confirm = input("是否已备份文件？输入 y 继续，其他键退出：")
    if confirm.lower() != "y":
        print("操作已取消！")
        exit(0)
    
    # 执行删除
    try:
        delete_paired_files_by_prefix(
            gt_dir=GT_DIR,
            hazy_dir=HAZY_DIR,
            target_prefixes=TARGET_PREFIXES,
            delete_num_per_prefix=DELETE_NUM_PER_PREFIX
        )
    except Exception as e:
        print(f"执行出错：{e}")