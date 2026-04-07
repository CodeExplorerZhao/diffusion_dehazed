import os
import random
import shutil

def ensure_dir_exists(dir_path: str):
    """确保目录存在，不存在则创建"""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录：{dir_path}")

def get_paired_files_by_prefix(test_gt_dir: str, test_hazy_dir: str, target_prefixes: list) -> dict:
    """
    从test目录按前缀筛选成对文件
    :param test_gt_dir: test/GT目录路径
    :param test_hazy_dir: test/hazy目录路径
    :param target_prefixes: 目标前缀列表（moderate/thin/thick）
    :return: 字典 {前缀: 成对文件完整前缀列表}
    """
    paired_files = {prefix: [] for prefix in target_prefixes}
    
    # 获取test/GT中所有目标前缀的文件完整前缀
    gt_full_prefixes = {}
    for filename in os.listdir(test_gt_dir):
        if os.path.isfile(os.path.join(test_gt_dir, filename)):
            full_prefix = os.path.splitext(filename)[0]
            # 匹配目标前缀（如moderate_001 → 匹配moderate）
            for target in target_prefixes:
                if full_prefix.startswith(f"{target}_"):
                    gt_full_prefixes[full_prefix] = target
                    break
    
    # 筛选test/hazy中对应的成对文件
    for full_prefix, target in gt_full_prefixes.items():
        has_match = False
        for filename in os.listdir(test_hazy_dir):
            if os.path.isfile(os.path.join(test_hazy_dir, filename)):
                if os.path.splitext(filename)[0] == full_prefix:
                    has_match = True
                    break
        if has_match:
            paired_files[target].append(full_prefix)
    
    return paired_files

def copy_paired_files(test_gt_dir: str, test_hazy_dir: str,
                     train_gt_dir: str, train_hazy_dir: str,
                     target_prefixes: list, copy_num_per_prefix: int = 5):
    """
    从test目录按前缀拷贝指定数量的成对文件到train目录（不删除test文件）
    :param test_gt_dir: test/GT路径
    :param test_hazy_dir: test/hazy路径
    :param train_gt_dir: train/GT路径
    :param train_hazy_dir: train/hazy路径
    :param target_prefixes: 目标前缀列表
    :param copy_num_per_prefix: 每个前缀拷贝的文件对数
    """
    # ========== 安全检查 ==========
    if not os.path.isdir(test_gt_dir) or not os.path.isdir(test_hazy_dir):
        raise FileNotFoundError("test/GT 或 test/hazy 目录不存在！")
    
    # 确保train目录存在
    ensure_dir_exists(train_gt_dir)
    ensure_dir_exists(train_hazy_dir)
    
    # ========== 获取test目录中成对文件 ==========
    paired_files = get_paired_files_by_prefix(test_gt_dir, test_hazy_dir, target_prefixes)
    
    # ========== 检查每个前缀的文件数量 ==========
    for prefix in target_prefixes:
        count = len(paired_files[prefix])
        if count < copy_num_per_prefix:
            raise ValueError(f"test目录中前缀「{prefix}」的成对文件仅{count}个，不足要拷贝的{copy_num_per_prefix}个！")
    
    # ========== 执行拷贝 ==========
    total_copied = 0
    print(f"开始从test拷贝文件到train（每个前缀拷贝{copy_num_per_prefix}对）：\n")
    
    for prefix in target_prefixes:
        # 随机选择要拷贝的文件前缀
        copy_list = random.sample(paired_files[prefix], copy_num_per_prefix)
        print(f"【{prefix}】随机选中的拷贝列表：{copy_list}")
        
        copied_count = 0
        for full_prefix in copy_list:
            # 拷贝test/GT → train/GT
            for filename in os.listdir(test_gt_dir):
                if os.path.splitext(filename)[0] == full_prefix:
                    src_path = os.path.join(test_gt_dir, filename)
                    dst_path = os.path.join(train_gt_dir, filename)
                    # 避免文件已存在导致覆盖
                    if os.path.exists(dst_path):
                        print(f"警告：train/GT已存在{filename}，跳过拷贝")
                        continue
                    shutil.copy(src_path, dst_path)  # 核心修改：copy替代move
                    print(f"已拷贝：{src_path} → {dst_path}")
            
            # 拷贝test/hazy → train/hazy
            for filename in os.listdir(test_hazy_dir):
                if os.path.splitext(filename)[0] == full_prefix:
                    src_path = os.path.join(test_hazy_dir, filename)
                    dst_path = os.path.join(train_hazy_dir, filename)
                    if os.path.exists(dst_path):
                        print(f"警告：train/hazy已存在{filename}，跳过拷贝")
                        continue
                    shutil.copy(src_path, dst_path)  # 核心修改：copy替代move
                    print(f"已拷贝：{src_path} → {dst_path}")
            
            copied_count += 1
        
        print(f"【{prefix}】已拷贝 {copied_count} 对文件\n")
        total_copied += copied_count
    
    # ========== 输出最终结果 ==========
    print("===== 拷贝完成 =====")
    print(f"总计拷贝成对文件数：{total_copied}")
    # 统计各目录文件数量
    def count_files(dir_path):
        return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    
    print(f"test/GT文件数（不变）：{count_files(test_gt_dir)}")
    print(f"test/hazy文件数（不变）：{count_files(test_hazy_dir)}")
    print(f"train/GT当前文件数：{count_files(train_gt_dir)}")
    print(f"train/hazy当前文件数：{count_files(train_hazy_dir)}")

if __name__ == "__main__":
    # 配置项（按需修改）
    TEST_GT_DIR = "/home/yueyinlei/paper/datasets/Haze1k_ddm3/test/GT"
    TEST_HAZY_DIR = "/home/yueyinlei/paper/datasets/Haze1k_ddm3/test/hazy"
    TRAIN_GT_DIR = "/home/yueyinlei/paper/datasets/Haze1k_ddm3/train/GT"
    TRAIN_HAZY_DIR = "/home/yueyinlei/paper/datasets/Haze1k_ddm3/train/hazy"

    TARGET_PREFIXES = ["moderate", "thin", "thick"]  # 目标前缀
    COPY_NUM_PER_PREFIX = 15  # 每个前缀拷贝15对
    
    # 备份提示
    print("⚠️  重要提示：请先备份train目录（避免覆盖）！执行以下命令备份：")
    print(f"cp -r ./train ./train_backup\n")
    
    confirm = input("是否已备份文件？输入 y 继续，其他键退出：")
    if confirm.lower() != "y":
        print("操作已取消！")
        exit(0)
    
    # 执行拷贝
    try:
        copy_paired_files(
            test_gt_dir=TEST_GT_DIR,
            test_hazy_dir=TEST_HAZY_DIR,
            train_gt_dir=TRAIN_GT_DIR,
            train_hazy_dir=TRAIN_HAZY_DIR,
            target_prefixes=TARGET_PREFIXES,
            copy_num_per_prefix=COPY_NUM_PER_PREFIX
        )
    except Exception as e:
        print(f"执行出错：{e}")