data = r'C:\Users\yugeyuge\Desktop\data\images'
import os


def rename_images(folder_path, prefix="ship", start_index=1, end_index=100):
    """
    重命名文件夹中的图像文件。

    参数:
        folder_path (str): 图像文件夹路径。
        prefix (str): 文件名前缀，默认为 "ship"。
        start_index (int): 起始编号，默认为 1。
        end_index (int): 结束编号，默认为 100。
    """
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 检查文件数量是否足够
    if len(files) < (end_index - start_index + 1):
        print(f"错误：文件夹中文件数量不足，需要至少 {end_index - start_index + 1} 个文件。")
        return

    # 遍历文件并重命名
    for i, filename in enumerate(files):
        if i + start_index > end_index:
            break

        # 获取文件扩展名
        file_ext = os.path.splitext(filename)[1]

        # 生成新文件名
        new_name = f"{prefix}{i + start_index:03d}{file_ext}"

        # 重命名文件
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

        print(f"重命名：{filename} -> {new_name}")

    print("重命名完成！")


# 示例用法
folder_path = r'C:\Users\yugeyuge\Desktop\data\images'  # 替换为你的图像文件夹路径
rename_images(folder_path, prefix="ship", start_index=1, end_index=100)