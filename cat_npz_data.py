import numpy as np
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="Load and display data from a .npz file")
parser.add_argument("--filename", help="The file to load data from")

# 加载.npz文件
data = np.load(parser.parse_args().filename)

# 列出文件中所有的数组名称
arrays = data.files
print("Arrays in the file:", arrays)

# 访问特定的数组
array_name = arrays[2]  # 假设你想要访问第一个数组
array_data = data[array_name]
print(len(array_data))
print(f"Data in {array_name}:\n", array_data)

# 记得关闭文件
data.close()