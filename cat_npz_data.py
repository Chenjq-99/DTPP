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

for array in arrays:
    print(f"Array: {array}")
    print(data[array].shape)
    print(data[array])
    print("\n")
data.close()