import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    """从文件中读取数据，每组数据之间有空行"""
    data = []
    with open(file_path, 'r') as file:
        group = []
        for line in file:
            line = line.strip()
            if line:  # 忽略空行
                group.append(list(map(float, line.split())))
            elif group:  # 当遇到空行且group不为空时，添加到data中
                data.append(np.array(group))
                group = []
        if group:  # 添加最后一组数据
            data.append(np.array(group))
    return data

def plot_paths(data, num_points=1200, candidate_data=None):
    """绘制路径"""
    num_paths = len(data)
    fig, axs = plt.subplots(num_paths, 1, figsize=(10, 5 * num_paths))
    
    if num_paths == 1:
        axs = [axs]  # 确保axs总是一个列表，方便后续迭代
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表，可以按需扩展
    for i, (ax, path) in enumerate(zip(axs, data)):
        path_data = path[:num_points]  # 假设我们只绘制前1200个点
        ax.plot(path_data[:, 0], path_data[:, 1], 'o', color=colors[i % len(colors)], markersize=1, label='Path')
        
        if candidate_data:
            for candidate_path in candidate_data:
                ax.plot(candidate_path[:, 0], candidate_path[:, 1], '-', color='k', alpha=0.5, label='Candidate Path')
        
        ax.set_title(f'Path {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    file_path = 'path.txt'
    candidate_path_file = 'candidate_path.txt'
    paths = read_data(file_path)
    candidate_paths = read_data(candidate_path_file)
    plot_paths(paths, num_points=1200, candidate_data=candidate_paths)

if __name__ == '__main__':
    main()