import os

# 让AI了解文件夹结构

def print_structure(path, level=0, file=None):
    for name in os.listdir(path):
        fullname = os.path.join(path, name)
        line = '    ' * level + '|-- ' + name
        print(line)
        if file:
            file.write(line + '\n')
        if os.path.isdir(fullname):
            print_structure(fullname, level+1, file)

# 输出到 result.txt 文件
output_path = "C:/Users/tjh/Desktop/t/dir_structure.txt"
with open(output_path, "w", encoding="utf-8") as f:
    print_structure("C:/Users/tjh/Desktop/t", file=f)

print(f"目录结构已保存到: {output_path}")
