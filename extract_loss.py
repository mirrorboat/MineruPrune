import re

def extract_ce_loss_from_log(file_path):
    # 初始化存储 ce_loss 值的列表
    ce_loss_values = []

    # 正则表达式匹配 EVAL 行中的 ce_loss 值
    pattern = r"EVAL.*ce_loss:\s*([\d\.]+)"

    # 打开并读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    ce_loss_values.append(float(match.group(1)))  # 提取并转换为浮点数
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

    return ce_loss_values

# 指定 log 文件路径
log_file_path = "/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/0424_prune2.log"  # 替换为你的 log 文件路径

# 调用函数提取 ce_loss 值
ce_loss_values = extract_ce_loss_from_log(log_file_path)

# # 打印结果
# print("提取的 ce_loss 值：", ce_loss_values)
print(len(ce_loss_values))
# 计算最后500个 ce_loss 值的平均值
if len(ce_loss_values) >= 500:
    average_ce_loss = sum(ce_loss_values[-500:]) / 500
    print("最后500个 ce_loss 值的平均值：", average_ce_loss)