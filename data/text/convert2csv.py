import csv

input_file = '/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/val/text_spk2'
output_file = '/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/val/GT_val_spk2.csv'

rows = []

# 读取文件并生成行列表
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)  # 分成 ID + 剩下文本
        audio_id = parts[0] + ".wav"
        speaker = parts[1].lower()
        rows.append([audio_id, speaker])

# 按 ID 排序
import re
# 定义按数字顺序的 key
def numeric_key(id_string):
    # 取出 ID 的部分（去掉 .wav 后缀）
    id_only = id_string.replace('.wav','')
    # 用 - 或 _ 分割，然后转为整数
    numbers = re.split('[-_]', id_only)
    return [int(n) for n in numbers]

# 按数字顺序排序
rows.sort(key=lambda x: numeric_key(x[0]))
# 写入 CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['ID', 'Speaker'])
    writer.writerows(rows)

print(f"✅ 转换完成并按 ID 排序，保存为 {output_file}")
