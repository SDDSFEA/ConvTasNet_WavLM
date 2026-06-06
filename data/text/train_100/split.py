import csv

# 读取 train_100 和 train_360 的 ID 列
def read_ids(csv_file):
    ids = set()
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row['ID'])
    return ids

train_100_ids = read_ids('/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/train_100/train_100_spk2.csv')
train_360_ids = read_ids('/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/train_360/train_360_spk2.csv')

# 打开 text_spk1 并分流
with open('/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/train_360/text_spk2', 'r', encoding='utf-8') as f:
    lines = f.readlines()

train_100_lines = []
train_360_lines = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    audio_id = line.split()[0]  # 假设空格分隔，取第一项
    audio_id = audio_id + ".wav"

    in_100 = audio_id in train_100_ids
    in_360 = audio_id in train_360_ids

    if in_100 and in_360:
        print(f"⚠️ WARNING: {audio_id} 同时在 train_100 和 train_360 中！")
    elif in_100:
        train_100_lines.append(line)
    elif in_360:
        train_360_lines.append(line)
    else:
        # 如果不在任何一个 CSV 中，可根据需要处理
        print(f"⚠️ WARNING: {audio_id} 不在任何训练集里！")

# 写入输出文件
with open('train_100_text_spk2', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_100_lines) + '\n')

with open('train_360_text_spk2', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_360_lines) + '\n')

print("✅ 分流完成！")
