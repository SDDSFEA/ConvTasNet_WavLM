import csv

def wer(ref, hyp):
    """
    计算词错误率 WER
    ref: reference string
    hyp: hypothesis string
    """
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()

    n = len(ref_words)
    if n == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # 编辑距离 DP
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,     # 删除
                    dp[i][j - 1] + 1,     # 插入
                    dp[i - 1][j - 1] + 1  # 替换
                )

    return dp[len(ref_words)][len(hyp_words)] / n


# ---------- 读取 CSV ----------
csv_data = {}
with open("train_100/train_100_spk2.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_data[row["ID"]] = row["Speaker"]

# ---------- 读取 TXT ----------
txt_data = {}
with open("train_100/train_100_text_spk2", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        id_, text = parts
        id_ = id_ +".wav"
        txt_data[id_] = text

total_wer = 0.0
count = 0
# ---------- 计算 WER 并输出 ----------
with open("train_100/wer_spk2.txt", "w", encoding="utf-8") as out:
    for id_ in csv_data:
        if id_ in txt_data:
            wer_value = wer(txt_data[id_], csv_data[id_])
            total_wer += wer_value
            count += 1
            if wer_value > 0.5:
                print(id_, wer_value, '\n',csv_data[id_].lower(), '\n' ,txt_data[id_].lower())
            out.write(f"{id_}\t{wer_value:.4f}\n{csv_data[id_].lower()}\n{txt_data[id_].lower()}\n")
    average_wer = total_wer / count if count > 0 else 0.0
    print(f"平均 WER: {average_wer:.4f}")
    out.write(f"平均 WER: {average_wer:.4f}\n")
