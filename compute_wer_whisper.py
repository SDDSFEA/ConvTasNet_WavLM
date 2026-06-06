#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from compute_wer import Calculator, normalize, characterize

# ====== 配置 ======
ref_csv_1 = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/data/text/test/GT_test_spk1.csv"
ref_csv_2 = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/data/text/test/GT_test_spk2.csv"
hyp_csv_1 = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/result_C64/asr_results_spk1.csv"
hyp_csv_2 = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/result_C64/asr_results_spk2.csv"

output_file = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/result_C64/utt_wer_result_PIT.txt"

text_column = "Speaker"
id_column = "ID"

case_sensitive = False
tochar = False

# ====== 读取数据 ======
ref1 = pd.read_csv(ref_csv_1)
ref2 = pd.read_csv(ref_csv_2)
hyp1 = pd.read_csv(hyp_csv_1)
hyp2 = pd.read_csv(hyp_csv_2)

ref1_dict = dict(zip(ref1[id_column], ref1[text_column]))
ref2_dict = dict(zip(ref2[id_column], ref2[text_column]))
hyp1_dict = dict(zip(hyp1[id_column], hyp1[text_column]))
hyp2_dict = dict(zip(hyp2[id_column], hyp2[text_column]))

calculator = Calculator()

# ====== 全局统计 ======
total_all = 0
total_cor = 0
total_sub = 0
total_del = 0
total_ins = 0

def prepare_tokens(text):
    text = str(text).strip()
    if tochar:
        tokens = characterize(text)
    else:
        tokens = text.split()
    return normalize(tokens, set(), case_sensitive)

with open(output_file, "w", encoding="utf-8") as fout:

    for utt_id in sorted(ref1_dict.keys()):

        if (utt_id not in ref2_dict or
            utt_id not in hyp1_dict or
            utt_id not in hyp2_dict):
            continue

        # ====== 准备文本 ======
        r1 = prepare_tokens(ref1_dict[utt_id])
        r2 = prepare_tokens(ref2_dict[utt_id])
        h1 = prepare_tokens(hyp1_dict[utt_id])
        h2 = prepare_tokens(hyp2_dict[utt_id])

        # ====== 情况 A ======
        res_a_1 = calculator.calculate(r1, h1)
        res_a_2 = calculator.calculate(r2, h2)

        err_a = (res_a_1['sub'] + res_a_1['del'] + res_a_1['ins'] +
                 res_a_2['sub'] + res_a_2['del'] + res_a_2['ins'])

        # ====== 情况 B ======
        res_b_1 = calculator.calculate(r1, h2)
        res_b_2 = calculator.calculate(r2, h1)

        err_b = (res_b_1['sub'] + res_b_1['del'] + res_b_1['ins'] +
                 res_b_2['sub'] + res_b_2['del'] + res_b_2['ins'])

        # ====== 选最优排列 ======
        if err_a <= err_b:
            chosen = [res_a_1, res_a_2]
            perm = "A (1→1, 2→2)"
        else:
            chosen = [res_b_1, res_b_2]
            perm = "B (1→2, 2→1)"

        # ====== 累计全局 ======
        for result in chosen:
            total_all += result['all']
            total_cor += result['cor']
            total_sub += result['sub']
            total_del += result['del']
            total_ins += result['ins']

        # ====== 输出本 utt 的 WER ======
        utt_err = sum(r['sub'] + r['del'] + r['ins'] for r in chosen)
        utt_all = sum(r['all'] for r in chosen)

        if utt_all != 0:
            utt_wer = utt_err * 100.0 / utt_all
        else:
            utt_wer = 0.0

        # ====== 写入详细结果 ======
        fout.write(f"utt: {utt_id}\n")
        fout.write(f"Chosen Permutation: {perm}\n")
        fout.write(f"Mixture WER: {utt_wer:.2f} %\n\n")

        for idx, r in enumerate(chosen, 1):
            fout.write(f"  Speaker {idx}\n")
            fout.write("  N=%d C=%d S=%d D=%d I=%d\n" %
                       (r['all'], r['cor'], r['sub'], r['del'], r['ins']))

            lab_lower = [w.lower() for w in r['lab']]
            rec_lower = [w.lower() for w in r['rec']]

            fout.write("  REF: " + " ".join(lab_lower) + "\n")
            fout.write("  HYP: " + " ".join(rec_lower) + "\n\n")

        fout.write("--------------------------------------------------\n\n")

    # ====== 全局 WER ======
    if total_all != 0:
        total_wer = (total_sub + total_del + total_ins) * 100.0 / total_all
    else:
        total_wer = 0.0

    fout.write("=====================================\n")
    fout.write("Global Result\n")
    fout.write("WER: %.2f %% N=%d C=%d S=%d D=%d I=%d\n" %
               (total_wer, total_all, total_cor,
                total_sub, total_del, total_ins))

print("Done.")
