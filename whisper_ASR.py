import os
import whisper
import pandas as pd

# ====== 配置 ======
audio_folder = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/result_C64_dev/spk1"   # 音频文件夹路径
output_csv = "/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/result_C64_dev/asr_results_spk1.csv"            # 输出csv文件名
model_size = "base"  # tiny, base, small, medium, large

# ====== 加载模型 ======
print("Loading Whisper model...")
model = whisper.load_model(model_size)

results = []

# ====== 遍历文件夹 ======
for filename in os.listdir(audio_folder):
    if filename.endswith((".wav", ".mp3", ".flac", ".m4a")):
        file_path = os.path.join(audio_folder, filename)

        print(f"Transcribing {filename} ...")

        try:
            result = model.transcribe(file_path)
            text = result["text"].strip()

            results.append({
                "ID": filename,
                "Speaker": text
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# ====== 保存CSV ======
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Done! Results saved to {output_csv}")
