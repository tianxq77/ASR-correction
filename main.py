import numpy as np
import librosa
import librosa.display
import pyworld as pw
from dtw import dtw
import whisper  # 需要安装openai-whisper
import os
from pydub import AudioSegment

# 设置绝对路径（注意斜杠方向）
ffmpeg_path = r"C:\Users\tianx\.conda\envs\asr\Library\bin\ffmpeg.exe"
ffprobe_path = r"C:\Users\tianx\.conda\envs\asr\Library\bin\ffprobe.exe"

# 强制添加到系统路径
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path


# 初始化Whisper ASR模型（小型模型，可替换为medium/large）
asr_model = whisper.load_model("tiny")
path = r"D:\pythonProjects\myASR\data\std_床前明月光.wav"
# 1. 加载标准音频和文本（示例：《静夜思》第一句）

std_text = "床前明月光"
std_audio, std_sr = librosa.load(path, sr=16000, dtype=np.float64)

# 2. 用户录音处理
def load_user_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    # 保持float32以兼容Whisper
    user_audio = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return user_audio, 16000

user_audio, user_sr = load_user_audio(path)

# 3. ASR识别（可能包含错误）
def transcribe_audio(audio):
    result = asr_model.transcribe(audio)
    return result["text"]


asr_text = transcribe_audio(user_audio)
print(f"ASR识别结果: {asr_text}")


# 4. 提取MFCC和Pitch特征
# def extract_features(audio, sr):
#     audio = audio.astype(np.float64)
#     # MFCC（13维+一阶差分）
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=160, n_fft=512)
#     delta_mfcc = librosa.feature.delta(mfcc)
#     mfcc_features = np.vstack([mfcc, delta_mfcc])
#
#     # Pitch（基频）
#     f0, _ = pw.harvest(audio, sr)
#     return mfcc_features.T, f0  # 转置为（帧数, 特征维数）
#
def extract_features(audio, sr):
    # 临时转换为float64供pyworld使用
    audio_f64 = audio.astype(np.float64)

    # MFCC特征（保持float32）
    mfcc = librosa.feature.mfcc(
        y=audio_f64,  # 注意这里用float64版本
        sr=sr,
        n_mfcc=13,
        hop_length=160,
        n_fft=512
    ).astype(np.float32)  # 最终转回float32

    # Pitch特征
    f0, _ = pw.harvest(audio_f64, sr)
    return mfcc.T, f0

std_mfcc, std_f0 = extract_features(std_audio, std_sr)
user_mfcc, user_f0 = extract_features(user_audio, user_sr)


# 5. DTW对齐与相似度计算
def compare_with_dtw(features1, features2):
    # 计算MFCC距离矩阵（欧氏距离）
    dist_matrix = np.zeros((len(features1), len(features2)))
    for i in range(len(features1)):
        for j in range(len(features2)):
            dist_matrix[i, j] = np.linalg.norm(features1[i] - features2[j])

    # DTW对齐
    alignment = dtw(dist_matrix)
    path = alignment.index1, alignment.index2

    # 计算平均帧距离（相似度=1/(1+平均距离)）
    avg_distance = np.mean([dist_matrix[i, j] for i, j in zip(*path)])
    return 1 / (1 + avg_distance), path


mfcc_sim, mfcc_path = compare_with_dtw(std_mfcc, user_mfcc)


# 6. 声调（Pitch）对比
def compare_pitch(f0_std, f0_user, alignment_path):
    # 对齐后的基频序列
    aligned_std_f0 = f0_std[alignment_path[0]]
    aligned_user_f0 = f0_user[alignment_path[1]]

    # 剔除静音帧（基频=0）
    voiced_frames = (aligned_std_f0 > 0) & (aligned_user_f0 > 0)
    std_f0_voiced = aligned_std_f0[voiced_frames]
    user_f0_voiced = aligned_user_f0[voiced_frames]

    # 计算相关系数（0-1）
    if len(std_f0_voiced) > 1:
        corr = np.corrcoef(std_f0_voiced, user_f0_voiced)[0, 1]
        return max(0, corr)  # 确保非负
    return 0


pitch_sim = compare_pitch(std_f0, user_f0, mfcc_path)


# 7. 综合决策：是否覆盖ASR错误
def correct_asr_error(asr_text, std_text, mfcc_sim, pitch_sim, threshold=0.8):
    if asr_text != std_text:
        if mfcc_sim > threshold or pitch_sim > 0.7:
            print(f"ASR误识已纠正：{asr_text} → {std_text}")
            return std_text
    return asr_text


final_text = correct_asr_error(asr_text, std_text, mfcc_sim, pitch_sim)

# 8. 输出报告
report = {
    "ASR原始文本": asr_text,
    "标准文本": std_text,
    "最终判定": final_text,
    "声学证据": {
        "MFCC相似度": f"{mfcc_sim:.2%}",
        "Pitch相似度": f"{pitch_sim:.2%}",
        "结论": "发音正确，ASR误识已修正" if final_text == std_text else "需人工复核"
    }
}

print("\n===== 发音评测报告 =====")
for key, value in report.items():
    print(f"{key}: {value}")