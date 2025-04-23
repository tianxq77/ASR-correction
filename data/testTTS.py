from gtts import gTTS


text = "床前明月光。"
tts = gTTS(text=text, lang='zh-cn')  # 中文语音合成
tts.save("D:\pythonProjects\myASR\data\std_床前明月光.wav")  # 保存为WAV文件
print("标准音频已生成！路径：data/std_床前明月光.wav")