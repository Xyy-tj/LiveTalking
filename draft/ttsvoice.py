import os
import dashscope
from dashscope.audio.tts_v2 import VoiceEnrollmentService, SpeechSynthesizer
import time

dashscope.api_key = "sk-541b710c10f4476a8405f211ba8b89c4"  # 如果您没有配置环境变量，请在此处用您的API-KEY进行替换
url = "https://catgpt0-1259034079.cos.ap-shanghai.myqcloud.com/digtalhuman/0517.WAV"  # 请按实际情况进行替换
prefix = 'prefix' # 请按实际情况进行替换
target_model = "cosyvoice-v2"

# # 创建语音注册服务实例
# service = VoiceEnrollmentService()

# # 调用create_voice方法复刻声音，并生成voice_id
# # 避免频繁调用 create_voice 方法。每次调用都会创建新音色，每个阿里云主账号最多可复刻 1000 个音色，超额时请删除不用的音色或申请扩容。
# voice_id = service.create_voice(target_model=target_model, prefix=prefix, url=url)
# print("request id为：", service.get_last_request_id())
# print(f"voice id为：{voice_id}")

voice_id = "cosyvoice-v2-prefix-f7b6006a6386487eb959dffb042a34cf"


# 使用复刻的声音进行语音合成
start_time = time.time()
synthesizer = SpeechSynthesizer(model=target_model, voice=voice_id)
audio = synthesizer.call("脚感软弹，穿一整天也不累脚，配短裤穿巨显腿长。你看这个鞋面全是透气孔，不管开车上班还是海边玩水穿着都非常凉快。")
print("requestId: ", synthesizer.get_last_request_id())
end_time = time.time()
print(f"合成时间: {end_time - start_time} 秒")

# 将合成的音频文件保存到本地文件
with open("output.mp3", "wb") as f:
    f.write(audio)
