import time
import pyaudio
import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts_v2 import *

from datetime import datetime

def get_timestamp():
    now = datetime.now()
    formatted_timestamp = now.strftime("[%Y-%m-%d %H:%M:%S.%f]")
    return formatted_timestamp

# 若没有将API Key配置到环境变量中，需将your-api-key替换为自己的API Key
dashscope.api_key = "sk-541b710c10f4476a8405f211ba8b89c4"

# 模型
model = "cosyvoice-v2"
# 音色
voice = "cosyvoice-v2-prefix-f7b6006a6386487eb959dffb042a34cf"

# # 创建语音注册服务实例
# service = VoiceEnrollmentService()
# service.update_voice(voice_id=voice_id, url=url)
# print("request id为：", service.get_last_request_id())
# voice = service.get_voice(voice_id=voice_id)

# 定义回调接口
class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print("连接建立：" + get_timestamp())
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=16000, output=True
        )

    def on_complete(self):
        print("语音合成完成，所有合成结果已被接收：" + get_timestamp())

    def on_error(self, message: str):
        print(f"语音合成出现异常：{message}")

    def on_close(self):
        print("连接关闭：" + get_timestamp())
        # 停止播放器
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        pass

    def on_data(self, data: bytes) -> None:
        print(get_timestamp() + " 二进制音频长度为：" + str(len(data)))
        self._stream.write(data)


callback = Callback()

test_text = [
    "流式文本语音合成SDK，",
    "可以将输入的文本",
    "合成为语音二进制数据，",
    "相比于非流式语音合成，",
    "流式合成的优势在于实时性",
    "更强。"    
]

# 实例化SpeechSynthesizer，并在构造方法中传入模型（model）、音色（voice）等请求参数
synthesizer = SpeechSynthesizer(
    model=model,
    voice=voice,
    format=AudioFormat.WAV_16000HZ_MONO_16BIT,  
    callback=callback,
)


# 流式发送待合成文本。在回调接口的on_data方法中实时获取二进制音频
for text in test_text:
    synthesizer.streaming_call(text)
    time.sleep(0.1)
# 结束流式语音合成
synthesizer.streaming_complete()

print('[Metric] requestId为：{}，首包延迟为：{}毫秒'.format(
    synthesizer.get_last_request_id(),
    synthesizer.get_first_package_delay()))