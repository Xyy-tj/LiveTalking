###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger
class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, opt, parent:BaseReal):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg:str,eventpoint=None): 
        if len(msg)>0:
            self.msgqueue.put((msg,eventpoint))

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def __init__(self, opt, parent:BaseReal):
        super().__init__(opt, parent)
        self.voice_name = opt.voice_name if hasattr(opt, 'voice_name') else "zh-CN-XiaoxiaoNeural"

    def txt_to_audio(self,msg):
        text,textevent = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(self.voice_name,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint=None
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text,'msgenvent':textevent}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text,'msgenvent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')

###########################################################################################
class FishTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        text,textevent = msg
        self.stream_tts(
            self.fish_speech(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def fish_speech(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'reference_id':reffile,
            'format':'wav',
            'streaming':True,
            'use_memory_cache':'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={
                    "content-type": "application/json",
                },
            )
            end = time.perf_counter()
            logger.info(f"fish_speech Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=17640): # 1764 44100*20ms*2
                #print('chunk len:',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"fish_speech Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('fishtts')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class VoitsTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        text,textevent = msg
        self.stream_tts(
            self.gpt_sovits(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'ogg',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=8820): # 882 22.05K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg):
        text,textevent = msg  
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            ),
            msg
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  

###########################################################################################
class DashScopeTTS(BaseTTS):
    def __init__(self, opt, parent:BaseReal):
        super().__init__(opt, parent)
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
        import dashscope
        
        # 恢复使用固定API密钥和模型配置
        dashscope.api_key = "sk-541b710c10f4476a8405f211ba8b89c4"
        
        self.current_text = None
        self.current_event = None
        self.first_chunk = True

    def _create_callback(self):
        from dashscope.audio.tts_v2 import ResultCallback
        
        class TTSCallback(ResultCallback):
            def __init__(self, parent):
                self.parent = parent
                
            def on_open(self):
                logger.info("DashScope TTS connection established")
                
            def on_complete(self):
                logger.info("DashScope TTS synthesis completed")
                # 发送结束事件
                if self.parent.current_text and self.parent.current_event:
                    eventpoint = {
                        'status': 'end',
                        'text': self.parent.current_text,
                        'msgenvent': self.parent.current_event
                    }
                    self.parent.parent.put_audio_frame(
                        np.zeros(self.parent.chunk, np.float32),
                        eventpoint
                    )
                
            def on_error(self, message: str):
                logger.error(f"DashScope TTS error: {message}")
                
            def on_close(self):
                logger.info("DashScope TTS connection closed")
                
            def on_data(self, data: bytes) -> None:
                if not data or len(data) == 0:
                    return
                    
                try:
                    # 将音频数据转换为numpy数组
                    stream = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767
                    # 重采样到目标采样率
                    stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.parent.sample_rate)
                    
                    streamlen = stream.shape[0]
                    idx = 0
                    
                    while streamlen >= self.parent.chunk and self.parent.state == State.RUNNING:
                        eventpoint = None
                        if self.parent.first_chunk:
                            eventpoint = {
                                'status': 'start',
                                'text': self.parent.current_text,
                                'msgenvent': self.parent.current_event
                            }
                            self.parent.first_chunk = False
                            
                        self.parent.parent.put_audio_frame(stream[idx:idx+self.parent.chunk], eventpoint)
                        streamlen -= self.parent.chunk
                        idx += self.parent.chunk
                        
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    
        return TTSCallback(self)

    def txt_to_audio(self, msg):
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
        import dashscope
        text, textevent = msg
        try:
            # 对于每次新的语音合成，需要确保状态是干净的
            self.current_text = text
            self.current_event = textevent
            self.first_chunk = True
            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v2",
                voice = "cosyvoice-v2-prefix-f7b6006a6386487eb959dffb042a34cf",
                format=AudioFormat.PCM_22050HZ_MONO_16BIT,
                callback=self._create_callback()
            )
            # 流式发送文本
            synthesizer.streaming_call(text)
            # 完成当前文本的合成
            synthesizer.streaming_complete()
            
        except Exception as e:
            logger.error(f"Error in txt_to_audio: {str(e)}")
            return

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  

###########################################################################################
class DashScopeStreamTTS(BaseTTS):
    def __init__(self, opt, parent:BaseReal):
        super().__init__(opt, parent)
        import dashscope
        # from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback # Moved to methods
        
        # 设置API Key
        dashscope.api_key = opt.api_key if hasattr(opt, 'api_key') else "sk-541b710c10f4476a8405f211ba8b89c4"
        
        self.opt = opt
        self.synthesizer = None # Will be initialized in txt_to_audio
        self.current_text = None
        self.current_event = None
        self.first_chunk = True
        # self.is_processing = False # No longer strictly needed with per-call synthesizer

    def _init_synthesizer(self):
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        # 每个 synthesizer 实例获取一个新的回调实例
        self.synthesizer = SpeechSynthesizer(
            model=self.opt.model if hasattr(self.opt, 'model') else "cosyvoice-v2",
            voice=self.opt.voice_name if hasattr(self.opt, 'voice_name') else "cosyvoice-v2-prefix-f7b6006a6386487eb959dffb042a34cf",
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=self._create_callback() 
        )
        logger.info("DashScope TTS synthesizer initialized/re-initialized.")

    def _create_callback(self):
        from dashscope.audio.tts_v2 import ResultCallback
        
        # TTSCallback 现在与特定的 DashScopeStreamTTS 实例关联
        class TTSCallback(ResultCallback):
            def __init__(self, parent_tts_instance: DashScopeStreamTTS):
                self.parent_tts = parent_tts_instance
                
            def on_open(self):
                logger.info("DashScope TTS connection established")
                
            def on_complete(self):
                # 使用 self.parent_tts 访问外部类成员
                logger.info("DashScope TTS synthesis completed for text: %s", self.parent_tts.current_text)
                if self.parent_tts.current_text is not None: # 确保 current_text 不是 None
                    eventpoint = {
                        'status': 'end',
                        'text': self.parent_tts.current_text,
                        'msgenvent': self.parent_tts.current_event
                    }
                    self.parent_tts.parent.put_audio_frame(
                        np.zeros(self.parent_tts.chunk, np.float32),
                        eventpoint
                    )
                # self.parent_tts.is_processing = False # No longer strictly needed
                # REMOVED: self.parent_tts._init_synthesizer()
                
            def on_error(self, message: str):
                logger.error(f"DashScope TTS error: {message} for text: {self.parent_tts.current_text}")
                # self.parent_tts.is_processing = False # No longer strictly needed
                # REMOVED: self.parent_tts._init_synthesizer()
                
            def on_close(self):
                logger.info("DashScope TTS connection closed for text: %s", self.parent_tts.current_text)
                # self.parent_tts.is_processing = False # No longer strictly needed
                # REMOVED: self.parent_tts._init_synthesizer()
                
            def on_data(self, data: bytes) -> None:
                if not data or len(data) == 0:
                    return
                    
                try:
                    stream = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767
                    stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.parent_tts.sample_rate)
                    
                    streamlen = stream.shape[0]
                    idx = 0
                    
                    while streamlen >= self.parent_tts.chunk and self.parent_tts.state == State.RUNNING:
                        eventpoint = None
                        if self.parent_tts.first_chunk:
                            eventpoint = {
                                'status': 'start',
                                'text': self.parent_tts.current_text,
                                'msgenvent': self.parent_tts.current_event
                            }
                            self.parent_tts.first_chunk = False # This is correctly scoped per call
                            
                        self.parent_tts.parent.put_audio_frame(stream[idx:idx+self.parent_tts.chunk], eventpoint)
                        streamlen -= self.parent_tts.chunk
                        idx += self.parent_tts.chunk
                        
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    
        return TTSCallback(self) # 传递当前的 DashScopeStreamTTS 实例

    def txt_to_audio(self, msg):
        if not msg or not isinstance(msg, tuple) or len(msg) != 2:
            logger.error("Invalid message format for DashScopeStreamTTS")
            return
            
        text, textevent = msg
        if not isinstance(text, str): # 确保 text 是字符串
            logger.error(f"Invalid text input for DashScopeStreamTTS: type {type(text)}, value {text}")
            return
        if not text: # 确保 text 非空
             logger.warning("Empty text input for DashScopeStreamTTS")
             # 可以选择直接返回，或者让SDK处理空字符串（如果它支持）
             # 为避免潜在问题，这里直接返回
             return
            
        try:
            # 每次调用都重新初始化合成器，确保状态干净
            self._init_synthesizer()
                
            # 这些属性现在与本次 specific_call 及其新的 synthesizer/callback 关联
            self.current_text = text
            self.current_event = textevent
            self.first_chunk = True 
            
            self.synthesizer.streaming_call(text)
            self.synthesizer.streaming_complete()
            
        except Exception as e:
            logger.error(f"Error in DashScopeStreamTTS txt_to_audio for text '{text}': {str(e)}")
            # 不需要在这里重新初始化，下一次 txt_to_audio 调用会处理
            # 也不需要管理 is_processing 标志
            return

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  