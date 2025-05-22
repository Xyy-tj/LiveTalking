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
import wave

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum
import os
import dotenv

dotenv.load_dotenv()

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
        # eventpoint={'status':'end','text':text,'msgenvent':textevent}
        # self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

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
        # eventpoint={'status':'end','text':text,'msgenvent':textevent}
        # self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

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
        # eventpoint={'status':'end','text':text,'msgenvent':textevent}
        # self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

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
        # eventpoint={'status':'end','text':text,'msgenvent':textevent}
        # self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  

###########################################################################################
class DashScopeTTS(BaseTTS):
    def __init__(self, opt, parent:BaseReal):
        super().__init__(opt, parent)
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
        import dashscope
        
        # 恢复使用固定API密钥和模型配置
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        self.current_text = None
        self.current_event = None
        self.first_chunk = True
        self.internal_audio_buffer = np.array([], dtype=np.float32)
        self.has_saved_first_raw_chunk = False
        self.truncate_first_data_chunk = True

    def _create_callback(self):
        from dashscope.audio.tts_v2 import ResultCallback
        
        class TTSCallback(ResultCallback):
            def __init__(self, parent):
                self.parent = parent
                
            def on_open(self):
                logger.info("DashScope TTS connection established")
                
            def on_complete(self):
                logger.info("DashScope TTS synthesis completed")

                # 处理内部缓冲区中剩余的音频数据
                if len(self.parent.internal_audio_buffer) > 0 and self.parent.state == State.RUNNING:
                    remaining_audio = self.parent.internal_audio_buffer
                    # 如果剩余数据不足一个标准块，则用静音填充
                    if len(remaining_audio) < self.parent.chunk:
                        padding = np.zeros(self.parent.chunk - len(remaining_audio), dtype=np.float32)
                        remaining_audio = np.concatenate((remaining_audio, padding))
                    
                    # 确保只发送一个标准块大小的音频数据
                    frame_to_send = remaining_audio[:self.parent.chunk]
                    self.parent.parent.put_audio_frame(frame_to_send, None) # 不再需要特殊事件，因为下一个是结束事件
                
                # 清空缓冲区，以防万一
                self.parent.internal_audio_buffer = np.array([], dtype=np.float32)
                
                # 发送一个全静音的音频帧作为明确的结束信号
                if self.parent.current_text and self.parent.current_event and self.parent.state == State.RUNNING:
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
                
                processed_data = data # Use a new variable for data after potential truncation

                # Save and potentially truncate the very first raw data chunk received from DashScope
                if not self.parent.has_saved_first_raw_chunk and data:
                    try:
                        # Save the original first chunk
                        original_filename = "first_chunk_original.wav"
                        with wave.open(original_filename, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(data)
                        logger.info(f"Saved the original first raw DashScope audio chunk to {original_filename}")

                        if self.parent.truncate_first_data_chunk:
                            TRUNCATE_BYTES = 320 # About 10ms at 16kHz, 16-bit mono
                            if len(data) > TRUNCATE_BYTES:
                                processed_data = data[TRUNCATE_BYTES:]
                                logger.info(f"Truncated the first data chunk by {TRUNCATE_BYTES} bytes.")
                                # Save the truncated first chunk for comparison
                                truncated_filename = "first_chunk_truncated.wav"
                                with wave.open(truncated_filename, 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(16000)
                                    wf.writeframes(processed_data)
                                logger.info(f"Saved the truncated first raw DashScope audio chunk to {truncated_filename}")
                            else:
                                logger.warning(f"First data chunk is too short ({len(data)} bytes) to truncate by {TRUNCATE_BYTES} bytes. Using original.")
                            self.parent.truncate_first_data_chunk = False # Only truncate the very first chunk of a sentence
                        
                        self.parent.has_saved_first_raw_chunk = True # Mark that we've processed (saved/truncated) the first chunk
                    except Exception as e:
                        logger.error(f"Error saving/truncating first raw DashScope chunk: {e}")
                
                if not processed_data: # If truncation resulted in empty data, skip
                    return

                try:
                    # 将音频数据转换为numpy数组, 使用 processed_data
                    new_audio_chunk = np.frombuffer(processed_data, dtype=np.int16).astype(np.float32) / 32767
                    # 仅当采样率不同时才进行重采样
                    if 16000 != self.parent.sample_rate: # DashScope 返回的是 16000 Hz
                        new_audio_chunk = resampy.resample(x=new_audio_chunk, sr_orig=16000, sr_new=self.parent.sample_rate)
                    
                    # 将新接收的音频数据追加到内部缓冲区
                    self.parent.internal_audio_buffer = np.concatenate((self.parent.internal_audio_buffer, new_audio_chunk))
                    
                    # 当内部缓冲区数据量达到或超过标准块大小时，分块发送
                    while len(self.parent.internal_audio_buffer) >= self.parent.chunk and self.parent.state == State.RUNNING:
                        current_eventpoint_for_frame = None # Default to no event for this specific frame
                        if self.parent.first_chunk:
                            # 'start' event is now associated with the initial silent frame
                            start_eventpoint = {
                                'status': 'start',
                                'text': self.parent.current_text,
                                'msgenvent': self.parent.current_event
                            }
                            # Send the initial silent frame WITH the 'start' event
                            self.parent.parent.put_audio_frame(np.zeros(self.parent.chunk, np.float32), start_eventpoint)
                            self.parent.first_chunk = False
                            
                            # The actual first audio frame will be processed in the next iteration or immediately if buffer is short,
                            # and it won't have the start event again.
                            # We need to ensure this loop iteration doesn't send another frame if the silent frame was the only thing processed.
                            # However, the standard logic below will pick up the first real audio frame correctly.
                            
                        # Take one standard-sized audio frame from the buffer
                        frame_to_send = self.parent.internal_audio_buffer[:self.parent.chunk]
                        self.parent.internal_audio_buffer = self.parent.internal_audio_buffer[self.parent.chunk:]
                        
                        # current_eventpoint_for_frame is None unless explicitly set (e.g. for a future different event type)
                        self.parent.parent.put_audio_frame(frame_to_send, current_eventpoint_for_frame)
                        
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
            self.first_chunk = True  # 标记这是新文本的第一个待处理块
            self.internal_audio_buffer = np.array([], dtype=np.float32) # 初始化内部音频缓冲区
            self.has_saved_first_raw_chunk = False # Reset flag for each new synthesis, to save its first chunk
            self.truncate_first_data_chunk = True # New flag to indicate the first actual data chunk should be truncated
            
            if self.opt.voice_id_dashscope != None:
                voice_id = self.opt.voice_id_dashscope
                logger.info("当前使用用户定义音色id: %s", voice_id)
            else:
                # 从环境变量中获取音色ID
                voice_id = os.getenv("DASHSCOPE_VOICE_ID")
                logger.info("当前使用环境变量音色id: %s", voice_id)
        
            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v2",
                voice = voice_id,
                format=AudioFormat.WAV_16000HZ_MONO_16BIT,
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
                # stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
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
        # eventpoint={'status':'end','text':text,'msgenvent':textevent}
        # self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  
