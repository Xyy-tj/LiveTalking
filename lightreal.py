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

import math
import torch
import numpy as np

#from .utils import *
import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp


from hubertasr import HubertASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

#from imgcache import ImgCache

from tqdm import tqdm

#new
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertModel
from torch.utils.data import DataLoader
from ultralight.unet import Model
from ultralight.audio2feature import Audio2Feature
from logger import logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Using {} for inference.'.format(device))


def load_model(opt):
    audio_processor = Audio2Feature()
    return audio_processor

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl" 
    
    model = Model(6, 'hubert').to(device)  # 假设Model是你自定义的类
    model.load_state_dict(torch.load(f"{avatar_path}/ultralight.pth"))
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return model.eval(),frame_list_cycle,face_list_cycle,coord_list_cycle


@torch.no_grad()
def warm_up(batch_size,avatar,modelres):
    logger.info('warmup model...')
    model,_,_,_ = avatar
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 32, 32, 32).to(device)
    model(img_batch, mel_batch)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_audio_features(features, index):
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds


def read_lms(lms_list):
    land_marks = []
    logger.info('reading lms...')
    for lms_path in tqdm(lms_list):
        file_landmarks = []  # Store landmarks for this file
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = list(filter(None, line.split(" ")))
                if arr:
                    arr = np.array(arr, dtype=np.float32)
                    file_landmarks.append(arr)
        land_marks.append(file_landmarks)  # Add the file's landmarks to the overall list
    return land_marks

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 


def inference(quit_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, model):
    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    logger.info('start inference')
    
    # 预先分配数组以减少内存分配
    indices = np.zeros(batch_size, dtype=np.int32)
    
    # 预缓存常用尺寸和区域
    face_region = (4, 164, 4, 164)  # y1, y2, x1, x2
    mask_region = (5, 5, 150, 145)  # x, y, w, h

    while not quit_event.is_set():
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
            
        # 收集音频帧
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size*2):
            frame, type_, eventpoint = audio_out_queue.get()
            audio_frames.append((frame, type_, eventpoint))
            if type_ == 0:
                is_all_silence = False
        
        # 处理静音帧
        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1
            continue
            
        # 开始计时
        t = time.perf_counter()
        
        # 获取当前批次的索引
        for i in range(batch_size):
            indices[i] = __mirror_index(length, index + i)
            
        # 批量处理图像 - 避免循环中的重复操作
        # 1. 预先分配一个批次的图像数组
        y1, y2, x1, x2 = face_region
        img_batch_np = np.zeros((batch_size, 6, y2-y1, x2-x1), dtype=np.float32)
        
        # 2. 并行处理每个图像 (可以考虑使用 NumPy 的向量化操作)
        for i in range(batch_size):
            idx = indices[i]
            crop_img = face_list_cycle[idx]
            
            # 提取人脸区域并创建只有一次副本
            img_real_ex = crop_img[y1:y2, x1:x2]
            
            # 创建遮罩图像 - 优化: 只在需要的区域应用遮罩
            img_masked = img_real_ex.copy()
            mx, my, mw, mh = mask_region
            img_masked[my:my+mh, mx:mx+mw] = 0
            
            # 直接转换并填充预分配的数组 - 减少临时对象
            # 第一个通道: 原始图像
            img_batch_np[i, 0:3] = img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0
            # 第二个通道: 遮罩图像
            img_batch_np[i, 3:6] = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # 3. 一次性移动到GPU
        img_batch = torch.from_numpy(img_batch_np).to(device, non_blocking=True)
        
        # 4. 处理mel特征
        if isinstance(mel_batch, list):
            # 优化: 使用numpy的批处理而不是列表推导
            mel_np = np.stack([arr.reshape(32, 32, 32) for arr in mel_batch])
            mel_batch_tensor = torch.from_numpy(mel_np).to(device, non_blocking=True)
        else:
            # 如果已经是numpy数组
            mel_batch_tensor = torch.from_numpy(mel_batch.reshape(-1, 32, 32, 32)).to(device, non_blocking=True)
            
        # 5. 模型推理 - 使用非阻塞传输可能提高并行性
        with torch.no_grad():
            pred = model(img_batch, mel_batch_tensor)
            
        # 6. 处理预测结果
        # 只在需要时执行CPU转换 (这是必要的瓶颈)
        pred_np = pred.cpu().numpy()
        pred_np = pred_np.transpose(0, 2, 3, 1) * 255.0
        
        # 7. 更新计时器和计数器
        counttime += (time.perf_counter() - t)
        count += batch_size
        if count >= 100:
            logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
            count = 0
            counttime = 0
            
        # 8. 处理输出帧 - 使用更高效的索引处理
        for i, res_frame in enumerate(pred_np):
            res_frame_queue.put((res_frame, indices[i], audio_frames[i*2:i*2+2]))
            index += 1

#            for i, pred_frame in enumerate(pred):
#                pred_frame_uint8 = np.array(pred_frame, dtype=np.uint8)
#                res_frame_queue.put((pred_frame_uint8, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
#                index = (index + 1) % length

        #print('total batch time:', time.perf_counter() - starttime)

    logger.info('lightreal inference processor stop')


class LightReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps # 20 ms per frame
        
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size*2)  #mp.Queue
        #self.__loadavatar()
        audio_processor = model
        self.model,self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = HubertASR(opt,self,audio_processor)
        self.asr.warm_up()
        #self.__warm_up()
        
        self.render_event = mp.Event()
    
    def __del__(self):
        logger.info(f'lightreal({self.sessionid}) delete')

   
    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        # 预分配缓冲区以避免频繁内存分配
        face_region = (4, 164, 4, 164)  # 人脸区域坐标 (y1, y2, x1, x2)
        
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            # 处理静音帧 - 不需要图像合成
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:  # 全为静音数据
                self.speaking = False
                audiotype = audio_frames[0][1]
                
                # 检查是否有自定义视频
                if self.custom_index.get(audiotype) is not None:  
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    # 直接使用预加载的帧，无需复制
                    combine_frame = self.frame_list_cycle[idx]
            else:
                # 处理语音帧 - 需要图像合成
                self.speaking = True
                t_start = time.perf_counter()  # 用于性能监控
                
                # 获取变量，避免重复访问
                bbox = self.coord_list_cycle[idx]
                x1, y1, x2, y2 = bbox
                height, width = y2-y1, x2-x1
                
                # 只在真正需要深拷贝的地方使用
                combine_frame = np.array(self.frame_list_cycle[idx], copy=True)
                crop_img = self.face_list_cycle[idx]
                
                try:
                    # 直接操作，减少中间变量
                    if res_frame is not None:
                        # 转换为uint8仅执行一次
                        if not isinstance(res_frame, np.ndarray) or res_frame.dtype != np.uint8:
                            res_frame = res_frame.astype(np.uint8)
                            
                        # 创建一个视图而不是复制
                        crop_img_processed = crop_img.copy()
                        y1_face, y2_face, x1_face, x2_face = face_region
                        crop_img_processed[y1_face:y2_face, x1_face:x2_face] = res_frame
                        
                        # 直接调整大小并应用到最终帧
                        # 使用INTER_LINEAR进行更快的调整大小
                        if (height, width) != crop_img_processed.shape[:2]:
                            crop_img_processed = cv2.resize(crop_img_processed, (width, height), interpolation=cv2.INTER_LINEAR)
                            
                        # 直接应用到最终帧
                        combine_frame[y1:y2, x1:x2] = crop_img_processed
                        
                        # 监控性能
                        process_time = time.perf_counter() - t_start
                        if process_time > 0.01:  # 10毫秒以上记录日志
                            logger.debug(f"帧处理时间: {process_time*1000:.2f}ms")
                except Exception as e:
                    logger.error(f"处理帧时出错: {e}")
                    continue

            # 优化视频帧处理：避免额外的内存复制
            # 确保combine_frame是连续的内存布局，这样from_ndarray更高效
            if not combine_frame.flags.c_contiguous:
                combine_frame = np.ascontiguousarray(combine_frame)
            
            # 创建视频帧并发送
            new_video_frame = VideoFrame.from_ndarray(combine_frame, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_video_frame,None)), loop)
            self.record_video_data(combine_frame)

            # 优化音频帧处理：预处理数据，减少循环中的操作
            for audio_data in audio_frames:
                frame, type_, eventpoint = audio_data
                
                # 快速路径：如果已经是正确格式，避免不必要的处理
                if isinstance(frame, np.ndarray):
                    # 只在必要时进行维度转换
                    if frame.ndim > 1 and frame.shape[1] > 0:
                        frame = frame[:, 0]  # 只取第一个声道
                
                if frame.size == 0: # Skip empty audio frames
                    logger.debug("Skipping empty audio frame")
                    continue

                processed_frame = (frame * 32767).astype(np.int16)
                new_audio_frame = AudioFrame(format='s16', layout='mono', samples=processed_frame.shape[0])
                new_audio_frame.planes[0].update(processed_frame.tobytes())
                new_audio_frame.sample_rate=16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_audio_frame,eventpoint)), loop)
                self.record_audio_data(processed_frame)
                #self.notify(eventpoint)
        logger.info('lightreal process_frames thread stop') 
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()
        Thread(target=inference, args=(quit_event,self.batch_size,self.face_list_cycle,self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.model,)).start()  #mp.Process
        

        #self.render_event.set() #start infer process render
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
            # Adjusted sleep logic to be less aggressive and potentially based on a target latency
            # This is a simple adjustment; more sophisticated adaptive logic might be needed.
            if video_track._queue.qsize() >= self.opt.batch_size * 1.5: # Sleep if queue is 1.5x batch_size
                sleep_duration = 0.01 * (video_track._queue.qsize() - self.opt.batch_size) # Sleep proportionally to excess queue size
                logger.debug('sleep qsize=%d, sleeping for %.3fs', video_track._queue.qsize(), sleep_duration)
                time.sleep(max(0, sleep_duration)) # Ensure sleep duration is not negative
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        #self.render_event.clear() #end infer process render
        logger.info('lightreal thread stop')
            

