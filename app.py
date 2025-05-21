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

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger
import cv2
import threading
import time
import os


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None

# 配置腾讯云COS
app.config.update({
    'COS_SECRET_ID': os.getenv('COS_SECRET_ID'),
    'COS_SECRET_KEY': os.getenv('COS_SECRET_KEY'),
    'COS_REGION': os.getenv('COS_REGION', 'ap-guangzhou'),
    'COS_BUCKET': os.getenv('COS_BUCKET')
})
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    logger.info(f'opt.model: {opt.model}')
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if len(nerfreals) >= opt.max_session:
        logger.info('reach max session')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": "reach max session"}
            ),
        )
    sessionid = randN(6) #len(nerfreals)
    logger.info('sessionid=%d',sessionid)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    if params.get('interrupt'):
        nerfreals[sessionid].flush_talk()

    if params['type']=='echo':
        # 使用异步批处理
        await nerfreals[sessionid].put_data(params['text'])
        result = await nerfreals[sessionid].get_result()
        nerfreals[sessionid].put_msg_txt(result)
    elif params['type']=='chat':
        res = await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])
        await nerfreals[sessionid].put_data(res)
        result = await nerfreals[sessionid].get_result()
        nerfreals[sessionid].put_msg_txt(result)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg":"err","data": ""+e.args[0]+""}
            ),
        )

async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)    
    nerfreals[sessionid].set_curr_state(params['audiotype'],params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def record(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    if params['type']=='start_record':
        # nerfreals[sessionid].put_msg_txt(params['text'])
        nerfreals[sessionid].start_recording()
    elif params['type']=='end_record':
        nerfreals[sessionid].stop_recording()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def is_speaking(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
   
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    
    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    #parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto') #
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--customvideo', action='store_true', help="custom video")
    # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
    # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    parser.add_argument('--tts', type=str, default='dashscope') #xtts gpt-sovits cosyvoice
    parser.add_argument('--voice_id_dashscope', type=str, default=None, help='DashScope voice name')
    parser.add_argument('--voice_name', type=str, default='zh-CN-XiaoxiaoNeural', help='EdgeTTS voice name')
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='wav2lip') #musetalk wav2lip

    parser.add_argument('--transport', type=str, default='rtcpush') #rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str, default='http://124.223.167.54:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=6006)

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    if opt.model == 'ernerf':       
        from nerfreal import NeRFReal,load_model,load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt) 
        
        # we still need test_loader to provide audio features for testing.
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = NeRFReal(opt, trainer, test_loader,audio_processor,audio_model)
        #     nerfreals.append(nerfreal)
    elif opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = MuseReal(opt,audio_processor,vae, unet, pe,timesteps)
        #     nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        # 确保模型在 GPU 上运行
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model("./models/wav2lip.pth").to(device)
        avatar_tuple = load_avatar(opt.avatar_id)
        # 处理元组中的每个张量
        avatar = tuple(tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor for tensor in avatar_tuple)
        
        # 预加载常用图片
        image_paths = [
            'data/avatars/wav2lip256_avatar1/full_imgs/00000000.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000001.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000002.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000003.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000004.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000005.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000006.png',
            'data/avatars/wav2lip256_avatar1/full_imgs/00000007.png',
        ]
        # 设置 sessionid
        opt.sessionid = 0
        nerfreal = LipReal(opt, model, avatar)
        nerfreal.preload_images(image_paths)
        
        warm_up(opt.batch_size,model,256)
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = LipReal(opt,model)
        #     nerfreals.append(nerfreal)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    if opt.transport=='rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # 配置客户端最大上传大小为50MB
    appasync = web.Application(client_max_size=50 * 1024 * 1024)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    
    # 添加语音上传页面路由
    async def voice_upload_page(request):
        return web.FileResponse('web/voice_upload.html')
    appasync.router.add_get('/voice_upload', voice_upload_page)

    async def upload_voice_handler(request):
        try:
            # 获取上传的文件
            data = await request.post()
            if 'voice_file' not in data:
                return web.json_response({'status': 'error', 'message': 'No file uploaded'}, status=400)
            
            file = data['voice_file']
            if not file.filename:
                return web.json_response({'status': 'error', 'message': 'No selected file'}, status=400)
            
            # 检查文件大小（限制为50MB）
            file_size = len(file.file.read())
            file.file.seek(0)  # 重置文件指针
            if file_size > 50 * 1024 * 1024:  # 50MB in bytes
                return web.json_response({
                    'status': 'error',
                    'message': 'File size exceeds 50MB limit'
                }, status=400)

            # 配置腾讯云COS
            from qcloud_cos import CosConfig
            from qcloud_cos import CosS3Client
            import uuid

            secret_id = app.config.get('COS_SECRET_ID')
            secret_key = app.config.get('COS_SECRET_KEY')
            region = app.config.get('COS_REGION')
            bucket = app.config.get('COS_BUCKET')

            if not all([secret_id, secret_key, region, bucket]):
                return web.json_response({'status': 'error', 'message': 'COS configuration missing'}, status=500)

            config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
            client = CosS3Client(config)

            # 生成唯一文件名
            file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'wav'
            unique_filename = f"voice_{str(uuid.uuid4())}.{file_extension}"

            # 读取文件内容
            file_content = file.file.read()

            # 上传到COS
            response = client.put_object(
                Bucket=bucket,
                Body=file_content,
                Key='digitalhuman/voice/'+unique_filename
            )

            # 获取文件访问URL
            file_url = client.get_object_url(
                Bucket=bucket,
                Key=unique_filename
            )

            # 调用create_voice创建音色
            from dashscope.audio.tts_v2 import VoiceEnrollmentService
            service = VoiceEnrollmentService()
            prefix = data.get('prefix', 'custom_voice')
            target_model = data.get('target_model', 'cosyvoice-v2')

            voice_id = service.create_voice(target_model=target_model, prefix=prefix, url=file_url)

            # 更新当前使用的voice_name
            app.config['voice_name'] = voice_id

            logger.info(f"Created voice with ID: {voice_id}")
            return web.json_response({
                'status': 'success',
                'voice_id': voice_id,
                'file_url': file_url
            })

        except Exception as e:
            logger.error(f"Error in upload_voice: {str(e)}")
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)

    appasync.router.add_post('/upload_voice', upload_voice_handler)
    
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('voice upload page: http://<serverip>:'+str(opt.listenport)+'/voice_upload')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    @app.route('/set_voice', methods=['POST'])
    def set_voice():
        data = request.get_json()
        if 'voice' in data:
            app.config['voice_name'] = data['voice']
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error'}), 400

    @app.route('/get_voice_id', methods=['GET'])
    def get_voice_id():
        try:
            voice_id = app.config.get('voice_name', '')
            return jsonify({
                'status': 'success',
                'voice_id': voice_id
            })
        except Exception as e:
            logger.error(f"Error getting voice ID: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500


    @app.route('/upload_voice', methods=['POST'])
    def upload_voice():
        try:
            # 检查是否有文件上传
            if 'voice_file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
            
            file = request.files['voice_file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'No selected file'}), 400

            # 导入腾讯云COS SDK
            from qcloud_cos import CosConfig
            from qcloud_cos import CosS3Client
            import sys
            import logging

            # 配置腾讯云COS
            secret_id = app.config.get('COS_SECRET_ID')  # 请在环境变量或配置文件中设置
            secret_key = app.config.get('COS_SECRET_KEY')  # 请在环境变量或配置文件中设置
            region = app.config.get('COS_REGION', 'ap-guangzhou')  # 替换为您的存储桶地区
            bucket = app.config.get('COS_BUCKET')  # 替换为您的存储桶名称

            config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
            client = CosS3Client(config)

            # 生成唯一的文件名
            import uuid
            file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'wav'
            unique_filename = f"voice_{str(uuid.uuid4())}.{file_extension}"

            # 上传到COS
            response = client.put_object(
                Bucket=bucket,
                Body=file.read(),
                Key=unique_filename
            )

            # 获取文件访问URL
            file_url = client.get_object_url(
                Bucket=bucket,
                Key=unique_filename
            )

            # 调用create_voice创建音色
            import dashscope
            from dashscope.audio.tts_v2 import VoiceEnrollmentService

            service = VoiceEnrollmentService()
            prefix = request.form.get('prefix', 'custom_voice')
            target_model = request.form.get('target_model', 'cosyvoice-v2')

            voice_id = service.create_voice(target_model=target_model, prefix=prefix, url=file_url)

            # 更新当前使用的voice_name
            app.config['voice_name'] = voice_id

            logger.info(f"Created voice with ID: {voice_id}")
            return jsonify({
                'status': 'success',
                'voice_id': voice_id,
                'file_url': file_url
            })

        except Exception as e:
            logger.error(f"Error in upload_voice: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/create_voice', methods=['POST'])
    def create_voice():
        import dashscope
        from dashscope.audio.tts_v2 import VoiceEnrollmentService

        data = request.get_json()
        if 'text' in data:
            text = data['text']
            voice_name = app.config['voice_name']
            voice_id = voice_name.replace(' ','_')
        service = VoiceEnrollmentService()
        prefix = data['prefix'] if 'prefix' in data else "prefix"
        target_model = data['target_model'] if 'target_model' in data else "cosyvoice-v2"

        voice_id = service.create_voice(target_model=target_model, prefix=prefix, url=url)

        logger.info(f"voice id为：{voice_id}")
        return jsonify({'status':'success','voice_id':voice_id})
        
    

class BaseReal:
    def __init__(self, opt, model, avatar):
        self.opt = opt
        self.model = model
        self.avatar = avatar
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = opt.batch_size
        self.processing_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.processing_task = None
        self.is_processing = False
        # 添加图片缓存
        self.image_cache = {}
        self.cache_size = 1000  # 最大缓存数量
        self.cache_lock = threading.Lock()
        # 设置 sessionid
        self.sessionid = getattr(opt, 'sessionid', 0)

    def get_cached_image(self, image_path):
        """获取缓存的图片，如果不存在则加载并缓存"""
        with self.cache_lock:
            if image_path in self.image_cache:
                return self.image_cache[image_path]
            
            # 如果缓存已满，删除最早的缓存
            if len(self.image_cache) >= self.cache_size:
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            
            # 加载新图片并缓存
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    self.image_cache[image_path] = image
                    return image
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
            return None

    def clear_image_cache(self):
        """清除图片缓存"""
        with self.cache_lock:
            self.image_cache.clear()

    def preload_images(self, image_paths):
        """预加载一批图片到缓存"""
        for path in image_paths:
            self.get_cached_image(path)

    async def start_processing(self):
        if not self.is_processing:
            self.is_processing = True
            self.processing_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        while self.is_processing:
            try:
                batch_data = []
                # 收集一批数据
                for _ in range(self.batch_size):
                    if not self.processing_queue.empty():
                        data = await self.processing_queue.get()
                        batch_data.append(data)
                    else:
                        break
                
                if batch_data:
                    # 批量处理数据
                    results = await self._process_batch(batch_data)
                    for result in results:
                        await self.result_queue.put(result)
                else:
                    await asyncio.sleep(0.01)  # 避免空转
            except Exception as e:
                logger.error(f"Error in processing queue: {e}")

    async def _process_batch(self, batch_data):
        # 在这里实现具体的批处理逻辑
        # 将数据转移到 GPU
        batch_tensor = torch.stack([d.to(self.device) for d in batch_data])
        with torch.no_grad():
            results = self.model(batch_tensor)
        return results.cpu()

    async def put_data(self, data):
        await self.processing_queue.put(data)
        await self.start_processing()

    async def get_result(self):
        return await self.result_queue.get()
    
    def render(self, image_path):
        # 使用缓存获取图片
        image = self.get_cached_image(image_path)
        if image is None:
            return None
        # 继续处理图片...

    