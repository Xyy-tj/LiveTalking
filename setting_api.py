from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
import shutil
import subprocess
from logger import logger
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import asyncio
import threading
from typing import Optional
import queue
from pydantic import BaseModel
from collections import deque
import json
import signal
import psutil

load_dotenv()
app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "setting_data.db")
WAV2LIP_DIR = os.path.join(BASE_DIR, "wav2lip")
RESULT_AVATARS_DIR = os.path.join(WAV2LIP_DIR, "results", "avatars")
DATA_AVATARS_DIR = os.path.join(BASE_DIR, "data", "avatars")
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "temp")

# 确保所有必要的目录存在
os.makedirs(WAV2LIP_DIR, exist_ok=True)
os.makedirs(RESULT_AVATARS_DIR, exist_ok=True)
os.makedirs(DATA_AVATARS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# 腾讯云COS配置（建议用环境变量或配置文件管理）
if os.getenv("COS_SECRET_ID") is None:
    raise ValueError("COS_SECRET_ID is not set")
if os.getenv("COS_SECRET_KEY") is None:
    raise ValueError("COS_SECRET_KEY is not set")
if os.getenv("COS_REGION") is None:
    raise ValueError("COS_REGION is not set")
if os.getenv("COS_BUCKET") is None:
    raise ValueError("COS_BUCKET is not set")

COS_SECRET_ID = os.getenv("COS_SECRET_ID")
COS_SECRET_KEY = os.getenv("COS_SECRET_KEY")
COS_REGION = os.getenv("COS_REGION", "ap-guangzhou")
COS_BUCKET = os.getenv("COS_BUCKET")

logger.info(f"COS_SECRET_ID: {COS_SECRET_ID}")
logger.info(f"COS_SECRET_KEY: {COS_SECRET_KEY}")
logger.info(f"COS_REGION: {COS_REGION}")
logger.info(f"COS_BUCKET: {COS_BUCKET}")

# 使用 deque 存储最近的100条日志
MAX_LOGS = 100
service_logs = deque(maxlen=MAX_LOGS)
service_process = None
log_thread = None
service_port = 6006

class ServiceConfig(BaseModel):
    voice_id: str
    avatar_id: str
    extra_params: str = ""

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS voices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voice_id TEXT,
            file_url TEXT,
            prefix TEXT,
            target_model TEXT,
            created_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS avatars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            avatar_id TEXT,
            file_url TEXT,
            name TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            progress INTEGER DEFAULT 0,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 重新初始化数据库（如果表结构已经存在，需要先删除旧表）
def reinit_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS voices")
    c.execute("DROP TABLE IF EXISTS avatars")
    conn.commit()
    conn.close()
    init_db()

# 如果需要重新初始化数据库，取消下面这行的注释
# reinit_db()

@app.get("/")
async def root():
    return FileResponse("web/setting.html")


@app.post("/upload_voice")
async def upload_voice(
    file: UploadFile = File(...),
    prefix: str = Form("custom_voice"),
    targetModel: str = Form("cosyvoice-v2")
):
    logger.info(f"upload_voice: {file}, {prefix}, {targetModel}")
    try:
        # 检查文件
        if not file or not file.filename:
            return JSONResponse({"status": "error", "message": "No file uploaded"}, status_code=400)

        # 导入腾讯云COS SDK
        from qcloud_cos import CosConfig, CosS3Client

        # 配置COS
        config = CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID, SecretKey=COS_SECRET_KEY)
        client = CosS3Client(config)

        # 生成唯一文件名
        file_extension = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'wav'
        unique_filename = f"voice_{str(uuid.uuid4())}.{file_extension}"

        # 上传到COS
        file_bytes = await file.read()
        client.put_object(
            Bucket=COS_BUCKET,
            Body=file_bytes,
            Key=unique_filename
        )

        # 获取文件访问URL
        file_url = client.get_object_url(
            Bucket=COS_BUCKET,
            Key=unique_filename
        )

        # 调用Dashscope创建音色
        from dashscope.audio.tts_v2 import VoiceEnrollmentService
        service = VoiceEnrollmentService()
        voice_id = service.create_voice(target_model=targetModel, prefix=prefix, url=file_url)

        logger.info(f"Created voice with ID: {voice_id}")

        # 写入sqlite
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO voices (voice_id, file_url, prefix, target_model, created_at) VALUES (?, ?, ?, ?, ?)",
            (voice_id, file_url, prefix, targetModel, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return {
            "status": "success",
            "voice_id": voice_id,
            "file_url": file_url
        }

    except Exception as e:
        logger.error(f"Error in upload_voice: {str(e)}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

def process_avatar_generation(avatar_id: str, temp_video_path: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 更新状态为处理中
        c.execute(
            "UPDATE avatars SET status = ?, progress = ? WHERE avatar_id = ?",
            ("processing", 0, avatar_id)
        )
        conn.commit()

        # 获取wav2lip目录下的genavatar.py的完整路径
        genavatar_script = os.path.join(WAV2LIP_DIR, "genavatar.py")
        if not os.path.exists(genavatar_script):
            raise Exception(f"找不到处理脚本: {genavatar_script}")

        # 执行wav2lip处理命令
        cmd = [
            "python",
            genavatar_script,
            "--video_path", temp_video_path,
            "--img_size", "256",
            "--avatar_id", avatar_id
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        logger.info(f"Working directory: {WAV2LIP_DIR}")
        
        process = subprocess.Popen(
            cmd,
            cwd=WAV2LIP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
            bufsize=1
        )

        # 实时读取并记录输出
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            if output:
                logger.info(f"wav2lip output: {output.strip()}")
                # 更新进度（这里需要根据实际输出解析进度）
                if "Processing frame" in output:
                    try:
                        progress = int(output.split()[2].strip('%'))
                        c.execute(
                            "UPDATE avatars SET progress = ? WHERE avatar_id = ?",
                            (progress, avatar_id)
                        )
                        conn.commit()
                    except:
                        pass
            if error:
                logger.error(f"wav2lip error: {error.strip()}")
                
            if output == '' and error == '' and process.poll() is not None:
                break
        
        if process.returncode != 0:
            raise Exception(f"处理视频失败: 进程返回值 {process.returncode}")

        # 检查生成的文件夹是否存在
        result_avatar_path = os.path.join(RESULT_AVATARS_DIR, avatar_id)
        if not os.path.exists(result_avatar_path):
            raise Exception(f"处理后的头像文件未生成，期望路径: {result_avatar_path}")

        # 移动生成的文件夹到data/avatars
        target_path = os.path.join(DATA_AVATARS_DIR, avatar_id)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(result_avatar_path, target_path)

        logger.info(f"Moved avatar from {result_avatar_path} to {target_path}")

        # 更新状态为完成
        c.execute(
            "UPDATE avatars SET status = ?, progress = ?, file_url = ? WHERE avatar_id = ?",
            ("completed", 100, target_path, avatar_id)
        )
        conn.commit()

        # 清理临时文件
        try:
            os.remove(temp_video_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_video_path}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in avatar generation: {str(e)}")
        # 更新状态为失败
        c.execute(
            "UPDATE avatars SET status = ?, error_message = ? WHERE avatar_id = ?",
            ("failed", str(e), avatar_id)
        )
        conn.commit()
    finally:
        conn.close()

@app.post("/upload_avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    try:
        # 检查文件格式
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext != ".mp4":
            return JSONResponse({
                "status": "error",
                "message": "仅支持MP4格式视频文件"
            })

        # 生成唯一的avatar_id
        avatar_id = f"wav2lip256_avatar{uuid.uuid4().hex[:8]}"
        
        # 保存上传的视频文件到临时目录
        temp_video_path = os.path.join(TEMP_UPLOAD_DIR, f"{avatar_id}{ext}")
        with open(temp_video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved video file to {temp_video_path}")

        # 写入sqlite数据库，初始状态为pending
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO avatars (avatar_id, name, status, progress, created_at) VALUES (?, ?, ?, ?, ?)",
            (avatar_id, name, "pending", 0, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        # 启动异步处理线程
        thread = threading.Thread(
            target=process_avatar_generation,
            args=(avatar_id, temp_video_path)
        )
        thread.start()

        return {
            "status": "success",
            "message": "数字人生成任务已提交",
            "avatar_id": avatar_id
        }

    except Exception as e:
        logger.error(f"Error in upload_avatar: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@app.get("/avatar_status/{avatar_id}")
def get_avatar_status(avatar_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT status, progress, error_message, file_url FROM avatars WHERE avatar_id = ?",
            (avatar_id,)
        )
        row = c.fetchone()
        conn.close()

        if not row:
            return JSONResponse({
                "status": "error",
                "message": "找不到指定的数字人任务"
            })

        return {
            "status": "success",
            "data": {
                "status": row[0],
                "progress": row[1],
                "error_message": row[2],
                "file_url": row[3]
            }
        }

    except Exception as e:
        logger.error(f"Error in get_avatar_status: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@app.get("/list_voices")
def list_voices():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, voice_id, file_url, prefix, target_model, created_at FROM voices ORDER BY created_at DESC")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        return [
            {
                "id": row[0],
                "voice_id": row[1],
                "file_url": row[2],
                "prefix": row[3],
                "target_model": row[4],
                "created_at": row[5]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error in list_voices: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"获取音色列表失败: {str(e)}"}
        )

@app.get("/list_avatars")
def list_avatars():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT 
                id, 
                avatar_id, 
                file_url, 
                name, 
                status,
                progress,
                error_message,
                created_at 
            FROM avatars 
            ORDER BY created_at DESC
        """)
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        return [
            {
                "id": row[0],
                "avatar_id": row[1],
                "file_url": row[2],
                "name": row[3],
                "status": row[4] or "pending",  # 如果为 NULL 则默认为 pending
                "progress": row[5] or 0,        # 如果为 NULL 则默认为 0
                "error_message": row[6],        # 保持 NULL 如果没有错误
                "created_at": row[7]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error in list_avatars: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"获取数字人列表失败: {str(e)}"}
        )

@app.delete("/delete_voice/{voice_id}")
def delete_voice(voice_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM voices WHERE voice_id = ?", (voice_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/delete_avatar/{avatar_id}")
def delete_avatar(avatar_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM avatars WHERE avatar_id = ?", (avatar_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/update_voice")
def update_voice(id: int = Form(...), prefix: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE voices SET prefix = ? WHERE id = ?", (prefix, id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/update_avatar")
def update_avatar(
    id: int = Form(...),
    name: Optional[str] = Form(default=...),
    avatar_id: Optional[str] = Form(default=...),
    status: Optional[str] = Form(default=...),
    progress: Optional[int] = Form(default=...),
    error_message: Optional[str] = Form(default=...),
    file_url: Optional[str] = Form(default=...)
):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # 构建更新语句和参数
        update_fields = []
        params = []
        
        # 获取表单数据中实际包含的字段
        form_data = {}
        for key, value in {
            'name': name,
            'avatar_id': avatar_id,
            'status': status,
            'progress': progress,
            'error_message': error_message,
            'file_url': file_url
        }.items():
            # 如果参数在请求中出现（包括空值），则更新
            if value is not Ellipsis:  # Ellipsis 表示参数未在请求中出现
                form_data[key] = value
                update_fields.append(f"{key} = ?")
                params.append(value)
            
        if not update_fields:
            return JSONResponse({
                "status": "error",
                "message": "没有提供需要更新的字段"
            })
            
        # 添加 WHERE 条件的参数
        params.append(id)
        
        # 构建并执行 SQL 语句
        sql = f"UPDATE avatars SET {', '.join(update_fields)} WHERE id = ?"
        logger.info(f"Executing SQL: {sql} with params: {params}")
        logger.info(f"Updating fields: {form_data}")
        
        c.execute(sql, params)
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "updated_fields": form_data  # 返回实际更新的字段
        }
        
    except Exception as e:
        logger.error(f"Error in update_avatar: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

def log_reader(process, log_queue):
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            # 添加时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = {
                'timestamp': timestamp,
                'message': output.strip(),
                'type': 'info'  # 可以根据输出内容判断类型
            }
            log_queue.append(log_entry)
    
    # 添加进程结束的日志
    if process.poll() is not None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'message': f'服务进程已结束，返回值: {process.poll()}',
            'type': 'info' if process.poll() == 0 else 'error'
        }
        log_queue.append(log_entry)
    
    process.stdout.close()

def kill_process_by_port(port):
    """通过端口号杀死进程"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # 获取进程的所有连接
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        # 发送 SIGTERM 信号
                        os.kill(proc.pid, signal.SIGTERM)
                        # 等待进程结束
                        psutil.Process(proc.pid).wait(timeout=3)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {str(e)}")
    return False

def cleanup_service():
    """清理服务相关的所有资源"""
    global service_process, log_thread
    try:
        # 1. 杀死可能占用端口的进程
        kill_process_by_port(service_port)
        
        # 2. 如果进程还在运行，尝试终止它
        if service_process:
            try:
                # 发送 SIGTERM 信号
                service_process.terminate()
                # 等待进程结束，最多等待5秒
                try:
                    service_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 如果等待超时，强制结束进程
                    service_process.kill()
                    service_process.wait()
            except Exception as e:
                logger.error(f"Error terminating service process: {str(e)}")
        
        # 3. 重置进程和线程变量
        service_process = None
        log_thread = None
        
        # 4. 清空日志
        service_logs.clear()
        
        return True
    except Exception as e:
        logger.error(f"Error in cleanup_service: {str(e)}")
        return False

@app.post("/start_service")
def start_service(config: ServiceConfig):
    global service_process, log_thread
    try:
        # 先清理旧的服务
        cleanup_service()
        
        # 添加启动日志
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        service_logs.append({
            'timestamp': timestamp,
            'message': f'正在启动服务...\n参数: voice_id={config.voice_id}, avatar_id={config.avatar_id}',
            'type': 'info'
        })

        # 构建命令
        cmd = ["uv", "run", "app.py", 
               "--voice_id_dashscope", config.voice_id,
               "--avatar_id", config.avatar_id]
        
        # 添加额外参数
        if config.extra_params:
            extra_params = config.extra_params.strip().split('\n')
            for param in extra_params:
                param = param.strip()
                if param:
                    cmd.extend(param.split())

        logger.info(f"Starting service with command: {' '.join(cmd)}")
        service_logs.append({
            'timestamp': timestamp,
            'message': f'执行命令: {" ".join(cmd)}',
            'type': 'info'
        })
        
        # 启动服务进程
        service_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows特定标志
        )

        # 启动日志读取线程
        log_thread = threading.Thread(target=log_reader, args=(service_process, service_logs))
        log_thread.daemon = True
        log_thread.start()

        return JSONResponse({
            "status": "success",
            "message": "服务已启动"
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error starting service: {error_msg}")
        # 添加错误日志
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        service_logs.append({
            'timestamp': timestamp,
            'message': f'启动服务失败: {error_msg}',
            'type': 'error'
        })
        return JSONResponse({
            "status": "error",
            "message": f"启动服务失败: {error_msg}"
        })

@app.post("/stop_service")
def stop_service():
    try:
        if service_process is None:
            return JSONResponse({
                "status": "error",
                "message": "服务未运行"
            })

        # 添加停止日志
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        service_logs.append({
            'timestamp': timestamp,
            'message': '正在停止服务...',
            'type': 'info'
        })

        # 清理所有服务相关资源
        if cleanup_service():
            # 添加完成日志
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            service_logs.append({
                'timestamp': timestamp,
                'message': '服务已停止',
                'type': 'info'
            })

            return JSONResponse({
                "status": "success",
                "message": "服务已停止"
            })
        else:
            raise Exception("清理服务资源失败")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error stopping service: {error_msg}")
        # 添加错误日志
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        service_logs.append({
            'timestamp': timestamp,
            'message': f'停止服务失败: {error_msg}',
            'type': 'error'
        })
        return JSONResponse({
            "status": "error",
            "message": f"停止服务失败: {error_msg}"
        })

@app.get("/service_logs")
def get_service_logs():
    return JSONResponse({
        "status": "success",
        "logs": list(service_logs)  # 转换 deque 为列表
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("setting_api:app", host="0.0.0.0", port=8001, reload=True) 