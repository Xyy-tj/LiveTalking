from fastapi import FastAPI, File, UploadFile, Form, Body, Depends, HTTPException, Header, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
import os
import uuid
import shutil
import subprocess
from logger import logger
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
import asyncio
import threading
from typing import Optional, Dict, Callable, Awaitable
import queue
from pydantic import BaseModel
from collections import deque
import json
import signal
import psutil
from fastapi.staticfiles import StaticFiles
import platform
import base64 # 用于演示目的的Key编码，不安全
import re # 新增导入
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteBaseResponse
from starlette.responses import RedirectResponse as StarletteRedirectResponse
from urllib.parse import urlencode # 新增导入

load_dotenv()
app = FastAPI()

# 优雅关闭：定义应用关闭时执行的函数
async def on_app_shutdown():
    logger.info("FastAPI application (setting_api.py) is shutting down. Cleaning up services.")
    cleanup_service() # 这个函数包含了停止 app.py 子进程的逻辑

# 注册关闭事件处理程序
app.add_event_handler("shutdown", on_app_shutdown)

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

# --- 授权机制开始 ---
# 重要警告: 以下 Key 生成和解析逻辑仅为演示，使用 Base64 编码，极不安全。
# 生产环境中必须替换为强加密算法 (例如 cryptography.fernet)。

def get_machine_id() -> Optional[str]:
    """获取机器唯一标识符"""
    system = platform.system()
    try:
        if system == "Windows":
            # 尝试使用 wmic 获取 UUID
            result = subprocess.check_output(
                ['wmic', 'csproduct', 'get', 'uuid'], 
                universal_newlines=True, 
                stderr=subprocess.DEVNULL
            )
            machine_id = result.split('\\n')[1].strip()
            if machine_id:
                return machine_id
        elif system == "Linux":
            # 尝试读取 /etc/machine-id 或 /var/lib/dbus/machine-id
            for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        machine_id = f.read().strip()
                        if machine_id:
                            return machine_id
            # 备选：尝试获取第一个非本地回环MAC地址
            for interface, snics in psutil.net_if_addrs().items():
                for snic in snics:
                    if snic.family == psutil.AF_LINK and snic.address and snic.address != "00:00:00:00:00:00":
                        return snic.address.replace(":", "").replace("-", "").lower()
        elif system == "Darwin": # macOS
            result = subprocess.check_output(
                ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'], 
                universal_newlines=True,
                stderr=subprocess.DEVNULL
            )
            for line in result.split('\\n'):
                if "IOPlatformUUID" in line:
                    machine_id = line.split('"')[-2]
                    if machine_id:
                        return machine_id
    except Exception as e:
        logger.error(f"获取机器码失败 ({system}): {e}")
    
    logger.warning("无法确定唯一的机器ID，将尝试使用主机名作为备用（不推荐）。")
    return platform.node() # 备用方案，唯一性较差

def parse_license_key(license_key: str) -> Optional[Dict[str, any]]:
    """
    解析（"解密"）License Key。
    当前实现使用 Base64，极不安全，仅为演示。
    真实场景下，这里应该是对应 generate_license_key 中加密算法的解密过程。
    """
    try:
        # 假设 Key 格式为: base64(machine_id|expiry_date_isoformat)
        decoded_payload = base64.urlsafe_b64decode(license_key.encode()).decode()
        parts = decoded_payload.split('|')
        if len(parts) == 2:
            machine_id = parts[0]
            expiry_date_str = parts[1]
            return {
                "machine_id": machine_id,
                "expiry_date": datetime.fromisoformat(expiry_date_str)
            }
        else:
            logger.warning(f"License Key 格式错误 (parts): {license_key}")
            return None
    except Exception as e:
        logger.error(f"解析 License Key 失败 '{license_key}': {e}")
        return None

async def verify_license_dependency(
    x_license_key_header: Optional[str] = Header(None, alias="X-License-Key"),
    x_license_key_cookie: Optional[str] = Cookie(None, alias="X-License-Key-Cookie")
):
    license_key_to_check = x_license_key_header
    source = "Header"

    if not license_key_to_check and x_license_key_cookie:
        license_key_to_check = x_license_key_cookie
        source = "Cookie"
    
    if not license_key_to_check:
        logger.warning("【API授权依赖】请求头和 Cookie 中均未找到授权码。")
        raise HTTPException(status_code=401, detail="授权失败: 未提供 License Key (Error: KEY_REQUIRED_ANYWHERE)")

    logger.info(f"【API授权依赖】使用 {source} 中的授权码进行验证。")
    try:
        validated_data = await validate_license_key_logic(license_key_to_check)
        return {
            "machine_id": validated_data["machine_id"], 
            "expires_at": validated_data["expiry_date"], 
            "license_key_used": license_key_to_check,
            "auth_source": source
        }
    except HTTPException as e:
        # 如果是通过Cookie授权失败，可能需要清除无效的Cookie
        # 但依赖项通常不直接操作响应。中间件更适合处理。
        # 这里仅重新抛出异常，让中间件或全局异常处理器决定。
        logger.warning(f"【API授权依赖】验证失败 ({source}): {e.detail}")
        raise e

# --- 授权机制结束 ---

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

# 挂载 static 目录以便访问 web 下的文件
app.mount("/web", StaticFiles(directory=os.path.join(BASE_DIR, "web")), name="web")

# --- 新的授权页面和逻辑 ---

ERROR_MESSAGES_MAP = {
    "SYS_KEY_MISSING": "系统授权服务未配置，请联系管理员。",
    "KEY_MISMATCH": "提供的授权码无效。",
    "SYS_KEY_PARSE_FAIL": "系统授权信息格式错误，请联系管理员。",
    "MID_FAIL": "无法获取设备信息以验证授权，请联系管理员。",
    "MID_MISMATCH": "授权码与当前设备不匹配。",
    "KEY_EXPIRED": "授权码已过期。",
    "KEY_REQUIRED": "请输入授权码。",
    "COOKIE_VALIDATION_FAILED": "当前授权凭证无效或已过期，请重新授权。",
    "SESSION_EXPIRED_OR_INVALIDATED": "您的会话已过期或授权已失效，请重新登录。",
    "UNKNOWN_ERROR": "发生未知错误，请重试或联系管理员。"
}

AUTH_PAGE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"UTF-8\">
    <title>请授权 - LiveTalking 服务</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background-color: #f0f2f5; }}
        .container {{ background: white; padding: 30px 40px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.1); text-align: center; width: 100%; max-width: 420px; }}
        h2 {{ margin-top:0; color: #333; font-size: 24px; margin-bottom: 15px;}}
        p {{ color: #555; margin-bottom: 25px; font-size: 16px; line-height: 1.6;}}
        input[type=\"text\"] {{ padding: 14px; margin-bottom: 25px; width: calc(100% - 30px); border: 1px solid #d9d9d9; border-radius: 6px; font-size: 16px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.075); }}
        input[type=\"text\"]:focus {{ border-color: #409EFF; box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2); outline: none; }}
        button {{ padding: 14px 28px; background-color: #409EFF; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 500; transition: background-color 0.2s ease-in-out; width: 100%; }}
        button:hover {{ background-color: #66b1ff; }}
        .error-message {{ background-color: #fef0f0; color: #f56c6c; border: 1px solid #fde2e2; padding: 12px 15px; border-radius: 6px; margin-bottom: 20px; text-align: left; font-size: 14px; }}
        .app-title {{ font-weight: 600; color: #409EFF;}}
    </style>
</head>
<body>
    <div class=\"container\">
        <h2>欢迎使用 <span class=\"app-title\">LiveTalking</span></h2>
        {{error_html}}
        <p>为确保服务安全，请输入您的授权码以继续访问。</p>
        <form method=\"POST\" action=\"/submit_license_key_from_auth_page\">
            <input type=\"text\" name=\"license_key\" placeholder=\"授权码\" required value=\"{{license_key_value}}\">
            <br>
            <button type=\"submit\">验证并访问</button>
        </form>
    </div>
</body>
</html>
"""

async def validate_license_key_logic(license_key: Optional[str]):
    system_license_key_env = os.getenv("SYSTEM_LICENSE_KEY")
    logger.info(f"SYSTEM_LICENSE_KEY: {system_license_key_env}")
    if not system_license_key_env:
        logger.error("【授权验证】SYSTEM_LICENSE_KEY 环境变量未设置。")
        raise HTTPException(status_code=503, detail="授权服务未配置或不可用 (Error: SYS_KEY_MISSING)")

    if not license_key:
        logger.warning("【授权验证】未提供 License Key。")
        raise HTTPException(status_code=401, detail="授权失败: 未提供 License Key (Error: KEY_REQUIRED)")

    if license_key != system_license_key_env:
        logger.warning(f"【授权验证】提供的 License Key 与系统配置不匹配。")
        raise HTTPException(status_code=403, detail="授权失败: License Key 无效 (Error: KEY_MISMATCH)")

    parsed_key_data = parse_license_key(license_key)
    if not parsed_key_data:
        logger.error(f"【授权验证】系统配置的 License Key 格式似乎有误 (解析失败): {license_key}")
        raise HTTPException(status_code=500, detail="授权系统内部错误: 系统 Key 解析失败 (Error: SYS_KEY_PARSE_FAIL)")

    key_machine_id = parsed_key_data["machine_id"]
    expiry_date = parsed_key_data["expiry_date"]
    current_machine_id = get_machine_id()

    if not current_machine_id:
        logger.error("【授权验证】无法确定当前机器的机器码用于验证。")
        raise HTTPException(status_code=500, detail="授权系统内部错误: 无法获取机器码 (Error: MID_FAIL)")

    if key_machine_id != current_machine_id:
        logger.warning(f"【授权验证】License Key 中的机器码 ({key_machine_id}) 与当前机器 ({current_machine_id}) 不匹配。")
        raise HTTPException(status_code=403, detail=f"授权失败: License Key 与当前设备不匹配 (Error: MID_MISMATCH)")

    if datetime.now() > expiry_date:
        logger.warning(f"【授权验证】License Key 已于 {expiry_date.isoformat()} 过期。")
        raise HTTPException(status_code=403, detail=f"授权失败: License Key 已过期 (Expired on: {expiry_date.date()}) (Error: KEY_EXPIRED)")
    
    logger.info(f"授权检查通过 (通用逻辑): 机器码 {current_machine_id}, Key 有效期至 {expiry_date.isoformat()}.")
    return parsed_key_data

@app.get("/auth_page", response_class=HTMLResponse)
async def get_auth_page(
    error_code: Optional[str] = None,
    license_key_value: Optional[str] = "" # For pre-filling if needed, though generally not for security
):
    err_html_content = ""
    if error_code:
        error_message_text = ERROR_MESSAGES_MAP.get(error_code, ERROR_MESSAGES_MAP["UNKNOWN_ERROR"])
        err_html_content = f"<div class='error-message'>{error_message_text}</div>"
    
    return AUTH_PAGE_HTML_TEMPLATE.format(error_html=err_html_content, license_key_value=license_key_value)

@app.post("/submit_license_key_from_auth_page")
async def handle_license_key_submission(license_key: str = Form(...)):
    try:
        await validate_license_key_logic(license_key)
        # 授权成功
        response = RedirectResponse(url="/web/setting.html", status_code=303)
        # Cookie 有效期 30 天
        response.set_cookie(key="X-License-Key-Cookie", value=license_key, httponly=True, samesite="Lax", max_age=30*24*60*60, path="/")
        logger.info(f"授权码提交成功，设置授权 Cookie。")
        return response
    except HTTPException as e:
        error_code = "UNKNOWN_ERROR"
        match = re.search(r"\(Error:\s*([A-Z_]+)\)", e.detail)
        if match:
            error_code = match.group(1)
        logger.warning(f"授权码提交失败: {e.detail} (Code: {error_code})")
        # 重定向回授权页面并携带错误代码和用户尝试过的key（可选）
        # 安全起见，不在URL中传递用户尝试过的key。
        redirect_url = f"/auth_page?error_code={error_code}"
        return RedirectResponse(url=redirect_url, status_code=303)

@app.get("/logout_and_reauth")
async def logout_and_reauth_endpoint():
    response = RedirectResponse(url="/auth_page?reauth=true", status_code=303)
    response.delete_cookie("X-License-Key-Cookie", path="/")
    logger.info("用户请求重新授权，清除授权 Cookie。")
    return response

# --- 认证中间件 ---
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: Callable[[StarletteRequest], Awaitable[StarletteBaseResponse]]):
        # 豁免路径
        exempt_paths = ["/auth_page", "/submit_license_key_from_auth_page", "/logout_and_reauth", "/favicon.ico"]
        if any(request.url.path.startswith(p) for p in exempt_paths):
            return await call_next(request)
        
        # API 路径（通常包含 /upload_, /list_, /delete_, /update_, /start_, /stop_, /service_, /avatar_status）
        # 由 verify_license_dependency 保护，中间件不直接处理它们的授权逻辑，但可以检查系统配置。
        api_path_patterns = ["/upload_", "/list_", "/delete_", "/update_", "/start_", "/stop_", "/service_", "/avatar_status"]
        is_api_call = any(pattern in request.url.path for pattern in api_path_patterns)

        # 系统级检查: SYSTEM_LICENSE_KEY 是否设置
        system_license_key_env = os.getenv("SYSTEM_LICENSE_KEY")
        if not system_license_key_env:
            logger.error(f"AuthMiddleware: SYSTEM_LICENSE_KEY 环境变量未设置。 Path: {request.url.path}")
            # API 调用会让其依赖项处理。页面请求在这里处理。
            if not is_api_call:
                 return HTMLResponse(content="<h1>服务配置错误 (SYS_KEY_MISSING)</h1><p>请联系管理员。系统授权未配置。</p>", status_code=503)
            # 对于API调用，允许其继续，由 verify_license_dependency 抛出错误
            return await call_next(request)

        # 页面路径授权检查 (/, /web/*)
        if request.url.path == "/" or request.url.path.startswith("/web"):
            license_key_cookie = request.cookies.get("X-License-Key-Cookie")
            is_authorized_for_page = False
            auth_error_code_for_redirect = None

            if license_key_cookie:
                try:
                    await validate_license_key_logic(license_key_cookie)
                    is_authorized_for_page = True
                    logger.info(f"AuthMiddleware: Cookie 授权成功 for page: {request.url.path}")
                except HTTPException as e:
                    match = re.search(r"\(Error:\s*([A-Z_]+)\)", e.detail)
                    if match: auth_error_code_for_redirect = match.group(1)
                    else: auth_error_code_for_redirect = "COOKIE_VALIDATION_FAILED"
                    logger.warning(f"AuthMiddleware: Cookie 授权失败 for page {request.url.path} - {e.detail}. Code: {auth_error_code_for_redirect}")
            else:
                logger.info(f"AuthMiddleware: 未找到授权 Cookie for page: {request.url.path}")
                auth_error_code_for_redirect = "KEY_REQUIRED" # No cookie means key is required

            if not is_authorized_for_page:
                redirect_url = f"/auth_page"
                if auth_error_code_for_redirect:
                    redirect_url += f"?error_code={auth_error_code_for_redirect}"
                
                response = StarletteRedirectResponse(url=redirect_url, status_code=307)
                if license_key_cookie: # 如果存在无效/过期的cookie，清除它
                    logger.info(f"AuthMiddleware: 清除无效的授权 Cookie for page {request.url.path}")
                    response.delete_cookie("X-License-Key-Cookie", path="/")
                return response
            
            # 如果授权成功且访问根路径，重定向到主面板
            if request.url.path == "/" and is_authorized_for_page:
                logger.info("AuthMiddleware: 已授权访问根路径, 重定向到 /web/setting.html")
                return StarletteRedirectResponse(url="/web/setting.html", status_code=303)

        # 其他所有请求（包括已授权的页面请求和API请求）
        return await call_next(request)

# 在CORS之后，路由之前添加中间件
app.add_middleware(AuthMiddleware)


@app.get("/")
async def root_redirect_handler():
    # 此路由理论上会被中间件处理。
    # 如果中间件逻辑有遗漏，或作为备用，可以保留一个简单的重定向。
    # 但中间件应该完全覆盖 / 的场景。
    # 为清晰起见，可以移除，或留一个最简形式。
    # 如果中间件正确工作，此路由不会在正常流程中被直接调用。
    # 暂时注释掉，如果发现问题再考虑恢复。
    # return RedirectResponse(url="/web/setting.html") # Or let middleware handle
    pass


@app.post("/upload_voice", dependencies=[Depends(verify_license_dependency)])
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

@app.post("/upload_avatar", dependencies=[Depends(verify_license_dependency)])
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

@app.delete("/delete_voice/{voice_id}", dependencies=[Depends(verify_license_dependency)])
def delete_voice(voice_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM voices WHERE voice_id = ?", (voice_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/delete_avatar/{avatar_id}", dependencies=[Depends(verify_license_dependency)])
def delete_avatar(avatar_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM avatars WHERE avatar_id = ?", (avatar_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/update_voice", dependencies=[Depends(verify_license_dependency)])
def update_voice(id: int = Form(...), prefix: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE voices SET prefix = ? WHERE id = ?", (prefix, id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/update_avatar", dependencies=[Depends(verify_license_dependency)])
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
            logger.info(f"[Service Process]: {output.strip()}")
    
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

@app.post("/start_service", dependencies=[Depends(verify_license_dependency)])
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

@app.post("/stop_service", dependencies=[Depends(verify_license_dependency)])
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