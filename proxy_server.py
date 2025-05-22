from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from fastapi.responses import StreamingResponse, PlainTextResponse
import uvicorn
import logging # 添加日志模块
import json # 添加 json 模块

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# SRS 服务器地址
SRS_SERVER = "http://124.223.167.54:1985"

async def forward_request(request: Request, method: str):
    params = dict(request.query_params)
    url = f"{SRS_SERVER}{request.url.path.replace('/proxy', '')}"
    # 仅转发必要的头部，移除可能引起问题的头部
    # 特别注意 content-type, content-length 等头部，如果由 httpx 自动处理，则不应从原始请求中直接复制
    headers_to_forward = {}
    for k, v in request.headers.items():
        # 过滤掉 host 和一些特定头部，其他大部分头部可以考虑转发
        if k.lower() not in ['host', 'origin', 'referer', 'connection', 'keep-alive', 'accept-encoding', 'content-length']:
            headers_to_forward[k] = v
    headers_to_forward["user-agent"] = "FastAPI-Proxy/0.1.1"

    body = await request.body()

    logger.info(f"--- Outgoing Request to SRS ---")
    logger.info(f"URL: {method} {url}")
    logger.info(f"Params: {params}")
    logger.info(f"Headers: {headers_to_forward}")
    if body:
        try:
            logger.info(f"Body: {body.decode()}") # 尝试解码为字符串打印，如果失败则打印原始字节
        except UnicodeDecodeError:
            logger.info(f"Body (bytes): {body}")
    else:
        logger.info("Body: None")
    logger.info(f"-------------------------------")

    async with httpx.AsyncClient() as client:
        try:
            if method == "GET":
                resp = await client.get(url, params=params, headers=headers_to_forward)
            elif method == "POST":
                # 对于POST请求，如果原始请求有Content-Type，则使用它
                # 否则 httpx 会根据 body 类型自动设置 (如 application/octet-stream)
                if 'content-type' not in headers_to_forward and body:
                     # 尝试从原始请求中获取content-type
                    original_content_type = request.headers.get('content-type')
                    if original_content_type:
                        headers_to_forward['content-type'] = original_content_type
                    elif body: # 如果还是没有，且有body，默认设为 application/octet-stream 或 application/json (取决于body内容)
                        try:
                            json.loads(body.decode())
                            headers_to_forward['content-type'] = 'application/json'
                        except:
                            headers_to_forward['content-type'] = 'application/octet-stream'
                
                logger.info(f"Final Headers for POST: {headers_to_forward}")
                resp = await client.post(url, params=params, content=body, headers=headers_to_forward)
            elif method == "OPTIONS":
                logger.info("Handling OPTIONS request directly.")
                return PlainTextResponse("OK", headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization", # 更明确地列出允许的头部
                    "Access-Control-Max-Age": "86400" # 预检请求的缓存时间
                })
            else:
                logger.warning(f"Unsupported method: {method}")
                raise HTTPException(status_code=405, detail="Method Not Allowed")

            logger.info(f"--- Incoming Response from SRS ---")
            logger.info(f"Status Code: {resp.status_code}")
            logger.info(f"Headers: {dict(resp.headers)}")
            # logger.info(f"Content: {await resp.aread()}") # 注意：读取内容后可能无法再次读取，仅调试时使用
            logger.info(f"--------------------------------")

            response_headers = dict(resp.headers)
            response_headers["Access-Control-Allow-Origin"] = "*"
            response_headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
            response_headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-encoding", None)
            response_headers.pop("alt-svc", None) # SRS 可能会返回这个头部

            return StreamingResponse(
                content=resp.iter_bytes(),
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type")
            )

        except httpx.RequestError as e:
            logger.error(f"Proxy RequestError to {e.request.method} {e.request.url}: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Proxy connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Internal server error during proxying: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/proxy/rtc/v1/whep/")
async def proxy_whep_get(request: Request):
    return await forward_request(request, "GET")

@app.post("/proxy/rtc/v1/whep/")
async def proxy_whep_post(request: Request):
    return await forward_request(request, "POST")

@app.options("/proxy/rtc/v1/whep/")
async def proxy_whep_options(request: Request):
    return await forward_request(request, "OPTIONS")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info") # 启用 uvicorn 的日志 