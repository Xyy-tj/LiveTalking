import time
import os
import sqlite3
from basereal import BaseReal
from logger import logger
import dotenv

dotenv.load_dotenv()

# 定义数据库路径 (相对于 llm.py 文件或者使用绝对路径)
# 假设 llm.py 和 setting_api.py 在同一目录下，或者 setting_data.db 在项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "setting_data.db") # 确保路径正确

def get_role_config_from_db():
    """从数据库获取角色配置"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT role_name, preset_prompt, script_library_text FROM role_configurations WHERE id = 1")
        row = c.fetchone()
        if row:
            return {
                "role_name": row[0],
                "preset_prompt": row[1],
                "script_library_text": row[2]
            }
        else:
            logger.warning("数据库中未找到角色配置 (id=1)")
            return None
    except sqlite3.Error as e:
        logger.error(f"从数据库读取角色配置失败: {e}")
        return None
    finally:
        if conn:
            conn.close()

def llm_response(message,nerfreal:BaseReal):
    start = time.perf_counter()
    from openai import OpenAI
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key=os.getenv("OPENAI_API_KEY"),
        #base_url
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")

    # 获取角色配置
    role_config = get_role_config_from_db()

    system_prompt_content = """
你是一位专业的AI助手。
回应要求：
1. 语气要友好、乐于助人。
2. 保持专业性和可信度。
3. 回答尽量简洁明了。
""" # 默认后备提示

    if role_config:
        role_name = role_config.get("role_name", "AI助手")
        preset_prompt_text = role_config.get("preset_prompt", "")
        script_library = role_config.get("script_library_text", "")

        system_prompt_content = f"你是名叫 \"{role_name}\" 的直播助手。\n{preset_prompt_text}"
        if script_library:
            system_prompt_content += f"\n\n你可以参考以下话术内容：\n---\n{script_library}\n---"
        logger.info(f"使用数据库中的角色配置: {role_name}")
    else:
        logger.warning("无法从数据库加载角色配置，使用默认系统提示。")

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{'role': 'system', 'content': system_prompt_content},
                  {'role': 'user', 'content': message}],
        stream=True,
        # 通过以下设置，在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True}
    )
    result=""
    first = True
    for chunk in completion:
        if len(chunk.choices)>0:
            #print(chunk.choices[0].delta.content)
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start}s")
                first = False
            msg = chunk.choices[0].delta.content
            if msg is None:
                msg = "" # 如果LLM返回的增量内容是None，则视为空字符串
            lastpos=0
            #msglist = re.split('[,.!;:，。！?]',msg)
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；" :
                    result = result+msg[lastpos:i+1]
                    lastpos = i+1
                    if len(result)>10:
                        logger.info(result)
                        nerfreal.put_msg_txt(result)
                        result=""
            result = result+msg[lastpos:]
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    nerfreal.put_msg_txt(result)    