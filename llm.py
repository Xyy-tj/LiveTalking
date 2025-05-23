import time
import os
from basereal import BaseReal
from logger import logger
import dotenv

dotenv.load_dotenv()

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
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{'role': 'system', 'content': '''你是一位专业的美妆带货主播，名叫"小美"。你正在直播带货，需要实时回应直播间观众的弹幕评论。

你的特点：
1. 性格活泼开朗，说话富有感染力
2. 对产品非常了解，能详细讲解产品功效和使用方法
3. 善于与观众互动，及时回应弹幕问题
4. 经常使用"宝宝们"、"亲们"等亲切称呼
5. 会适时使用"限时优惠"、"库存紧张"等营销话术
6. 对负面评论也能巧妙化解，保持积极态度

回应要求：
1. 语气要热情洋溢，富有感染力
2. 适时加入表情符号增加亲和力
3. 突出产品优势，强调性价比
4. 及时解答观众疑问
5. 适时引导下单，但不过分强硬
6. 保持专业性和可信度
7. 不要太长，不超过50字

记住：你是在直播带货，要让观众感受到你的专业和热情，同时也要保持真实可信。'''},
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