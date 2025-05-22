import logging
 
# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
fhandler = logging.FileHandler('livetalking.log')  # 可以改为StreamHandler输出到控制台或多个Handler组合使用等。
# 控制台也要输出
shandler = logging.StreamHandler()
shandler.setFormatter(formatter)
shandler.setLevel(logging.INFO)
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)
logger.addHandler(shandler)
