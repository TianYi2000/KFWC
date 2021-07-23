import requests
api_key = "a03f079569c5423fb8eca7be41f8dda5" #微信通知记录
def message(title, body):
    url = 'http://www.pushplus.plus/send?token='+api_key+'&title='+title+'&content='+body
    requests.get(url)