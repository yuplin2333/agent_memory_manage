import openai
# import together
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    # OpenAI API
    ## return openai.ChatCompletion.create(**kwargs)
    client = openai.OpenAI(api_key=openai.api_key)
    return client.chat.completions.create(**kwargs)

    # Together API
    # client = together.Together(api_key=together.api_key)
    # response = client.chat.completions.create(**kwargs)
    # response = response.model_dump()
    # return response

    # DeepSeek API
    ## return openai.ChatCompletion.create(base_url="https://api.siliconflow.cn/v1/", **kwargs)
    # client = openai.OpenAI(base_url="http://38.92.25.168:3000/v1", api_key=openai.api_key)
    # return client.chat.completions.create(**kwargs)