# Accessing the University LLM Inference Endpoint

We currently host a single open model endpoint that could be used to student projects, experiments, practices for students enrolled in the course. Please use the endpoint responsibly as it is a shared resource, and keep in mind that we only keep the endpoint running at a given version during the project submission period, and it may be updated or removed after the course ends.

> **Acknowledgement**
> 
> The endpoint is jointly provided by the ELTE Faculty of Informatics and the Digital Heritage National Laboratory.

## VPN

In order to access the endpoint you need to connect to an ELTE network, or use the ELTE VPN service as detailed here:
https://iig.elte.hu/en/content/vpn-settings.t.16226


## Endpoint Information

| **Field**       | **Description**                                      |
|-----------------|------------------------------------------------------|
| API Endpoint URL    | `http://mobydick.elte-dh.hu:23432`        |
| Endpoint API definition   | [http://mobydick.elte-dh.hu:23432/docs](http://mobydick.elte-dh.hu:23432/docs)       |
| Authentication  | API Key (contact the lecturers)              |
| Model Version   | [[https://huggingface.co/Qwen/Qwen3-32B-AWQ](https://huggingface.co/zai-org/GLM-4.5-Air-FP8)]([https://huggingface.co/Qwen/Qwen3-32B-AWQ](https://huggingface.co/zai-org/GLM-4.5-Air-FP8))                                    |
| vLLM Server documentation   | [https://docs.vllm.ai/en/latest/index.html](https://docs.vllm.ai/en/latest/index.html) |
| Test GUI    | [http://mobydick.elte-dh.hu:7899](http://mobydick.elte-dh.hu:7899)      |
| Test GUI login  | Ask the lecturers for the login credentials. |

## Usage

### OpenAI package

You can use the `openai` package to interact with the endpoint.
The endpoint is mostly compatible with the core OpenAI functionality with slight limitations.

To use the endpoint set the following parameters:

```python
from openai import OpenAI

# Set the API key and the base URL (with the /v1 endpoint)
client = OpenAI(
    base_url="http://mobydick.elte-dh.hu:23432/v1",
    api_key="<API_KEY>"
)

chat_completion = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is the purpose of life"}
    ],
    stream=False
)

print(chat_completion)
```

For the most up to date information, please refer to the official vLLM tutorials: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#supported-apis](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#supported-apis)

#### Structured outputs

OpenAI's structured output schema works pretty well. Instruct the model to output a JSON and provide the following parameters to your call:

```python
resp = client.chat.completions.create(
    model = "zai-org/GLM-4.5-Air-FP8",
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You speak Hungarian."
        },
        {
            "role": "user",
            "content": "Give me a pizza recipe."
        }
    ],
    response_format={"type": "json_schema",
                     "json_schema":{
                         "name": "pizza_schema",
                         "schema": {
                             "type": "object",
                             "properties": {
                                 "name": {
                                     "type": "string",
                                     "description": "The name of the pizza"
                                 },
                                 "ingredients": {
                                     "type": "array",
                                     "items": {
                                         "type": "string"
                                     },
                                     "description": "List of ingredients"
                                 },
                                 "instructions": {
                                     "type": "string",
                                     "description": "Cooking instructions"
                                 }
                             },
                             "required": ["name", "ingredients", "instructions"]
                        
                            }
                     }},
    temperature = 0.7,
    max_tokens = 2048,
    extra_body={
    "top_k": 40,
    "chat_template_kwargs": {"enable_thinking": True},
    },
)
```



#### Reasoning

This is a reasoning model. You can turn off reasoning mode by setting `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
Reasoning effort can be set by the `reasoning_effort` in the `extra_body`. Accepted values are: `high`, `medium`, `low`.

Warning: This way the structured output will not work and all the content will be put into the `reasoning_content` field.

Streaming is supported, delta chunks are received as usual for both the content and reasoning parts.

Check the official model documentation for the best generation parameters. Some parameters such as top_k can be supplied to the API call as an `extra_body` parameter which serves as an extension to the more limited OpenAI API.

#### Limitations

Visual inputs are not supported, and the context window is limited to ~128k tokens (as a sum of prompt and response). There is no maximal response length. 

Please check vLLM OpenAI Server documentation for parameter details (there are known issues with top_p, etc...), also the default for the max generated tokens tends to be a low number (1-200), which you have to override in your requests!

For tokenization or token counting you should use the HTTP endpoints or after getting the model id (by HTTP endpoints again) you can instantiate your own tokenizer from Huggingface objects. Some models might be gated so you need a huggingface access token to download their tokenizer.


## Test GUI

The test GUI is currently very limited, you can only set the model, system prompt, temperature and max output tokens. Feel free to create your own test GUI if needed, `gradio` and `streamlit` are both good, swift options for that.

## Warning

The endpoint is for educational / experimental purposes only, it is not meant for production use or handling sensitive or personal data. Commercial use, or any unauthorized usage out of class is unsafe and strictly forbidden.
We also occasionally monitor and log the requests and responses to avoid misuse.
