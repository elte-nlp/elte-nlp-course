# Accessing the University LLM Inference Endpoint

We currently host a single open model endpoint that could be used to student projects, experiments, practices for students enrolled in the course. Please use the endpoint responsibly as it is a shared resource, and keep in mind that we only keep the endpoint running at a given version during the project submission period, and it may be updated or removed after the course ends.

> **Acknowledgement**
> 
> The endpoint is hosted by the Digital Heritage National Laboratory, who kindly provide some usage quota for our course as well. We are grateful for their support.

## Endpoint Information

| **Field**       | **Description**                                      |
|-----------------|------------------------------------------------------|
| API Endpoint URL    | `http://mobydick.elte-dh.hu:12321`        |
| Endpoint API definition   | [http://mobydick.elte-dh.hu:12321/docs](http://mobydick.elte-dh.hu:12321/docs)       |
| Authentication  | API Key (contact the lecturers)              |
| Model Version (might be out of date)   | [casperhansen/llama-3.3-70b-instruct-awq](https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq)                                          |
| vLLM Server documentation   | [https://docs.vllm.ai/en/latest/index.html](https://docs.vllm.ai/en/latest/index.html) |
| Test GUI    | [http://mobydick.elte-dh.hu:3000](http://mobydick.elte-dh.hu:3000)      |
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
    base_url="http://mobydick.elte-dh.hu:12321/v1",
    api_key="<API_KEY>"
)

chat_completion = client.chat.completions.create(
    model="casperhansen/llama-3.3-70b-instruct-awq",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is the purpose of life"}
    ],
    stream=False
)

print(chat_completion)
```

For the most up to date information, please refer to the official vLLM tutorials: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#supported-apis](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#supported-apis)

#### Limitations

Visual inputs are not supported, and the context window is limited to ~90k tokens (as a sum of prompt and response). There is no maximal response length. 

Please check vLLM OpenAI Server documentation for parameter details (there are known issues with top_p, etc...), also the default for the max generated tokens tends to be a low number (1-200), which you have to override in your requests!

For tokenization or token counting you should use the HTTP endpoints or after getting the model id (by HTTP endpoints again) you can instantiate your own tokenizer from Huggingface objects. Some models might be gated so you need a huggingface access token to download their tokenizer.


## Test GUI

The test GUI is currently very limited, you can only set the model, system prompt, temperature and max output tokens. Feel free to create your own test GUI if needed, `gradio` and `streamlit` are both good, swift options for that.

## Warning

The endpoint is for educational / experimental purposes only, it is not meant for production use or handling sensitive or personal data. Commercial use, or any unauthorized usage out of class is unsafe and strictly forbidden.
We also occasionally monitor and log the requests and responses to avoid misuse.
