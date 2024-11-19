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
| Model Version (might be out of date)   | `LLama3.1-70B 4bit AWQ quantized`                                               |
| TGI Server documentation   | [https://huggingface.co/docs/text-generation-inference/index](https://huggingface.co/docs/text-generation-inference/index) |
| Test GUI    | [http://mobydick.elte-dh.hu:3000](http://mobydick.elte-dh.hu:3000)      |
| Test GUI login  | Ask the lecturers for the login credentials. |

## Usage

### OpenAI package

You can use the `openai` package to interact with the endpoint.
The endpoint is mostly compatible with the core OpenAI functionality with slight limitations.

To use the endpoint set the following parameters:

```python
import openai

openai.api_key = "<API_KEY>" # STEP1 set the API key
openai.api_base = "http://mobydick.elte-dh.hu:12321/v1" # STEP2 use the /v1 endpoint

response = openai.chat.completions.create(
    model="tgi", # STEP3 set the model to "tgi"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the purpose of life?"},
    ],
)
```

For the most up to date information, please refer to the official TGI tutorial: [https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi#openai-client](https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi#openai-client)

#### Limitations

Visual inputs are not supported, and the context window is limited to ~20k tokens (as a sum of prompt and response). There is no maximal response length. 

Please check TGI documentation for parameter details (there are known issues with top_p, etc...)

Tooling is currently in limited support (automatic selection works fine, but forced selection might not work as expected), and there is no json output support yet. 

For enforcing output schema you can instantiate your own tokenizer and use huggingface's `generate` with an inference client, where you are able to provide guidance grammar to enforce the output schema. [Details](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance#hugging-face-hub-python-library)

For tokenization or token counting you should use the HTTP endpoints or after getting the model id (by HTTP endpoints again) you can instantiate your own tokenizer from Huggingface objects. Some models might be gated so you need a huggingface access token to download their tokenizer.

### HTTP Requests

You can also use HTTP requests to interact with the endpoint. In this case the API key should be passed as a Bearer token:

```python
import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer <API_KEY>"
}

data = {
    'inputs': 'What is the purpose of life?',
    'parameters': {
        'max_new_tokens': 20,
    },
}
response = requests.post('http://127.0.0.1:8080/generate', headers=headers, json=data)
```

Some functionalities are not available via the OpenAI package (nor the official Huggingface Hub Client), such as listing the available model information, or performing tokenizations, health checks, etc. Check the API documentation for more details.

## Test GUI

The test GUI is currently very limited, you can only set the model, system prompt, temperature and max output tokens. Feel free to create your own test GUI if needed, `gradio` and `streamlit` are both good, swift options for that.

## Warning

The endpoint is for educational / experimental purposes only, it is not meant for production use or handling sensitive or personal data. Commercial use, or any unauthorized usage out of class is unsafe and strictly forbidden.
We also occasionally monitor and log the requests and responses to avoid misuse.
