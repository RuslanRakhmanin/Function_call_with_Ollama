# Function Calling with Local Models & LangChain - Ollama, Llama3

## Introduction

In this repo I tried to implement function calling examples with Ollama and Llama3.
As I found in the process, Ollama does not support function calling natively.
At least it did not in the version 0.1.41.
I started with the [video by Sam Witteveen](https://youtu.be/Ss_GdU0KqE0),
where he demonstrated how to implement function calling with Ollama and LangChain.
In this video Sam uses the LangChain Experimental library to implement function calling generated by Ollama.
Unfortunately, this example covers only the step where Ollama requests a function call.
There is no response to Ollama and step after when Ollama generates a response with additional data from the function call.
So, this implementation of function calling is not as complete as OpenAI documentation shows in the [example](https://platform.openai.com/docs/guides/function-calling).
I tried to make it as complete as possible.

## Versions

- Ollama: 0.1.41
- Llama3: llama3:8b-instruct-q8_0
- LangChain Experimental 0.0.60

## Try 1. Current weather in a given location

You can find the code [here](ollama_Llama3_function_current_weather.py).

I took the code from the [video by Sam Witteveen](https://youtu.be/Ss_GdU0KqE0) as a starting point. You can find the original file [here](https://github.com/samwit/agent_tutorials/blob/main/ollama_agents/llama3_local/llama3_ollama_functions.py) or a local copy [here](Examples/llama3_ollama_functions.py).

This is what was shown in the video by Sam:

[![](https://mermaid.ink/img/pako:eNpNjkFrwzAMhf-K0dktS5eSzYfCYNeyQ2_FF2Frmaktp54Na0P--9yEjL2T9PQ99EYw0RIo-KZrITb07rBPGDSLqreeOIvN4SA-vMeASsTkesfoxZBiGPKCLUexeYBzRInPwia7yMKg9yAhUArobH00PjIa8hcF0qDqaDFdNGieKoclx9ONDaicCkkog8W8llpNsi7HdFyKz_0lDMjnGP-QuoIa4QdUs2sl3EC97rbPT__UTBLuc6LZrta-a9qXru2mXyBbVxs?type=png)](https://mermaid.live/edit#pako:eNpNjkFrwzAMhf-K0dktS5eSzYfCYNeyQ2_FF2Frmaktp54Na0P--9yEjL2T9PQ99EYw0RIo-KZrITb07rBPGDSLqreeOIvN4SA-vMeASsTkesfoxZBiGPKCLUexeYBzRInPwia7yMKg9yAhUArobH00PjIa8hcF0qDqaDFdNGieKoclx9ONDaicCkkog8W8llpNsi7HdFyKz_0lDMjnGP-QuoIa4QdUs2sl3EC97rbPT__UTBLuc6LZrta-a9qXru2mXyBbVxs)

<!-- ```Mermaid
sequenceDiagram
    Agent ->> Ollama: original prompt
    Ollama -- >> Agent: function call
``` -->

This is what I can see in the [OpenAI documentation about function calling](https://platform.openai.com/docs/guides/function-calling):

[![](https://mermaid.ink/img/pako:eNp1ULluAjEQ_RVraoOykAYXSEhpUkQp6CI3I3vYWFmPHR9KYLX_jtkVShqmmeu9N8cIJlgCBZm-K7GhF4d9Qq9ZNDv0xEWs9nvxHokPr0qE5HrHOIiYgo9lgS1NsboBZ4oSp8qmuMBZGByGR2p_qEQ5Nk-PBeepyPmHEkjwlDw62xYfbxQN5ZM8aVAttJi-NGieGg5rCcczG1AlVZJQo8VyPxLUCYfcqmRdCelt-cT8EAkR-SMEfye2FNQIv6C6zbOEM6jdZr19-mfdJOEyM7rpCkfDbYo?type=png)](https://mermaid.live/edit#pako:eNp1ULluAjEQ_RVraoOykAYXSEhpUkQp6CI3I3vYWFmPHR9KYLX_jtkVShqmmeu9N8cIJlgCBZm-K7GhF4d9Qq9ZNDv0xEWs9nvxHokPr0qE5HrHOIiYgo9lgS1NsboBZ4oSp8qmuMBZGByGR2p_qEQ5Nk-PBeepyPmHEkjwlDw62xYfbxQN5ZM8aVAttJi-NGieGg5rCcczG1AlVZJQo8VyPxLUCYfcqmRdCelt-cT8EAkR-SMEfye2FNQIv6C6zbOEM6jdZr19-mfdJOEyM7rpCkfDbYo)

<!-- ```Mermaid
sequenceDiagram
    Agent ->> OpenAI: original prompt
    OpenAI -- >> Agent: functions call
    Agent - >> OpenAI: functions response
    OpenAI -- >> Agent: final answer
``` -->

To make the Ollama example follow the OpenAI documentation, I made some changes in the code:

- I have changed the DEFAULT_SYSTEM_TEMPLATE because the original one makes LLM always respond with a function call. It is literally said in the original prompt: `You must always select one of the above tools and respond with only a JSON object`

- The AI temperature was set to 0. It makes the LLM answers constant for the same prompt.

- Langchain has only 3 types of messages for Ollama: HumanMessage, AIMessage, SystemMessage.
 It is better to have here a ToolMessage or a FunctionMessage.
  But it is what it is.
  So the response after a function call was made like HumanMessage.
  The response was added to the top of the message history. Otherwise, LLama3 returned a function call.

Fun fact. In my prompt the word "Respond" should be written with the first letter capitalized. Otherwise, AI returns a function call for the second invoke.

The final answer depends on the original prompt. For example, the prompt `what is the weather in Singapore? Give a short answer.` confuses llama3:8b-instruct-q8_0.
 On the other hand, the prompt `Give a short answer. what is the weather in Singapore?` works well.

 Quantized version llama3:8b-instruct-q5_K_S performs even worse than llama3:8b-instruct-q8_0. It answered correctly only for 1 prompt from 3.
 It is better to use the full version of Llama3 8b.

The final conversation between Agent and Ollama.Llama3 is shown below:

[![](https://mermaid.ink/img/pako:eNqdkk1Lw0AQhv_KOJdcYrFRCgatiIIIigdvGihjMiaL2d26H8Ya-t_dbZpSLyLuIWHfmX2fmWF6LHXFmKPld8-q5GtBtSFZKAjnsmbl4HA-h4e2JUmTu_g9zqHAG_HBQGAbbRyQsh2bCXQNORAWXMPQMYWfAaHgUaialtrwRYGD8Q87OIyEDSs6O63bRUlta8-f-0SR5CSHpGa3KL0xIWmxtU5SSMjUNoT7pNUlOaFVzN3xYoZXwkWx5NYKb5P1WMMvzUm2lmqGabz0BY7mBUZhZ19gGutluWRDzkchxrPZEIjkQdmyCwxwOHsxcxgJWQzDv4f5h3HeusRCR0YG4wqsV2r1wyaFTrgm0Pf6AP0K2Qwqrg2zhauh_IMCMQ2zMZJEFVamj_DQf8Ny23lF5i3WtA555J1-XKkSc2c8p-iXFblxvTB_pdYGlSvhtLkfdnCziikuST1pLceH4Yp5j5-YT7OTFFeYn2aT46O9M12n-LV5MV1_A_3R6pw?type=png)](https://mermaid.live/edit#pako:eNqdkk1Lw0AQhv_KOJdcYrFRCgatiIIIigdvGihjMiaL2d26H8Ya-t_dbZpSLyLuIWHfmX2fmWF6LHXFmKPld8-q5GtBtSFZKAjnsmbl4HA-h4e2JUmTu_g9zqHAG_HBQGAbbRyQsh2bCXQNORAWXMPQMYWfAaHgUaialtrwRYGD8Q87OIyEDSs6O63bRUlta8-f-0SR5CSHpGa3KL0xIWmxtU5SSMjUNoT7pNUlOaFVzN3xYoZXwkWx5NYKb5P1WMMvzUm2lmqGabz0BY7mBUZhZ19gGutluWRDzkchxrPZEIjkQdmyCwxwOHsxcxgJWQzDv4f5h3HeusRCR0YG4wqsV2r1wyaFTrgm0Pf6AP0K2Qwqrg2zhauh_IMCMQ2zMZJEFVamj_DQf8Ny23lF5i3WtA555J1-XKkSc2c8p-iXFblxvTB_pdYGlSvhtLkfdnCziikuST1pLceH4Yp5j5-YT7OTFFeYn2aT46O9M12n-LV5MV1_A_3R6pw)

<!-- ```Mermaid
sequenceDiagram
    Agent ->> Ollama.Llama3: "Give a short answer. what is the weather in Singapore?"
    Ollama.Llama3 -- >> Agent: "tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore', 'unit': 'celsius'}"
    Agent ->> Ollama.Llama3: "message 1: "{"location": "Singapore", "temperature": "26", "unit": "celsius"}" <br> message 2: " Give a short answer. what is the weather in Singapore?""
    Ollama.Llama3 -- >> Agent: "It's warm and sunny in Singapore, with a temperature of 26 degrees Celsius!"
``` -->

## Try 2. Calculator tool call

The source code is available [here](ollama_llama3_function_calculator.py).

Llama3:8b-instruct-q8_0 can extract a calculation request from the prompt pretty well. It generated a ocrrect mathematical expression even when a part was defined with math characters and another part with words. Unfortunately, Ollama did not request several function calls for two math expressions in the same prompt. I will try to change the system prompt to achieve this in future tries.

The LLM did not want to see a function call results in the message history. To solve this problem I put the results in the top of the user prompt message.

[![](https://mermaid.ink/img/pako:eNrFUttKxDAQ_ZVhXqoSl43ZFRtwRRAE8fKgIGhAYju7BnOpbSqupf9uuuuCCuKj8xCSmTlzzkymwyKUhBIbemnJF3Ri9KLWTnlIdrwgH2F3NoMra7XTo_PhFBIU3j7pCKaB-ERQU9PaCGEOnOUHE8bFFHYgZ2IiGOf74FLUVNZQCY9LEEdwal4JHIGGs-urywFfBd_QSOGa9xsb7A4CVlIkxBDsQ6GtbQ7vu8xrR5mELDmK1uoY6oxBputFk7xdRm9VKt2Y4IekLZ60fUpLypKw7XQVWf9nrzffetyQGb-AX2oOg5lO9_lkzMciF_nBeAT_NDCFnVJe4ZpSofwhLAV7hcjQUe20KdMqdENNhUmoowGgsNT180DVpzzdxnC99AXKWLfEsK1KHTdrg3KubZO8VJr0Gxfr3VqtGMNK-7sQ3AaYnig7fEPJ9yYMlyjzvZEYfzHeM3xfIXj_AbXvy44?type=png)](https://mermaid.live/edit#pako:eNrFUttKxDAQ_ZVhXqoSl43ZFRtwRRAE8fKgIGhAYju7BnOpbSqupf9uuuuCCuKj8xCSmTlzzkymwyKUhBIbemnJF3Ri9KLWTnlIdrwgH2F3NoMra7XTo_PhFBIU3j7pCKaB-ERQU9PaCGEOnOUHE8bFFHYgZ2IiGOf74FLUVNZQCY9LEEdwal4JHIGGs-urywFfBd_QSOGa9xsb7A4CVlIkxBDsQ6GtbQ7vu8xrR5mELDmK1uoY6oxBputFk7xdRm9VKt2Y4IekLZ60fUpLypKw7XQVWf9nrzffetyQGb-AX2oOg5lO9_lkzMciF_nBeAT_NDCFnVJe4ZpSofwhLAV7hcjQUe20KdMqdENNhUmoowGgsNT180DVpzzdxnC99AXKWLfEsK1KHTdrg3KubZO8VJr0Gxfr3VqtGMNK-7sQ3AaYnig7fEPJ9yYMlyjzvZEYfzHeM3xfIXj_AbXvy44)

<!-- ```Mermaid
sequenceDiagram
    Agent ->> Ollama.Llama3: "What is the result of 1,984,135 * 9,343,116 multiplied by 3? Give me a JSON response."
    Ollama.Llama3 -- >> Agent: tool_calls=[{'name': 'calculator', 'args': {'expression': '(1984135 * 9343116) * 3'}
    Agent ->> Ollama.Llama3: "The result of calculating (1984135 * 9343116) * 3 is 55614010393980. What is the result of 1,984,135 * 9,343,116 multiplied by 3? Give me a JSON response."
    Ollama.Llama3 -- >> Agent: "{\n"result": 55614010393980\n}"
``` -->

## Try 3. Multiple functions call

```text
TODO
```

## Links

### Examples of function calling with Ollama and LangChain

Video by Sam Witteveen: [Function Calling with Local Models & LangChain - Ollama, Llama3 & Phi-3](https://youtu.be/Ss_GdU0KqE0)

Repo for the video: https://github.com/samwit/agent_tutorials/tree/main/ollama_agents/llama3_local

Copies of the source code for the video are available [here](Examples).

### Function Calling in Ollama with raw requests

Video by Matt Williams: [Function Calling in Ollama vs OpenAI](https://youtu.be/RXDWkiuXtG0)

In this video Matt focuses on getting structured output from Ollama in JSON.
He ignores the OpenAI API capabilities to ask the user side for additional information as a function call and use this information for the answer generation.
Maybe OpenAI documentation did not cover it well at that time.

The source code for this video is available [here](Examples/fc.py).

### Function calling with OpenAI API

[OpenAI documentation. Function calling](https://platform.openai.com/docs/guides/function-calling)

You can find the source code example from this documentation [here](Examples/openAI_multiple_function_calls.py).

### Tools usage (function calling) with Anthropic

[Tool use (function calling)](https://docs.anthropic.com/en/docs/tool-use)

[Tool use examples](https://docs.anthropic.com/en/docs/tool-use-examples)

[Example. Using a Calculator Tool with Claude](https://github.com/anthropics/anthropic-cookbook/blob/main/tool%5Fuse/calculator%5Ftool.ipynb)
