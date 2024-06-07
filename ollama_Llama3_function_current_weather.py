from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage, ToolMessage
import json


# I tried to make a second AL call with function results but failed with getting content answer.
# It doesn't work because of the langchain prompt to ollama like:
#  "You must always select one of the above tools and respond with only a JSON object matching the following schema"
# Every time ollama returns a function call. So decided to use another prompt.

# Surprisingly, the word "Respond" should be written with the first letter capitalized.
# Otherwise AI returns a function call for the second invoke.
# It is so unreliable and random because of one letter in the prompt!

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must allways check if you need an additional information.
If you do not need an additional information, Respond conversationally.
If you need an additional information select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    elif "singapore" in location.lower():
        return json.dumps({"location": "Singapore", "temperature": "26", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation(question):
    """
    Runs a conversation with the given question.

    Args:
        question (str): The question to ask.

    Returns:
        Tuple[Any, List[HumanMessage]]: A tuple containing the response to the question 
        and the list of messages in the conversation.
    """
    model = OllamaFunctions(
        model="llama3:8b-instruct-q8_0",
        # model="llama3:8b-instruct-q5_K_S",
        format="json",
        temperature=0.0
        )

    model.tool_system_prompt_template = DEFAULT_SYSTEM_TEMPLATE

    model = model.bind_tools(
        tools=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, " "e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call={"name": "get_current_weather"},

    )


    messages = [
        HumanMessage( content=question )
    ]
    response = model.invoke(messages)

    print("First AI response:")
    print(response)
    # content=''
    #  id='run-71f650a7-7557-4db2-b781-939c6db6bb99-0'
    #  tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore', 'unit': 'celsius'},
    #  'id': 'call_cd09480568f349c5b1f2bdb689137198'}]

    # messages.append(response)

    available_functions = {
        "get_current_weather": get_current_weather,
    }  # only one function in this example, but you can have multiple

    need_a_second_call = False
    for tool_call in response.tool_calls:
        function_name = tool_call['name']
        function_to_call = available_functions[function_name]
        function_args = tool_call['args']
        function_response = function_to_call(
            location=function_args["location"],
            unit=function_args["unit"],
        )
        # Unfortunately, langchain has only 3 types of messages for ollama: HumanMessage, AIMessage, SystemMessage
        # In terms of roles it is: user, assistant, system
        # It is better to have here a ToolMessage or a FunctionMessage


        # extend conversation with function response

        # This approach to give the function calling id and response to ollama/Llama3 is not working.
        # messages.append(HumanMessage(json.dumps({"tool_call_id":tool_call['id'], "content":function_response})))

        # Adding the function response to the end of the conversation doesn't work. So I add it to the beginning
        messages.insert(0, HumanMessage(content=function_response))
        need_a_second_call = True

    # print("="*20)
    # print(messages)
    # print("="*20)
    if need_a_second_call:
        response2 = model.invoke(messages)
    else:
        response2 = response

    # print(response2)

    return response2, messages


# Tryes for model="llama3:8b-instruct-q8_0


# With this prompt, the model gives answer like:
# content="According to my knowledge, Singapore's weather is typically hot and humid 
# throughout the year. The temperature you provided, 26 degrees Celsius, is quite 
# common for this tropical city-state. However, it's always a good idea to check
#  current weather conditions before planning your activities."

response, messages = run_conversation('what is the weather in Singapore?')
print(messages)
print(response)

# With this prompt, the model gives answer like:
# content='' id='run-35841776-b47e-42e2-a781-c473a3a2b9e5-0' 
# tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore'},
#  'id': 'call_6574337642184757a42378511917a83a'}]

response, messages = run_conversation('what is the weather in Singapore? Give a short answer.')
print(messages)
print(response)

# With this prompt, the model gives answer like:
# content="It's warm and sunny in Singapore, with a temperature of 26 degrees Celsius!"

response, messages = run_conversation(" Give a short answer. what is the weather in Singapore?")
print(messages)
print(response)


# Tryes for model="llama3:8b-instruct-q5_K_S"
# Only last prompt works correctly. " Give a short answer. what is the weather in Singapore?"

# First AI response:
# content='' id='run-6ecb93d2-8668-43a4-ac4b-39550f6c1475-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore', 'unit': 'celsius'}, 'id': 'call_8ba0895ba0ad4460a595a84c44fa60a3'}]
# [HumanMessage(content='{"location": "Singapore", "temperature": "26", "unit": "celsius"}'), HumanMessage(content='what is the weather in Singapore?')]
# content='' id='run-b0a0a104-72f1-4677-ae9b-833e000084e4-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore'}, 'id': 'call_adc095d6ef464c1791d0d4278f9d09c5'}]

# First AI response:
# content='' id='run-feca99af-84e2-4ded-9f8f-008282c465c5-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore', 'unit': 'celsius'}, 'id': 'call_49fa105d3d044a38b2aab8b6e9ccf64c'}]
# [HumanMessage(content='{"location": "Singapore", "temperature": "26", "unit": "celsius"}'), HumanMessage(content='what is the weather in Singapore? Give a short answer.')]
# content='' id='run-45cf51bf-8027-4e71-844e-8f6e1a9a3ba8-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore'}, 'id': 'call_7aaed2ecac2f40039216f167ccbc27f5'}]

# First AI response:
# content='' id='run-618d9e59-c2f6-4f67-b5a5-5c3d79bbcbfc-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore', 'unit': 'celsius'}, 'id': 'call_c9464466a7144252aff9c9059867ad27'}]
# [HumanMessage(content='{"location": "Singapore", "temperature": "26", "unit": "celsius"}'), HumanMessage(content=' Give a short answer. what is the weather in Singapore?')]
# content="It's warm and sunny in Singapore, with a temperature of 26 degrees Celsius!" id='run-b7f21fae-351f-4da3-aed1-3e6a3b168ba6-0'
