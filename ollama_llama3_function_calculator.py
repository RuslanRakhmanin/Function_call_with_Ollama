from langchain_experimental.llms.ollama_functions import OllamaFunctions, ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
import json
import re

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You should check if you need an additional information.
If you do not need an additional information, respond conversationally.
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

def calculate(expression):
    """Remove any non-digit or non-operator characters from the expression"""
    expression_modified = re.sub(r'[^0-9+\-*/().]', '', expression)
    
    try:
        # Evaluate the expression using the built-in eval() function
        result = eval(expression_modified)
        return f"The result of calculating {expression} is {result}"
    except (SyntaxError, ZeroDivisionError, NameError, TypeError, OverflowError):
        return "Error: Invalid expression"

def run_conversation(question):

    chat_model = ChatOllama(
        model="llama3:8b-instruct-q8_0", 
        format="json",
        temperature=0.0
        )
    
    model = OllamaFunctions(
        model="llama3:8b-instruct-q8_0", 
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
            },
            {
                "name": "calculator",
                "description": "A simple calculator that performs basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4')."
                        }
                    },
                    "required": ["expression"]
                }
            }        
        ],
        # function_call= {"name": "get_current_weather"},
        function_call= {"name": "calculator"},
        
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
        "calculator": calculate
    }  # only one function in this example, but you can have multiple

    need_a_second_call = False
    for tool_call in response.tool_calls:
        function_name = tool_call['name']
        function_to_call = available_functions[function_name]
        if function_to_call is None:
            print(f"Error: function {function_name} not found")
            continue
        function_args = tool_call['args']
        function_response = function_to_call(**function_args)


        # Unfortunately, langchain has only 3 types of messages for ollama: HumanMessage, AIMessage, SystemMessage
        # In terms of roles it is: user, assistant, system
        # It is better to have here a ToolMessage or a FunctionMessage

        # extend conversation with function response
        # messages.insert(0, HumanMessage(content=function_response))
        # messages.append(HumanMessage(content=f"'id': {tool_call['id']}, 'function_response': {function_response}"))
        # messages.append(HumanMessage(function_response))
        messages = [
            HumanMessage( content=function_response + ". " + question )
        ]        
        need_a_second_call = True

    # print("="*20)
    # print(messages)
    # print("="*20)
    if need_a_second_call:
        response2 = chat_model.invoke(messages)
    else:
        response2 = response

    # print(response2)

    return response2, messages
    # content=''
    #  id='run-4cc432bd-30d5-45f9-b6b8-e91623e720c6-0'
    #  tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore'},
    #  'id': 'call_e3f7c770d95142cca6bc48f1f4933a72'}]

    # It doesn't work because of the langchain prompt to ollama like:
    #  "You must always select one of the above tools and respond with only a JSON object matching the following schema"
    # Every time ollama returns a function call.




# With this prompt, the model gives answer like:
# content="According to my knowledge, Singapore's weather is typically hot and humid 
# throughout the year. The temperature you provided, 26 degrees Celsius, is quite 
# common for this tropical city-state. However, it's always a good idea to check
#  current weather conditions before planning your activities."

# response, messages = run_conversation('what is the weather in Singapore?')


# With this prompt, the model gives answer like:
# content='' id='run-35841776-b47e-42e2-a781-c473a3a2b9e5-0' 
# tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Singapore'},
#  'id': 'call_6574337642184757a42378511917a83a'}]

# response, messages = run_conversation('what is the weather in Singapore? Give a short answer.')


# With this prompt, the model gives answer like:
# content="It's warm and sunny in Singapore, with a temperature of 26 degrees Celsius!"

# response, messages = run_conversation(" Give a short answer. what is the weather in Singapore?")


response, messages = run_conversation("What is the result of 1,984,135 * 9,343,116?")


print(messages)
print(response)