import json
import re
from langchain_experimental.llms.ollama_functions import OllamaFunctions, ChatOllama
from langchain_core.messages import HumanMessage


DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You should check if you need an additional information.
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
    """
    Runs a conversation with the given question.

    Args:
        question (str): The question to ask.

    Returns:
        Tuple[Any, List[HumanMessage]]: A tuple containing the response to the question 
        and the list of messages in the conversation.
    """
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

    # messages.append(response)

    available_functions = {
        "get_current_weather": get_current_weather,
        "calculator": calculate
    } 

    need_a_second_call = False
    content = [question]
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
        content.insert(0, function_response)
        need_a_second_call = True

    # print("="*20)
    # print(messages)
    # print("="*20)
    if need_a_second_call:
        # content.append("Give a short answer.")
        messages = [
            HumanMessage( content=". ".join(content) )
        ]
        response2 = chat_model.invoke(messages)
        # response2 = model.invoke(messages)
    else:
        response2 = response

    # print(response2)

    return response2, messages



if __name__ == "__main__":

    response, messages = run_conversation("What is the result of 1,984,135 * 9,343,116?")
    print(messages)
    print(response)

    response, messages = run_conversation("What is the result of 1,984,135 * 9,343,116 multiplied by 3? Give me a JSON response.")
    print(messages)
    print(response)

    response, messages = run_conversation("What is the result of 1,984,135 * 9,343,116 multiplied by 3? What is the result of 2**8? Give me a JSON response.")
        #content='{\n"result": {\n"calculation": "1984135 * 9343116 * 3",\n"value": 55614010393980\n},\n"power_of_two": {\n"base": 2,\n"exponent": 8,\n"value": 256\n}\n}' response_metadata={'model': 'llama3:8b-instruct-q8_0', 'created_at': '2024-06-09T15:00:05.1080377Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 14563812700, 'load_duration': 1788800, 'prompt_eval_count': 72, 'prompt_eval_duration': 1071184000, 'eval_count': 59, 'eval_duration': 13489625000} id='run-1cff0b74-a1c0-431f-9b08-cfb060245d0e-0'
    print(messages)
    print(response)