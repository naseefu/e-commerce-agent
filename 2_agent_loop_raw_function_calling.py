from dotenv import load_dotenv
from langsmith import traceable
from groq import Groq
import json

load_dotenv()

# -------- Groq Client -----------

client = Groq()

# ------- Model Details -----------

MAX_ITERATION = 10
MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0

# ------ Data of our products ----------

prices = {"laptop": 1299.99, "headphone": 149.95, "keyboard": 89.50}
discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}


# ------ Tools (Langchain @tool decorator) -------


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"   >> Executing get_product_price(product = '{product}')")
    return prices.get(product, 0)  # if no product found, returns 0 ( default )


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"   >> Executing apply_discount(price={price}, discount_tier={discount_tier})"
    )
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, eg: 'laptop','headphones', 'keyboard'",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

tools_dict = {"get_product_price": get_product_price, "apply_discount": apply_discount}


# NOTE: Ollama can also auto generate these schemas if you pass the functions
# directly as tools ( similar to LangChain's @tool decorator )
# tools_for_llm = [get_product_details, apply_discount]
# However, this requires your docstring to follow the Google docstring format
# so ollama can parse parameter description from the Args section, for example:
# def get_product_details(product:str) -> float:
#   """Look up the price of a product in the catalog
#
#      Args:
#          product: The product name, e.g. 'laptop','headphones', 'keyboard'
#
#      Returns:
#          The price of the product, or 0 if not found
#   """
# we keep the manual JSON version here so you can see what @tool hides from you


# --------- Helper: traced Groq call --------
# Difference 3: without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Groq Chat", run_type="llm")
def groq_chat_traced(messages):
    return client.chat.completions.create(
        model=MODEL, tools=tools_for_llm, messages=messages, temperature=TEMPERATURE
    )


# ------- Agent Loop --------


@traceable(name="Groq Agent Loop")
def run_agent(question: str):

    print(f"Question : {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price. "
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    # --------- iteration starts here ----------------

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n----- Iteration {iteration} ------\n")
        response = groq_chat_traced(messages=messages)
        ai_message = response.choices[0].message

        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}\n\n")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_id = tool_call.id
        tool_args = json.loads(tool_call.function.arguments)

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)
        print(f"   [Tool Result] {observation}")

        messages.append(
            {"role": "tool", "content": str(observation), "tool_call_id": tool_id}
        )
    print("ERROR : Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    while True:
        result = run_agent(input("Please Enter Message : "))
