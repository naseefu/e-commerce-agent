from dotenv import load_dotenv
from langsmith import traceable
from groq import Groq
import json
import re
import inspect

from tenacity import stop_never

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
    return round(float(price) * (1 - discount / 100), 2)


tools_dict = {"get_product_price": get_product_price, "apply_discount": apply_discount}

# CHANGE 3 : Delete the JSON schemas, Tools now live inside the prompt as plain text
# We derive descriptions from the functions themselves using inspect


def get_tool_description(tools_dic):
    description = []

    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable add *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        description.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(description)


TOOLS_DESCRIPTION = get_tool_description(tools_dict)
TOOLS_NAMES = ", ".join(tools_dict.keys())


# react prompt

react_prompt = f"""
    STRICT RULES — you must follow these exactly:
    1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
    2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
    3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
    4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.
    
    Answer the following questions as best you can. You have access to the following tools:
    
    {TOOLS_DESCRIPTION}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{TOOLS_NAMES}]
    Action Input: the input to the action, as comma separated values
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {{question}}
    Thought:
"""



@traceable(name="Groq Chat", run_type="llm")
def groq_chat_traced(model, messages, stop):
    return client.chat.completions.create(
        model=model, messages=messages, temperature=TEMPERATURE, stop=stop
    )


# ------- Agent Loop --------


@traceable(name="Groq Agent Loop")
def run_agent(question: str):

    print(f"Question : {question}")
    print("=" * 60)

    # Change: one prompt string replaces the system/user message split
    prompt = react_prompt.format(question=question)
    scratchpad = "" # stores the history what llm did so far

    # --------- iteration starts here ----------------

    for iteration in range(1, MAX_ITERATION + 1):

        print(f"\n----- Iteration {iteration} ------\n")
        full_prompt = prompt + scratchpad
        response = groq_chat_traced(model=MODEL,messages=[{"role":"user","content":full_prompt}], stop="\nObservation")

        output = response.choices[0].message.content

        print(f" [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "="*60)
            print(f"Final Answer : {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        # if no tool calls, this is the final answer
        if not action_match or not action_input_match:
            print(
                " [Parsing] Error: Could not parse Action/Action input from LLM outpu"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"   [Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        tool_args = [x.split("=", 1)[-1].strip().strip("'\'") for x in raw_args]

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference: Direct function call instead of tool.invoke()
        observation = tool_to_use(*tool_args)
        print(f"   [Tool Result] {observation}")

        scratchpad+=f"{output}\nObservation: {observation}\nThought:"

    print("ERROR : Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    while True:
        result = run_agent(input("Please Enter Message : "))
