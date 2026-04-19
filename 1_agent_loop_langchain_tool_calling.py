from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

load_dotenv()

# ------- Model Details -----------

MAX_ITERATION = 10
MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0

# ------ Data of our products ----------

prices = {"laptop": 1299.99, "headphone": 149.95, "keyboard": 89.50}
discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}


# ------ Tools (Langchain @tool decorator) -------


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"   >> Executing get_product_price(product = '{product}')")
    return prices.get(product, 0)  # if no product found, returns 0 ( default )


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"   >> Executing apply_discount(price={price}, discount_tier={discount_tier})"
    )
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


ALL_TOOLS = [get_product_price, apply_discount]
TOOLS_DICT = {t.name: t for t in ALL_TOOLS}


# ------ Agent Loop -------

llm = init_chat_model(f"groq:{MODEL}", temperature=TEMPERATURE)
# we only need to change groq to openai or anything for switching models
llm_with_tools = llm.bind_tools(ALL_TOOLS)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print(f"Question : {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    # --------- iteration starts here ----------------

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n----- Iteration {iteration} ------\n")
        ai_message = llm_with_tools.invoke(messages)

        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}\n\n")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = TOOLS_DICT.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"   [Tool Result] {observation}")

        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR : Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    while True:
        result = run_agent(input("Please Enter Message : "))
