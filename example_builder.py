"""A simple example of using the PlanBuilderV2."""

from dotenv import load_dotenv
from pydantic import BaseModel

from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.reference import Input, StepOutput
from portia.cli import CLIExecutionHooks
from portia.portia import Portia

load_dotenv()


class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity."""

    price: float
    currency: str


class FinalOutput(BaseModel):
    """Final output of the plan."""

    poem: str
    email_address: str


portia = Portia(execution_hooks=CLIExecutionHooks())

plan = (
    PlanBuilderV2("Write a poem about the price of gold")
    .input(name="purchase_quantity", description="The quantity of gold to purchase in ounces")
    .input(name="currency", description="The currency to purchase the gold in", default_value="GBP")
    .invoke_tool_step(
        step_name="Search gold price",
        tool="search_tool",
        args={
            "search_query": f"What is the price of gold per ounce in {Input('currency')}?",
        },
        output_schema=CommodityPriceWithCurrency,
    )
    .function_step(
        function=lambda price_with_currency, purchase_quantity: (
            price_with_currency.price * purchase_quantity
        ),
        args={
            "price_with_currency": StepOutput("Search gold price"),
            "purchase_quantity": Input("purchase_quantity"),
        },
    )
    .llm_step(
        task="Write a poem about the current price of gold",
        inputs=[StepOutput(0), Input("currency")],
    )
    .single_tool_agent_step(
        task="Send the poem to Robbie in an email at not_an_email@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput(2)],
    )
    .final_output(
        output_schema=FinalOutput,
    )
    .build()
)

plan_run = portia.run_plan(plan, plan_run_inputs={"purchase_quantity": 100})
print(plan_run)  # noqa: T201
