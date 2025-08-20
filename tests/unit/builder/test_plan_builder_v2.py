"""Test the PlanBuilderV2 class."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import Input, StepOutput
from portia.builder.step_v2 import FunctionStep, InvokeToolStep, LLMStep, SingleToolAgentStep
from portia.tool import Tool


class OutputSchema(BaseModel):
    """Output schema for testing."""

    result: str
    count: int


def example_function_for_testing(x: int, y: str) -> str:
    """Example function for function call tests."""  # noqa: D401
    return f"{y}: {x}"


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self) -> None:
        """Initialize mock tool."""
        super().__init__(
            id="mock_tool",
            name="Mock Tool",
            description="A mock tool for testing",
            output_schema=("str", "Mock result string"),
        )

    def run(self, ctx: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG002
        """Run the mock tool."""
        return "mock result"


class TestPlanBuilderV2:
    """Test cases for PlanBuilderV2."""

    def test_initialization_default_label(self) -> None:
        """Test PlanBuilderV2 initialization with default label."""
        builder = PlanBuilderV2()

        assert isinstance(builder.plan, PlanV2)
        assert builder.plan.label == "Run the plan built with the Plan Builder"
        assert builder.plan.steps == []
        assert builder.plan.plan_inputs == []
        assert builder.plan.summarize is False
        assert builder.plan.final_output_schema is None

    def test_initialization_custom_label(self) -> None:
        """Test PlanBuilderV2 initialization with custom label."""
        custom_label = "Custom Plan Label"
        builder = PlanBuilderV2(label=custom_label)

        assert builder.plan.label == custom_label

    def test_input_method(self) -> None:
        """Test the input() method for adding plan inputs."""
        builder = PlanBuilderV2()

        # Test adding input with name only
        result = builder.input(name="user_name")

        assert result is builder  # Should return self for chaining
        assert len(builder.plan.plan_inputs) == 1
        assert builder.plan.plan_inputs[0].name == "user_name"
        assert builder.plan.plan_inputs[0].description is None
        assert builder.plan.plan_inputs[0].value is None

    def test_input_method_with_description(self) -> None:
        """Test the input() method with description."""
        builder = PlanBuilderV2()

        builder.input(name="user_name", description="The name of the user")

        assert len(builder.plan.plan_inputs) == 1
        assert builder.plan.plan_inputs[0].name == "user_name"
        assert builder.plan.plan_inputs[0].description == "The name of the user"
        assert builder.plan.plan_inputs[0].value is None

    def test_input_method_multiple_inputs(self) -> None:
        """Test adding multiple inputs."""
        builder = PlanBuilderV2()

        builder.input(name="name", description="User name").input(
            name="age", description="User age"
        )

        assert len(builder.plan.plan_inputs) == 2
        assert builder.plan.plan_inputs[0].name == "name"
        assert builder.plan.plan_inputs[1].name == "age"
        assert builder.plan.plan_inputs[0].value is None
        assert builder.plan.plan_inputs[1].value is None

    def test_input_method_with_default_value(self) -> None:
        """Test the input() method with default value."""
        builder = PlanBuilderV2()

        builder.input(
            name="user_name", description="The name of the user", default_value="John Doe"
        )

        assert len(builder.plan.plan_inputs) == 1
        assert builder.plan.plan_inputs[0].name == "user_name"
        assert builder.plan.plan_inputs[0].description == "The name of the user"
        assert builder.plan.plan_inputs[0].value == "John Doe"

    def test_input_method_with_various_default_values(self) -> None:
        """Test the input() method with various types of default values."""
        builder = PlanBuilderV2()

        # Test with different types of default values
        default_bool = True
        builder.input(
            name="string_input", description="A string input", default_value="default_string"
        )
        builder.input(name="int_input", description="An integer input", default_value=42)
        builder.input(name="bool_input", description="A boolean input", default_value=default_bool)
        builder.input(
            name="list_input", description="A list input", default_value=["item1", "item2"]
        )
        builder.input(name="dict_input", description="A dict input", default_value={"key": "value"})
        builder.input(
            name="none_input", description="An input with explicit None", default_value=None
        )

        assert len(builder.plan.plan_inputs) == 6

        # Check each input's default value
        inputs = {inp.name: inp for inp in builder.plan.plan_inputs}
        assert inputs["string_input"].value == "default_string"
        assert inputs["int_input"].value == 42
        assert inputs["bool_input"].value is True
        assert inputs["list_input"].value == ["item1", "item2"]
        assert inputs["dict_input"].value == {"key": "value"}
        assert inputs["none_input"].value is None

    def test_llm_step_method_basic(self) -> None:
        """Test the llm_step() method with basic parameters."""
        builder = PlanBuilderV2()

        result = builder.llm_step(task="Analyze the data")

        assert result is builder  # Should return self for chaining
        assert len(builder.plan.steps) == 1
        assert isinstance(builder.plan.steps[0], LLMStep)
        assert builder.plan.steps[0].task == "Analyze the data"
        assert builder.plan.steps[0].inputs == []
        assert builder.plan.steps[0].output_schema is None
        assert builder.plan.steps[0].step_name == "step_0"

    def test_llm_step_method_with_all_parameters(self) -> None:
        """Test the llm_step() method with all parameters."""
        builder = PlanBuilderV2()
        inputs = ["input1", StepOutput(0), Input("user_input")]

        builder.llm_step(
            task="Process the inputs",
            inputs=inputs,
            output_schema=OutputSchema,
            step_name="custom_step",
        )

        step = builder.plan.steps[0]
        assert isinstance(step, LLMStep)
        assert step.task == "Process the inputs"
        assert step.inputs == inputs
        assert step.output_schema == OutputSchema
        assert step.step_name == "custom_step"

    def test_llm_step_method_auto_generated_step_name(self) -> None:
        """Test that step names are auto-generated correctly."""
        builder = PlanBuilderV2()

        builder.llm_step(task="First step")
        builder.llm_step(task="Second step")

        assert builder.plan.steps[0].step_name == "step_0"
        assert builder.plan.steps[1].step_name == "step_1"

    def test_invoke_tool_step_method_with_string_tool(self) -> None:
        """Test the invoke_tool_step() method with string tool identifier."""
        builder = PlanBuilderV2()
        args = {"param1": "value1", "param2": StepOutput(0)}

        result = builder.invoke_tool_step(tool="search_tool", args=args)

        assert result is builder  # Should return self for chaining
        assert len(builder.plan.steps) == 1
        assert isinstance(builder.plan.steps[0], InvokeToolStep)
        assert builder.plan.steps[0].tool == "search_tool"
        assert builder.plan.steps[0].args == args
        assert builder.plan.steps[0].output_schema is None
        assert builder.plan.steps[0].step_name == "step_0"

    def test_invoke_tool_step_method_with_tool_instance(self) -> None:
        """Test the invoke_tool_step() method with Tool instance."""
        builder = PlanBuilderV2()
        mock_tool = MockTool()

        builder.invoke_tool_step(tool=mock_tool, args={"input": "test"})

        step = builder.plan.steps[0]
        assert isinstance(step, InvokeToolStep)
        assert step.tool is mock_tool

    def test_invoke_tool_step_method_with_all_parameters(self) -> None:
        """Test the invoke_tool_step() method with all parameters."""
        builder = PlanBuilderV2()

        builder.invoke_tool_step(
            tool="test_tool",
            args={"arg1": "value1"},
            output_schema=OutputSchema,
            step_name="tool_step",
        )

        step = builder.plan.steps[0]
        assert isinstance(step, InvokeToolStep)
        assert step.tool == "test_tool"
        assert step.args == {"arg1": "value1"}
        assert step.output_schema == OutputSchema
        assert step.step_name == "tool_step"

    def test_invoke_tool_step_method_no_args(self) -> None:
        """Test the invoke_tool_step() method with no args."""
        builder = PlanBuilderV2()

        builder.invoke_tool_step(tool="no_args_tool")

        step = builder.plan.steps[0]
        assert isinstance(step, InvokeToolStep)
        assert step.args == {}

    def test_function_step_method_basic(self) -> None:
        """Test the function_step() method with basic parameters."""
        builder = PlanBuilderV2()

        result = builder.function_step(function=example_function_for_testing)

        assert result is builder  # Should return self for chaining
        assert len(builder.plan.steps) == 1
        assert isinstance(builder.plan.steps[0], FunctionStep)
        assert builder.plan.steps[0].function is example_function_for_testing
        assert builder.plan.steps[0].args == {}
        assert builder.plan.steps[0].output_schema is None
        assert builder.plan.steps[0].step_name == "step_0"

    def test_function_step_method_with_all_parameters(self) -> None:
        """Test the function_step() method with all parameters."""
        builder = PlanBuilderV2()
        args = {"x": 42, "y": Input("user_input")}

        builder.function_step(
            function=example_function_for_testing,
            args=args,
            output_schema=OutputSchema,
            step_name="func_step",
        )

        step = builder.plan.steps[0]
        assert isinstance(step, FunctionStep)
        assert step.function is example_function_for_testing
        assert step.args == args
        assert step.output_schema == OutputSchema
        assert step.step_name == "func_step"

    def test_single_tool_agent_step_method_basic(self) -> None:
        """Test the single_tool_agent_step() method with basic parameters."""
        builder = PlanBuilderV2()

        result = builder.single_tool_agent_step(tool="agent_tool", task="Complete the task")

        assert result is builder  # Should return self for chaining
        assert len(builder.plan.steps) == 1
        assert isinstance(builder.plan.steps[0], SingleToolAgentStep)
        assert builder.plan.steps[0].tool == "agent_tool"
        assert builder.plan.steps[0].task == "Complete the task"
        assert builder.plan.steps[0].inputs == []
        assert builder.plan.steps[0].output_schema is None
        assert builder.plan.steps[0].step_name == "step_0"

    def test_single_tool_agent_step_method_with_all_parameters(self) -> None:
        """Test the single_tool_agent_step() method with all parameters."""
        builder = PlanBuilderV2()
        inputs = ["context", StepOutput(0)]

        builder.single_tool_agent_step(
            tool="complex_tool",
            task="Process complex data",
            inputs=inputs,
            output_schema=OutputSchema,
            step_name="agent_step",
        )

        step = builder.plan.steps[0]
        assert isinstance(step, SingleToolAgentStep)
        assert step.tool == "complex_tool"
        assert step.task == "Process complex data"
        assert step.inputs == inputs
        assert step.output_schema == OutputSchema
        assert step.step_name == "agent_step"

    def test_final_output_method_basic(self) -> None:
        """Test the final_output() method with basic parameters."""
        builder = PlanBuilderV2()

        result = builder.final_output()

        assert result is builder  # Should return self for chaining
        assert builder.plan.final_output_schema is None
        assert builder.plan.summarize is False

    def test_final_output_method_with_schema(self) -> None:
        """Test the final_output() method with output schema."""
        builder = PlanBuilderV2()

        builder.final_output(output_schema=OutputSchema)

        assert builder.plan.final_output_schema == OutputSchema
        assert builder.plan.summarize is False

    def test_final_output_method_with_summarize(self) -> None:
        """Test the final_output() method with summarize enabled."""
        builder = PlanBuilderV2()

        builder.final_output(summarize=True)

        assert builder.plan.final_output_schema is None
        assert builder.plan.summarize is True

    def test_final_output_method_with_all_parameters(self) -> None:
        """Test the final_output() method with all parameters."""
        builder = PlanBuilderV2()

        builder.final_output(output_schema=OutputSchema, summarize=True)

        assert builder.plan.final_output_schema == OutputSchema
        assert builder.plan.summarize is True

    def test_build_method(self) -> None:
        """Test the build() method returns correct PlanV2 instance."""
        builder = PlanBuilderV2(label="Test Plan")

        builder.input(name="test_input", description="Test input description")
        builder.llm_step(task="Test task")
        builder.final_output(output_schema=OutputSchema, summarize=True)

        plan = builder.build()

        assert isinstance(plan, PlanV2)
        assert plan is builder.plan  # Should return the same instance
        assert plan.label == "Test Plan"
        assert len(plan.plan_inputs) == 1
        assert len(plan.steps) == 1
        assert plan.final_output_schema == OutputSchema
        assert plan.summarize is True

    def test_method_chaining(self) -> None:
        """Test that all methods return self for proper chaining."""
        builder = PlanBuilderV2("Chaining Test")

        result = (
            builder.input(
                name="user_name", description="Name of the user", default_value="John Doe"
            )
            .input(name="user_age", description="Age of the user", default_value=25)
            .llm_step(task="Analyze user info", inputs=[Input("user_name"), Input("user_age")])
            .invoke_tool_step(tool="search_tool", args={"query": StepOutput(0)})
            .function_step(function=example_function_for_testing, args={"x": 1, "y": "test"})
            .single_tool_agent_step(tool="agent_tool", task="Final processing")
            .final_output(output_schema=OutputSchema, summarize=True)
        )

        assert result is builder

        # Verify the plan was built correctly
        plan = builder.build()
        assert len(plan.plan_inputs) == 2
        assert len(plan.steps) == 4
        assert isinstance(plan.steps[0], LLMStep)
        assert isinstance(plan.steps[1], InvokeToolStep)
        assert isinstance(plan.steps[2], FunctionStep)
        assert isinstance(plan.steps[3], SingleToolAgentStep)
        assert plan.final_output_schema == OutputSchema
        assert plan.summarize is True

        # Verify default values are set correctly
        inputs = {inp.name: inp for inp in plan.plan_inputs}
        assert inputs["user_name"].value == "John Doe"
        assert inputs["user_age"].value == 25

    def test_empty_plan_build(self) -> None:
        """Test building an empty plan."""
        builder = PlanBuilderV2()
        plan = builder.build()

        assert isinstance(plan, PlanV2)
        assert len(plan.steps) == 0
        assert len(plan.plan_inputs) == 0
        assert plan.final_output_schema is None
        assert plan.summarize is False

    def test_step_name_generation_with_mixed_steps(self) -> None:
        """Test step name generation with different types of steps."""
        builder = PlanBuilderV2()

        builder.llm_step(task="LLM task")
        builder.invoke_tool_step(tool="tool1")
        builder.function_step(function=example_function_for_testing)
        builder.single_tool_agent_step(tool="agent_tool", task="Agent task")

        assert builder.plan.steps[0].step_name == "step_0"
        assert builder.plan.steps[1].step_name == "step_1"
        assert builder.plan.steps[2].step_name == "step_2"
        assert builder.plan.steps[3].step_name == "step_3"

    def test_custom_step_names_override_auto_generation(self) -> None:
        """Test that custom step names override auto-generation."""
        builder = PlanBuilderV2()

        builder.llm_step(task="First", step_name="custom_first")
        builder.llm_step(task="Second")  # Should get step_1
        builder.llm_step(task="Third", step_name="custom_third")

        assert builder.plan.steps[0].step_name == "custom_first"
        assert builder.plan.steps[1].step_name == "step_1"
        assert builder.plan.steps[2].step_name == "custom_third"

    def test_references_in_inputs_and_args(self) -> None:
        """Test using references (StepOutput and Input) in various contexts."""
        builder = PlanBuilderV2()

        # Add inputs to reference
        builder.input(name="user_query", description="The user's query")

        # Add steps with references
        builder.llm_step(task="Process query", inputs=[Input("user_query"), "additional context"])
        builder.invoke_tool_step(tool="search_tool", args={"query": StepOutput(0), "limit": 10})
        builder.function_step(
            function=example_function_for_testing, args={"x": 42, "y": StepOutput(1)}
        )

        plan = builder.build()

        # Verify references are preserved
        llm_step = plan.steps[0]
        assert isinstance(llm_step, LLMStep)
        assert isinstance(llm_step.inputs[0], Input)
        assert llm_step.inputs[0].name == "user_query"

        tool_step = plan.steps[1]
        assert isinstance(tool_step, InvokeToolStep)
        assert isinstance(tool_step.args["query"], StepOutput)
        assert tool_step.args["query"].step == 0

        func_step = plan.steps[2]
        assert isinstance(func_step, FunctionStep)
        assert isinstance(func_step.args["y"], StepOutput)
        assert func_step.args["y"].step == 1
