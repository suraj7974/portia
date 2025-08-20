"""Tests for async storage methods."""

import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from pytest_httpx import HTTPXMock

from portia.end_user import EndUser
from portia.errors import PlanNotFoundError, PlanRunNotFoundError, StorageError
from portia.execution_agents.output import (
    AgentMemoryValue,
    LocalDataValue,
)
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunUUID
from portia.storage import (
    MAX_STORAGE_OBJECT_BYTES,
    DiskFileStorage,
    InMemoryStorage,
    PlanRunListResponse,
    PortiaCloudStorage,
)
from portia.tool_call import ToolCallRecord, ToolCallStatus
from tests.utils import get_test_config, get_test_tool_call


@pytest.mark.asyncio
async def test_async_plan_storage_methods() -> None:
    """Test async plan storage methods."""
    storage = InMemoryStorage()
    plan = Plan(
        id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
        plan_context=PlanContext(
            query="test query",
            tool_ids=["test_tool"],
        ),
        steps=[],
        plan_inputs=[],
    )

    # Test async save_plan
    await storage.asave_plan(plan)
    assert storage.plan_exists(plan.id)

    # Test async get_plan
    retrieved_plan = await storage.aget_plan(plan.id)
    assert retrieved_plan.id == plan.id
    assert retrieved_plan.plan_context.query == plan.plan_context.query

    # Test async plan_exists
    exists = await storage.aplan_exists(plan.id)
    assert exists is True

    # Test async get_plan_by_query
    retrieved_plan = await storage.aget_plan_by_query("test query")
    assert retrieved_plan.id == plan.id


@pytest.mark.asyncio
async def test_async_run_storage_methods() -> None:
    """Test async run storage methods."""
    storage = InMemoryStorage()
    plan_run = PlanRun(
        id=PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
        plan_id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
        end_user_id="test_user",
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
        outputs=PlanRunOutputs(),
        plan_run_inputs={},
    )

    # Test async save_plan_run
    await storage.asave_plan_run(plan_run)
    retrieved_run = storage.get_plan_run(plan_run.id)
    assert retrieved_run.id == plan_run.id

    # Test async get_plan_run
    retrieved_run = await storage.aget_plan_run(plan_run.id)
    assert retrieved_run.id == plan_run.id

    # Test async get_plan_runs
    runs_response = await storage.aget_plan_runs()
    assert len(runs_response.results) == 1
    assert runs_response.results[0].id == plan_run.id


@pytest.mark.asyncio
async def test_async_additional_storage_methods() -> None:
    """Test async additional storage methods."""
    storage = InMemoryStorage()
    tool_call = ToolCallRecord(
        plan_run_id=PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
        tool_name="test_tool",
        step=1,
        end_user_id="test_user",
        input={"test": "input"},
        output="test output",
        status=ToolCallStatus.SUCCESS,
        latency_seconds=1.0,
    )

    # Test async save_tool_call
    await storage.asave_tool_call(tool_call)
    # Tool calls are just logged, so we just verify no exception is raised

    # Test async save_end_user
    end_user = EndUser(
        external_id="test_user",
        name="Test User",
        email="test@example.com",
        phone_number="",
        additional_data={},
    )

    saved_user = await storage.asave_end_user(end_user)
    assert saved_user.external_id == end_user.external_id

    # Test async get_end_user
    retrieved_user = await storage.aget_end_user("test_user")
    assert retrieved_user is not None
    assert retrieved_user.external_id == "test_user"


@pytest.mark.asyncio
async def test_async_aget_plan_run_success(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_run method with successful response."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan_run_id = PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987")

    # Mock successful response
    mock_response_data = {
        "id": str(plan_run_id),
        "plan": {"id": "plan-12345678-1234-5678-1234-567812345678"},
        "end_user": "test_user",
        "current_step_index": 2,
        "state": "IN_PROGRESS",
        "outputs": {"final_output": None, "step_outputs": {}, "clarifications": []},
        "plan_run_inputs": {
            "input1": {"value": "test_value", "summary": "test summary"},
            "input2": {"value": "42", "summary": None},
        },
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run_id}/",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method
    result = await storage.aget_plan_run(plan_run_id)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v0/plan-runs/{plan_run_id}/"

    # Verify the returned PlanRun object
    assert isinstance(result, PlanRun)
    assert result.id == plan_run_id
    assert result.plan_id == PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")
    assert result.end_user_id == "test_user"
    assert result.current_step_index == 2
    assert result.state == PlanRunState.IN_PROGRESS
    assert isinstance(result.outputs, PlanRunOutputs)
    assert len(result.plan_run_inputs) == 2
    assert result.plan_run_inputs["input1"].value == "test_value"
    assert result.plan_run_inputs["input1"].summary == "test summary"
    assert result.plan_run_inputs["input2"].value == "42"
    assert result.plan_run_inputs["input2"].summary is None


@pytest.mark.asyncio
async def test_async_aget_plan_run_http_error(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_run method with HTTP error response."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan_run_id = PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987")

    # Mock error response
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run_id}/",
        status_code=404,
        content=b"Plan run not found",
    )

    # Test that StorageError is raised
    with pytest.raises(StorageError, match="Plan run not found"):
        await storage.aget_plan_run(plan_run_id)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v0/plan-runs/{plan_run_id}/"


@pytest.mark.asyncio
async def test_async_aget_plan_run_request_exception(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_run method with request exception."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan_run_id = PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987")

    # Mock request exception
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run_id}/",
        exception=httpx.ConnectError("Connection failed"),
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_run(plan_run_id)


@pytest.mark.asyncio
async def test_async_aget_plan_runs_no_params(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_runs method with no parameters."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock successful response
    mock_response_data = {
        "results": [
            {
                "id": "prun-87654321-4321-8765-4321-876543210987",
                "plan": {"id": "plan-12345678-1234-5678-1234-567812345678"},
                "end_user": "test_user_1",
                "current_step_index": 1,
                "state": "IN_PROGRESS",
                "outputs": {"final_output": None, "step_outputs": {}, "clarifications": []},
                "plan_run_inputs": {
                    "input1": {"value": "test_value_1", "summary": "test summary 1"}
                },
            },
            {
                "id": "prun-87654321-4321-8765-4321-876543210988",
                "plan": {"id": "plan-12345678-1234-5678-1234-567812345679"},
                "end_user": "test_user_2",
                "current_step_index": 2,
                "state": "COMPLETE",
                "outputs": {"final_output": None, "step_outputs": {}, "clarifications": []},
                "plan_run_inputs": {},
            },
        ],
        "count": 2,
        "current_page": 1,
        "total_pages": 1,
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method
    result = await storage.aget_plan_runs()

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == "/api/v0/plan-runs/"
    assert not request.url.query or str(request.url.query) in ["", "b''"]

    # Verify the returned PlanRunListResponse object
    assert isinstance(result, PlanRunListResponse)
    assert len(result.results) == 2
    assert result.count == 2
    assert result.current_page == 1
    assert result.total_pages == 1

    # Verify first plan run
    plan_run_1 = result.results[0]
    assert plan_run_1.id == PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987")
    assert plan_run_1.plan_id == PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")
    assert plan_run_1.end_user_id == "test_user_1"
    assert plan_run_1.current_step_index == 1
    assert plan_run_1.state == PlanRunState.IN_PROGRESS
    assert len(plan_run_1.plan_run_inputs) == 1
    assert plan_run_1.plan_run_inputs["input1"].value == "test_value_1"

    # Verify second plan run
    plan_run_2 = result.results[1]
    assert plan_run_2.id == PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210988")
    assert plan_run_2.state == PlanRunState.COMPLETE
    assert len(plan_run_2.plan_run_inputs) == 0


@pytest.mark.asyncio
async def test_async_aget_plan_runs_with_state(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_runs method with run_state parameter."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock successful response
    mock_response_data = {
        "results": [
            {
                "id": "prun-87654321-4321-8765-4321-876543210987",
                "plan": {"id": "plan-12345678-1234-5678-1234-567812345678"},
                "end_user": "test_user",
                "current_step_index": 3,
                "state": "COMPLETE",
                "outputs": {"final_output": None, "step_outputs": {}, "clarifications": []},
                "plan_run_inputs": {},
            }
        ],
        "count": 1,
        "current_page": 1,
        "total_pages": 1,
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?run_state=COMPLETE",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method with run_state
    result = await storage.aget_plan_runs(run_state=PlanRunState.COMPLETE)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == "/api/v0/plan-runs/"
    assert "run_state=COMPLETE" in str(request.url.query)

    # Verify the returned data
    assert isinstance(result, PlanRunListResponse)
    assert len(result.results) == 1
    assert result.results[0].state == PlanRunState.COMPLETE


@pytest.mark.asyncio
async def test_async_aget_plan_runs_with_page(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_runs method with page parameter."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock successful response
    mock_response_data = {"results": [], "count": 0, "current_page": 2, "total_pages": 2}

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?page=2",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method with page
    result = await storage.aget_plan_runs(page=2)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == "/api/v0/plan-runs/"
    assert "page=2" in str(request.url.query)

    # Verify the returned data
    assert isinstance(result, PlanRunListResponse)
    assert len(result.results) == 0
    assert result.current_page == 2
    assert result.total_pages == 2


@pytest.mark.asyncio
async def test_async_aget_plan_runs_both_params(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_runs method with both run_state and page parameters."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock successful response
    mock_response_data = {
        "results": [
            {
                "id": "prun-87654321-4321-8765-4321-876543210987",
                "plan": {"id": "plan-12345678-1234-5678-1234-567812345678"},
                "end_user": "test_user",
                "current_step_index": 0,
                "state": "FAILED",
                "outputs": {"final_output": None, "step_outputs": {}, "clarifications": []},
                "plan_run_inputs": {},
            }
        ],
        "count": 1,
        "current_page": 3,
        "total_pages": 5,
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?page=3&run_state=FAILED",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method with both parameters
    result = await storage.aget_plan_runs(run_state=PlanRunState.FAILED, page=3)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == "/api/v0/plan-runs/"
    query_str = str(request.url.query)
    assert "page=3" in query_str
    assert "run_state=FAILED" in query_str

    # Verify the returned data
    assert isinstance(result, PlanRunListResponse)
    assert len(result.results) == 1
    assert result.results[0].state == PlanRunState.FAILED
    assert result.current_page == 3
    assert result.total_pages == 5


@pytest.mark.asyncio
async def test_async_aget_plan_runs_http_error(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_runs method with HTTP error response."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock error response
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?",
        status_code=500,
        content=b"Internal server error",
    )

    # Test that StorageError is raised
    with pytest.raises(StorageError, match="Internal server error"):
        await storage.aget_plan_runs()

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == "/api/v0/plan-runs/"


@pytest.mark.asyncio
async def test_async_aget_plan_runs_request_exception(httpx_mock: HTTPXMock) -> None:
    """Test async aget_plan_runs method with request exception."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock request exception
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?page=1&run_state=IN_PROGRESS",
        exception=httpx.ConnectError("Connection failed"),
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_runs(run_state=PlanRunState.IN_PROGRESS, page=1)


@pytest.mark.asyncio
async def test_async_aget_end_user_success(httpx_mock: HTTPXMock) -> None:
    """Test async aget_end_user method with successful response."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    external_id = "test_user_123"

    # Mock successful response
    mock_response_data = {
        "external_id": external_id,
        "name": "Test User",
        "email": "test.user@example.com",
        "phone_number": "+1234567890",
        "additional_data": {"department": "Engineering", "role": "Developer"},
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{external_id}/",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method
    result = await storage.aget_end_user(external_id)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v0/end-user/{external_id}/"

    # Verify the returned EndUser object
    assert result is not None
    assert isinstance(result, EndUser)
    assert result.external_id == external_id
    assert result.name == "Test User"
    assert result.email == "test.user@example.com"
    assert result.phone_number == "+1234567890"
    assert result.additional_data == {"department": "Engineering", "role": "Developer"}


@pytest.mark.asyncio
async def test_async_aget_end_user_minimal_data(httpx_mock: HTTPXMock) -> None:
    """Test async aget_end_user method with minimal user data."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    external_id = "minimal_user"

    # Mock successful response with minimal data
    mock_response_data = {
        "external_id": external_id,
        "name": "",
        "email": "",
        "phone_number": "",
        "additional_data": {},
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{external_id}/",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method
    result = await storage.aget_end_user(external_id)

    # Verify the returned EndUser object
    assert result is not None
    assert isinstance(result, EndUser)
    assert result.external_id == external_id
    assert result.name == ""
    assert result.email == ""
    assert result.phone_number == ""
    assert result.additional_data == {}


@pytest.mark.asyncio
async def test_async_aget_end_user_not_found(httpx_mock: HTTPXMock) -> None:
    """Test async aget_end_user method with 404 not found response."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    external_id = "nonexistent_user"

    # Mock 404 error response
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{external_id}/",
        status_code=404,
        content=b"End user not found",
    )

    # Test that StorageError is raised
    with pytest.raises(StorageError, match="End user not found"):
        await storage.aget_end_user(external_id)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v0/end-user/{external_id}/"


@pytest.mark.asyncio
async def test_async_aget_end_user_server_error(httpx_mock: HTTPXMock) -> None:
    """Test async aget_end_user method with server error response."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    external_id = "test_user"

    # Mock 500 server error response
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{external_id}/",
        status_code=500,
        content=b"Internal server error",
    )

    # Test that StorageError is raised
    with pytest.raises(StorageError, match="Internal server error"):
        await storage.aget_end_user(external_id)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v0/end-user/{external_id}/"


@pytest.mark.asyncio
async def test_async_aget_end_user_request_exception(httpx_mock: HTTPXMock) -> None:
    """Test async aget_end_user method with request exception."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    external_id = "test_user"

    # Mock request exception
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{external_id}/",
        exception=httpx.ConnectError("Connection failed"),
    )

    with pytest.raises(StorageError):
        await storage.aget_end_user(external_id)


@pytest.mark.asyncio
async def test_async_aget_end_user_special_characters(httpx_mock: HTTPXMock) -> None:
    """Test async aget_end_user method with special characters in external_id."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Test with special characters that might need URL encoding
    external_id = "user@domain.com"

    # Mock successful response
    mock_response_data = {
        "external_id": external_id,
        "name": "User With Email ID",
        "email": "user@domain.com",
        "phone_number": "",
        "additional_data": {},
    }

    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{external_id}/",
        status_code=200,
        json=mock_response_data,
    )

    # Test the method
    result = await storage.aget_end_user(external_id)

    # Verify the request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v0/end-user/{external_id}/"

    # Verify the returned EndUser object
    assert result is not None
    assert isinstance(result, EndUser)
    assert result.external_id == external_id
    assert result.name == "User With Email ID"
    assert result.email == "user@domain.com"


@pytest.mark.asyncio
async def test_async_aget_similar_plans_threaded_execution() -> None:
    """Test async aget_similar_plans method uses asyncio.to_thread correctly."""
    storage = InMemoryStorage()

    # Mock data to return from get_similar_plans
    mock_plan = Plan(
        id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
        plan_context=PlanContext(
            query="test query",
            tool_ids=["test_tool"],
        ),
        steps=[],
        plan_inputs=[],
    )
    mock_plans = [mock_plan]

    # Mock the synchronous get_similar_plans method
    with patch.object(storage, "get_similar_plans", return_value=mock_plans) as mock_get_similar:
        # Test the async method
        result = await storage.aget_similar_plans(query="test query", threshold=0.7, limit=5)

        # Verify that get_similar_plans was called with correct parameters
        mock_get_similar.assert_called_once_with("test query", 0.7, 5)

        # Verify the result is returned correctly
        assert result == mock_plans
        assert len(result) == 1
        assert result[0].id == mock_plan.id
        assert result[0].plan_context.query == "test query"


@pytest.mark.asyncio
async def test_async_aget_similar_plans_default_parameters() -> None:
    """Test async aget_similar_plans method with default parameters."""
    storage = InMemoryStorage()

    mock_plans = []

    # Mock the synchronous get_similar_plans method
    with patch.object(storage, "get_similar_plans", return_value=mock_plans) as mock_get_similar:
        # Test the async method with default parameters
        result = await storage.aget_similar_plans("search query")

        # Verify that get_similar_plans was called with default threshold and limit
        mock_get_similar.assert_called_once_with("search query", 0.5, 10)

        # Verify the result
        assert result == mock_plans
        assert len(result) == 0


@pytest.mark.asyncio
async def test_async_agent_memory_methods() -> None:
    """Test async agent memory methods."""
    storage = InMemoryStorage()
    from portia.execution_agents.output import LocalDataValue

    output = LocalDataValue(
        summary="test summary",
        value="test value",
    )

    # Test async save_plan_run_output
    result = await storage.asave_plan_run_output(
        "test_output",
        output,
        PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
    )
    assert result.summary == "test summary"

    # Test async get_plan_run_output
    retrieved_output = await storage.aget_plan_run_output(
        "test_output",
        PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
    )
    assert retrieved_output.value == "test value"


@pytest.mark.asyncio
async def test_async_disk_file_storage_methods() -> None:
    """Test async disk file storage methods."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DiskFileStorage(temp_dir)
        plan = Plan(
            id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
            plan_context=PlanContext(
                query="test query",
                tool_ids=["test_tool"],
            ),
            steps=[],
            plan_inputs=[],
        )

        # Test async save_plan
        await storage.asave_plan(plan)
        assert storage.plan_exists(plan.id)

        # Test async get_plan
        retrieved_plan = await storage.aget_plan(plan.id)
        assert retrieved_plan.id == plan.id
        assert retrieved_plan.plan_context.query == plan.plan_context.query

        # Test async plan_exists
        exists = await storage.aplan_exists(plan.id)
        assert exists is True


@pytest.mark.asyncio
async def test_async_error_handling() -> None:
    """Test that async methods properly propagate errors."""
    storage = InMemoryStorage()

    # Test that non-existent plan raises error
    with pytest.raises(PlanNotFoundError):
        await storage.aget_plan(PlanUUID.from_string("plan-99999999-9999-9999-9999-999999999999"))

    # Test that non-existent plan_run raises error
    with pytest.raises(PlanRunNotFoundError):
        await storage.aget_plan_run(
            PlanRunUUID.from_string("prun-99999999-9999-9999-9999-999999999999")
        )


@pytest.mark.asyncio
async def test_async_concurrent_operations() -> None:
    """Test that async methods can be called concurrently."""
    storage = InMemoryStorage()
    plans = []

    # Create multiple plans
    for i in range(5):
        plan = Plan(
            id=PlanUUID.from_string(f"plan-{i:08d}-1234-5678-1234-567812345678"),
            plan_context=PlanContext(
                query=f"test query {i}",
                tool_ids=["test_tool"],
            ),
            steps=[],
            plan_inputs=[],
        )
        plans.append(plan)

    # Test concurrent save operations
    import asyncio

    tasks = [storage.asave_plan(plan) for plan in plans]
    await asyncio.gather(*tasks)

    # Verify all plans were saved
    for plan in plans:
        assert storage.plan_exists(plan.id)

    # Test concurrent get operations
    tasks = [storage.aget_plan(plan.id) for plan in plans]
    retrieved_plans = await asyncio.gather(*tasks)

    # Verify all plans were retrieved correctly
    for i, retrieved_plan in enumerate(retrieved_plans):
        assert retrieved_plan.id == plans[i].id
        assert retrieved_plan.plan_context.query == plans[i].plan_context.query


@pytest.mark.asyncio
async def test_async_portia_cloud_storage(httpx_mock: HTTPXMock) -> None:
    """Test async PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
        plan_inputs=[
            PlanInput(name="key1", description="Test input 1"),
            PlanInput(name="key2", description="Test input 2"),
        ],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
        plan_run_inputs={
            "param1": LocalDataValue(value="test"),
            "param2": LocalDataValue(value=456),
        },
    )
    tool_call = get_test_tool_call(plan_run)

    end_user = EndUser(external_id="123")

    # Test async save_plan failure
    httpx_mock.add_response(
        method="POST",
        url=f"{config.portia_api_endpoint}/api/v0/plans/",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.asave_plan(plan)

    # Test async get_plan failure
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plans/{plan.id}/",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.aget_plan(plan.id)

    # Test async save_run failure
    httpx_mock.add_response(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.asave_plan_run(plan_run)

    # Test async get_run failure
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.aget_plan_run(plan_run.id)

    # Test async get_runs failure
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.aget_plan_runs()

    # Test async save_tool_call - should not raise an exception
    httpx_mock.add_response(
        method="POST",
        url=f"{config.portia_api_endpoint}/api/v0/tool-calls/",
        status_code=500,
        content=b"An error occurred.",
    )

    await storage.asave_tool_call(tool_call)

    # Test async get_end_user failure
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{end_user.external_id}/",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.aget_end_user(end_user.external_id)

    # Test async save_end_user failure
    httpx_mock.add_response(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{end_user.external_id}/",
        status_code=500,
        content=b"An error occurred.",
    )

    with pytest.raises(StorageError, match="An error occurred."):
        await storage.asave_end_user(end_user)


@pytest.mark.asyncio
async def test_async_portia_cloud_storage_errors(httpx_mock: HTTPXMock) -> None:
    """Test async PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
    )

    tool_call = get_test_tool_call(plan_run)

    end_user = EndUser(external_id="123")

    # Test async save_plan failure - simulate network error
    httpx_mock.add_exception(
        method="POST",
        url=f"{config.portia_api_endpoint}/api/v0/plans/",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.asave_plan(plan)

    # Test async get_plan failure - simulate network error
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plans/{plan.id}/",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.aget_plan(plan.id)

    # Test async save_run failure - simulate network error
    httpx_mock.add_exception(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.asave_plan_run(plan_run)

    # Test async get_run failure - simulate network error
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_run(plan_run.id)

    # Test async get_runs failure - simulate network error
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_runs()

    # Test async get_runs with parameters failure - simulate network error
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?page=10&run_state=COMPLETE",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_runs(run_state=PlanRunState.COMPLETE, page=10)

    # Test async save_tool_call - should not raise an exception
    httpx_mock.add_exception(
        method="POST",
        url=f"{config.portia_api_endpoint}/api/v0/tool-calls/",
        exception=RuntimeError("An error occurred."),
    )

    await storage.asave_tool_call(tool_call)

    # Test async save_end_user failure - simulate network error
    httpx_mock.add_exception(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{end_user.external_id}/",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.asave_end_user(end_user)

    # Test async get_end_user failure - simulate network error
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/end-user/{end_user.external_id}/",
        exception=RuntimeError("An error occurred."),
    )

    with pytest.raises(StorageError):
        await storage.aget_end_user(end_user.external_id)


@pytest.mark.asyncio
async def test_async_portia_cloud_agent_memory(httpx_mock: HTTPXMock) -> None:
    """Test async PortiaCloudStorage agent memory."""
    config = get_test_config(portia_api_key="test_api_key")
    agent_memory = PortiaCloudStorage(config)
    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
    )
    output = LocalDataValue(value="test value", summary="test summary")

    # Test saving an output
    httpx_mock.add_response(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
        status_code=200,
    )

    result = await agent_memory.asave_plan_run_output("test_output", output, plan_run.id)

    # Verify the PUT request was made correctly
    assert len(httpx_mock.get_requests()) == 1
    put_request = httpx_mock.get_requests()[0]
    assert put_request.method == "PUT"
    assert (
        put_request.url.path == f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/"
    )

    # Verify the result
    assert isinstance(result, AgentMemoryValue)
    assert result.output_name == "test_output"
    assert result.plan_run_id == plan_run.id
    assert result.summary == output.get_summary()
    assert Path(f".portia/cache/agent_memory/{plan_run.id}/test_output.json").is_file()

    # Test getting an output when it is cached locally
    # Since we're using httpx_mock, we need to mock the cache read instead
    with patch.object(agent_memory, "_read_from_cache", return_value=output):
        result = await agent_memory.aget_plan_run_output("test_output", plan_run.id)

        # Verify the returned output
        assert result.get_summary() == output.get_summary()
        assert result.get_value() == output.get_value()

    # Test getting an output when it is not cached locally
    # Mock the metadata response
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output2/",
        status_code=200,
        json={
            "summary": "test summary 2",
            "url": "https://example.com/output2",
        },
    )

    # Mock the value response
    httpx_mock.add_response(
        method="GET",
        url="https://example.com/output2",
        status_code=200,
        content=b"test value 2",
    )

    with (
        patch.object(agent_memory, "_read_from_cache", side_effect=FileNotFoundError),
        patch.object(agent_memory, "_write_to_cache") as mock_write_cache,
    ):
        result = await agent_memory.aget_plan_run_output("test_output2", plan_run.id)

        # Verify that both HTTP requests were made
        assert len(httpx_mock.get_requests()) >= 3  # Previous requests + 2 from get

        # Verify the metadata request
        metadata_request = httpx_mock.get_requests()[-2]
        assert metadata_request.method == "GET"
        assert (
            metadata_request.url.path
            == f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output2/"
        )

        # Verify the value request
        value_request = httpx_mock.get_requests()[-1]
        assert value_request.method == "GET"
        assert value_request.url == "https://example.com/output2"

        # Verify that it wrote to the local cache
        mock_write_cache.assert_called_once()

        # Verify the returned output
        assert result.get_summary() == "test summary 2"
        assert result.get_value() == "test value 2"


@pytest.mark.asyncio
async def test_async_portia_cloud_agent_memory_errors(httpx_mock: HTTPXMock) -> None:
    """Test async PortiaCloudStorage raises StorageError on agent memory failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    agent_memory = PortiaCloudStorage(config)
    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
    )
    output = LocalDataValue(value="test value", summary="test summary")

    mock_exception = RuntimeError("An error occurred.")

    # Test async save_plan_run_output error
    httpx_mock.add_exception(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
        exception=mock_exception,
    )

    with pytest.raises(StorageError):
        await agent_memory.asave_plan_run_output("test_output", output, plan_run.id)

    # Test async get_plan_run_output error
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
        exception=mock_exception,
    )

    with (
        patch.object(
            agent_memory,
            "_read_from_cache",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(StorageError),
    ):
        await agent_memory.aget_plan_run_output("test_output", plan_run.id)

    # Check with an output that's too large
    with (
        patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
        pytest.raises(StorageError),
    ):
        await agent_memory.asave_plan_run_output(
            "large_output",
            LocalDataValue(value="large value"),
            plan_run.id,
        )

    # Test for 413 REQUEST_ENTITY_TOO_LARGE response status
    httpx_mock.add_response(
        method="PUT",
        url=f"{config.portia_api_endpoint}/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/too_large_output/",
        status_code=httpx.codes.REQUEST_ENTITY_TOO_LARGE,
        content=b"Some content that's too large",
    )

    with pytest.raises(StorageError):
        await agent_memory.asave_plan_run_output(
            "too_large_output",
            LocalDataValue(value="too large value"),
            plan_run.id,
        )

    # Test for response.request.content > MAX_STORAGE_OBJECT_BYTES
    with (
        patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
        pytest.raises(StorageError),
    ):
        await agent_memory.asave_plan_run_output(
            "over_size_limit",
            LocalDataValue(value="value that creates a large request"),
            plan_run.id,
        )


@pytest.mark.asyncio
async def test_async_similar_plans(httpx_mock: HTTPXMock) -> None:
    """Test the async similar_plans method."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)
    mock_id = "plan-00000000-0000-0000-0000-000000000000"
    mock_response = {
        "id": mock_id,
        "steps": [],
        "query": "Test query",
        "tool_ids": [],
    }
    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=200,
        method="POST",
        match_json={
            "query": "Test query",
            "threshold": 0.5,
            "limit": 5,
        },
        json=[mock_response, mock_response],
    )

    plans = await storage.aget_similar_plans("Test query")
    assert len(plans) == 2
    assert plans[0].id == PlanUUID.from_string(mock_id)
    assert plans[1].id == PlanUUID.from_string(mock_id)


@pytest.mark.asyncio
async def test_async_similar_plans_error(httpx_mock: HTTPXMock) -> None:
    """Test the async similar_plans method with an error."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)
    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=500,
    )

    with pytest.raises(StorageError):
        await storage.aget_similar_plans("Test query")


@pytest.mark.asyncio
async def test_async_plan_exists_portia_cloud_storage(httpx_mock: HTTPXMock) -> None:
    """Test async plan_exists method with PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[],
    )

    # Test when plan exists
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plans/{plan.id}/",
        status_code=200,
        json={"id": str(plan.id)},
    )
    exists = await storage.aplan_exists(plan.id)
    assert exists is True

    # Test when plan doesn't exist
    different_plan_id = PlanUUID()
    httpx_mock.add_response(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plans/{different_plan_id}/",
        status_code=404,
        content=b"Not found",
    )
    exists = await storage.aplan_exists(different_plan_id)
    assert exists is False

    # Test when API call fails
    httpx_mock.add_exception(
        method="GET",
        url=f"{config.portia_api_endpoint}/api/v0/plans/{plan.id}/",
        exception=Exception("API Error"),
    )
    exists = await storage.aplan_exists(plan.id)
    assert exists is False


@pytest.mark.asyncio
async def test_async_get_plan_by_query_portia_cloud_storage(httpx_mock: HTTPXMock) -> None:
    """Test async get_plan_by_query method with PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock the get_similar_plans response
    mock_plan_response = {
        "id": "plan-00000000-0000-0000-0000-000000000000",
        "steps": [],
        "query": "test query",
        "tool_ids": ["tool1"],
    }

    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=200,
        method="POST",
        match_json={
            "query": "test query",
            "threshold": 1.0,
            "limit": 1,
        },
        json=[mock_plan_response],
    )

    # Test finding existing plan
    found_plan = await storage.aget_plan_by_query("test query")
    assert found_plan.plan_context.query == "test query"
    assert found_plan.id == PlanUUID.from_string("plan-00000000-0000-0000-0000-000000000000")

    # Test with no matching plans
    httpx_mock.add_response(
        url=url,
        status_code=200,
        method="POST",
        match_json={
            "query": "non-existent query",
            "threshold": 1.0,
            "limit": 1,
        },
        json=[],
    )

    with pytest.raises(StorageError, match="No plan found for query: non-existent query"):
        await storage.aget_plan_by_query("non-existent query")


@pytest.mark.asyncio
async def test_async_get_plan_by_query_portia_cloud_storage_error(httpx_mock: HTTPXMock) -> None:
    """Test async get_plan_by_query method with PortiaCloudStorage when API fails."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"

    # Test with API error
    httpx_mock.add_response(
        url=url,
        status_code=500,
        method="POST",
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_by_query("test query")

    # Test with network error - we need to use a callback to raise the exception
    def raise_connection_error(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    httpx_mock.add_callback(
        raise_connection_error,
        url=url,
        method="POST",
        match_json={
            "query": "test query",
            "threshold": 1.0,
            "limit": 1,
        },
    )

    with pytest.raises(StorageError):
        await storage.aget_plan_by_query("test query")
