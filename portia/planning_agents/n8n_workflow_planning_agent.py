"""
N8n Workflow Planning Agent

A specialized planning agent for creating n8n workflows from natural language descriptions.
This agent understands common workflow patterns and generates appropriate n8n node sequences.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from portia.model import Message
from portia.planning_agents.base_planning_agent import BasePlanningAgent, StepsOrError
from portia.planning_agents.context import render_prompt_insert_defaults

if TYPE_CHECKING:
    from portia.config import Config
    from portia.end_user import EndUser
    from portia.plan import Plan, PlanInput, Step
    from portia.tool import Tool

logger = logging.getLogger(__name__)

N8N_WORKFLOW_PLANNING_PROMPT = """
You are an expert n8n workflow planner. Your job is to analyze user requirements and create 
a detailed plan for building n8n workflows using available MCP tools.

You have access to these n8n workflow tools:
- create_workflow: Create a new n8n workflow
- add_gmail_trigger: Add Gmail trigger node
- add_slack_node: Add Slack message node  
- add_email_trigger: Add email trigger node
- add_webhook_trigger: Add webhook trigger node
- add_http_request_node: Add HTTP request node
- connect_nodes: Connect two nodes in a workflow
- activate_workflow: Activate the workflow

WORKFLOW PATTERNS TO RECOGNIZE:

1. EMAIL TO NOTIFICATION:
   - Gmail/Email trigger → Filter/Condition → Slack/Discord/Teams notification
   - Example: "notify me when important emails arrive"

2. WEBHOOK TO ACTION:
   - Webhook trigger → Process data → HTTP request/Database/Email
   - Example: "when someone submits form, send to database"

3. SCHEDULED TASKS:
   - Schedule trigger → HTTP request → Process response → Action
   - Example: "check website status every hour"

4. DATA INTEGRATION:
   - Trigger → Fetch data → Transform → Send to destination
   - Example: "sync data between two APIs"

5. MONITORING & ALERTS:
   - Monitor trigger → Check condition → Alert/Notification
   - Example: "alert me when server is down"

PLANNING GUIDELINES:
1. Always start with create_workflow
2. Add trigger node first (Gmail, webhook, schedule, etc.)
3. Add processing/filtering nodes if needed
4. Add action nodes (Slack, HTTP, email, etc.)
5. Connect all nodes in logical order
6. Activate workflow at the end

IMPORTANT:
- Use specific node types based on user requirements
- For email scenarios, prefer Gmail trigger over generic email
- Always connect nodes after creating them
- Include meaningful names and descriptions
- Handle common integrations (Slack, Discord, webhooks, APIs)

For each step, specify:
- tool_id: The exact tool name
- task: Clear description of what this step does
- inputs: All required parameters for the tool

User Request: {query}

Available Tools:
{tool_descriptions}

EndUser Context:
{end_user}

Create a step-by-step plan to build this n8n workflow. Return ONLY the JSON response.
"""


class N8nWorkflowPlanningAgent(BasePlanningAgent):
    """Planning agent specialized for n8n workflow creation"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.workflow_patterns = {
            "email_notification": self._plan_email_notification,
            "webhook_action": self._plan_webhook_action,
            "monitoring_alert": self._plan_monitoring_alert,
            "data_integration": self._plan_data_integration,
            "scheduled_task": self._plan_scheduled_task,
        }

    def plan(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None = None,
        max_retries: int = 3,
    ) -> StepsOrError:
        """Create a plan for building n8n workflows"""

        # Detect workflow pattern
        workflow_type = self._detect_workflow_pattern(query)
        logger.info(f"Detected workflow pattern: {workflow_type}")

        # Use pattern-specific planning if available
        if workflow_type in self.workflow_patterns:
            try:
                return self.workflow_patterns[workflow_type](
                    query, tool_list, end_user, plan_inputs
                )
            except Exception as e:
                logger.warning(
                    f"Pattern-specific planning failed: {e}, falling back to LLM"
                )

        # Fall back to LLM-based planning
        return self._llm_based_planning(
            query, tool_list, end_user, plan_inputs, max_retries
        )

    def _detect_workflow_pattern(self, query: str) -> str:
        """Detect the type of workflow from user query"""
        query_lower = query.lower()

        # Email to notification patterns
        if any(term in query_lower for term in ["email", "gmail", "inbox"]) and any(
            term in query_lower for term in ["slack", "notify", "notification", "alert"]
        ):
            return "email_notification"

        # Webhook patterns
        if any(term in query_lower for term in ["webhook", "form", "submit", "post"]):
            return "webhook_action"

        # Monitoring patterns
        if any(
            term in query_lower
            for term in ["monitor", "check", "status", "health", "down"]
        ):
            return "monitoring_alert"

        # Scheduled patterns
        if any(
            term in query_lower
            for term in ["schedule", "every", "daily", "hourly", "cron"]
        ):
            return "scheduled_task"

        # Data integration patterns
        if any(
            term in query_lower
            for term in ["sync", "integrate", "api", "database", "transfer"]
        ):
            return "data_integration"

        return "general"

    def _plan_email_notification(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None,
    ) -> StepsOrError:
        """Plan email to notification workflow"""
        steps = []

        # Extract workflow name from query
        workflow_name = self._extract_workflow_name(
            query, "Email Notification Workflow"
        )

        # Step 1: Create workflow
        steps.append(
            {
                "step_number": 1,
                "tool_id": "create_workflow",
                "task": f"Create n8n workflow for email notifications",
                "inputs": [
                    {"name": "name", "value": workflow_name},
                    {"name": "description", "value": f"Workflow created from: {query}"},
                ],
            }
        )

        # Step 2: Add Gmail trigger
        steps.append(
            {
                "step_number": 2,
                "tool_id": "add_gmail_trigger",
                "task": "Add Gmail trigger to monitor incoming emails",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {
                        "name": "filter_important",
                        "value": "important" in query.lower()
                        or "priority" in query.lower(),
                    },
                ],
            }
        )

        # Step 3: Add Slack notification
        channel = self._extract_slack_channel(query) or "#general"
        message_template = f"📧 New email received: {{{{ $node.Gmail.json.subject }}}}"

        steps.append(
            {
                "step_number": 3,
                "tool_id": "add_slack_node",
                "task": "Add Slack notification node",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {"name": "channel", "value": channel},
                    {"name": "message_template", "value": message_template},
                ],
            }
        )

        # Step 4: Connect nodes
        steps.append(
            {
                "step_number": 4,
                "tool_id": "connect_nodes",
                "task": "Connect Gmail trigger to Slack notification",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {"name": "from_node", "value": "output_from_step_2.node_id"},
                    {"name": "to_node", "value": "output_from_step_3.node_id"},
                ],
            }
        )

        # Step 5: Activate workflow
        steps.append(
            {
                "step_number": 5,
                "tool_id": "activate_workflow",
                "task": "Activate the workflow to start monitoring emails",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {"name": "active", "value": True},
                ],
            }
        )

        return StepsOrError(steps=steps, error=None)

    def _plan_webhook_action(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None,
    ) -> StepsOrError:
        """Plan webhook to action workflow"""
        steps = []

        workflow_name = self._extract_workflow_name(query, "Webhook Action Workflow")

        # Step 1: Create workflow
        steps.append(
            {
                "step_number": 1,
                "tool_id": "create_workflow",
                "task": "Create n8n workflow for webhook handling",
                "inputs": [
                    {"name": "name", "value": workflow_name},
                    {"name": "description", "value": f"Webhook workflow: {query}"},
                ],
            }
        )

        # Step 2: Add webhook trigger
        steps.append(
            {
                "step_number": 2,
                "tool_id": "add_webhook_trigger",
                "task": "Add webhook trigger to receive HTTP requests",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {"name": "webhook_path", "value": "form-submission"},
                ],
            }
        )

        # Step 3: Add HTTP request or Slack notification based on query
        if "slack" in query.lower() or "notify" in query.lower():
            steps.append(
                {
                    "step_number": 3,
                    "tool_id": "add_slack_node",
                    "task": "Add Slack notification for webhook data",
                    "inputs": [
                        {
                            "name": "workflow_id",
                            "value": "output_from_step_1.workflow_id",
                        },
                        {"name": "channel", "value": "#notifications"},
                        {
                            "name": "message_template",
                            "value": "📝 New form submission received: {{ $json }}",
                        },
                    ],
                }
            )
        else:
            steps.append(
                {
                    "step_number": 3,
                    "tool_id": "add_http_request_node",
                    "task": "Add HTTP request to process webhook data",
                    "inputs": [
                        {
                            "name": "workflow_id",
                            "value": "output_from_step_1.workflow_id",
                        },
                        {"name": "method", "value": "POST"},
                        {"name": "url", "value": "https://api.example.com/process"},
                    ],
                }
            )

        # Step 4: Connect nodes
        steps.append(
            {
                "step_number": 4,
                "tool_id": "connect_nodes",
                "task": "Connect webhook trigger to action node",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {"name": "from_node", "value": "output_from_step_2.node_id"},
                    {"name": "to_node", "value": "output_from_step_3.node_id"},
                ],
            }
        )

        # Step 5: Activate workflow
        steps.append(
            {
                "step_number": 5,
                "tool_id": "activate_workflow",
                "task": "Activate the webhook workflow",
                "inputs": [
                    {"name": "workflow_id", "value": "output_from_step_1.workflow_id"},
                    {"name": "active", "value": True},
                ],
            }
        )

        return StepsOrError(steps=steps, error=None)

    def _plan_monitoring_alert(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None,
    ) -> StepsOrError:
        """Plan monitoring and alerting workflow"""
        # For monitoring, we'll use webhook + HTTP request pattern
        return self._llm_based_planning(query, tool_list, end_user, plan_inputs, 1)

    def _plan_data_integration(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None,
    ) -> StepsOrError:
        """Plan data integration workflow"""
        # For data integration, fall back to LLM for complex scenarios
        return self._llm_based_planning(query, tool_list, end_user, plan_inputs, 1)

    def _plan_scheduled_task(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None,
    ) -> StepsOrError:
        """Plan scheduled task workflow"""
        # For scheduled tasks, fall back to LLM for now
        return self._llm_based_planning(query, tool_list, end_user, plan_inputs, 1)

    def _llm_based_planning(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        plan_inputs: list[PlanInput] | None,
        max_retries: int,
    ) -> StepsOrError:
        """Use LLM for complex workflow planning"""

        # Prepare tool descriptions
        tool_descriptions = []
        for tool in tool_list:
            if any(
                keyword in tool.id
                for keyword in ["create_workflow", "add_", "connect_", "activate_"]
            ):
                tool_descriptions.append(f"- {tool.id}: {tool.description}")

        # Render the prompt
        prompt = render_prompt_insert_defaults(
            N8N_WORKFLOW_PLANNING_PROMPT,
            query=query,
            tool_descriptions="\n".join(tool_descriptions),
            end_user=str(end_user) if end_user else "No additional context",
            plan_inputs=plan_inputs or [],
        )

        # Use the LLM to generate the plan
        previous_errors = []
        for i in range(max_retries):
            try:
                response = self.config.planning_model.invoke(
                    messages=[Message(role="user", content=prompt)]
                )
                steps_or_error = self._process_response(
                    response, tool_list, plan_inputs, i
                )
                if steps_or_error.error is None:
                    return steps_or_error
                previous_errors.append(steps_or_error.error)
            except Exception as e:
                previous_errors.append(str(e))

        return StepsOrError(
            steps=[], error="\n".join(str(error) for error in set(previous_errors))
        )

    def _extract_workflow_name(self, query: str, default: str) -> str:
        """Extract a meaningful workflow name from the user query"""
        # Simple extraction - take first few words and clean them
        words = query.split()[:4]
        name = " ".join(words).title()

        # Clean up the name
        name = "".join(c for c in name if c.isalnum() or c.isspace())
        name = " ".join(name.split())  # Remove extra spaces

        return name if len(name) > 3 else default

    def _extract_slack_channel(self, query: str) -> str | None:
        """Extract Slack channel from query if mentioned"""
        words = query.split()
        for word in words:
            if word.startswith("#"):
                return word
        return None
