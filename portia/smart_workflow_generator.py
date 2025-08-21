"""Smart n8n Workflow Generator using AI for dynamic workflow creation."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Dict, Any

import click
import requests
from dotenv import load_dotenv

from portia import Portia
from portia.end_user import EndUser
from portia.config import Config
from portia.logger import logger
from portia.tool_registry import McpToolRegistry
from portia.mcp_session import StdioMcpClientConfig

if TYPE_CHECKING:
    pass


class SmartWorkflowGenerator:
    """Smart n8n workflow generator that uses AI to create workflows from natural language."""

    def __init__(self):
        """Initialize the workflow generator."""
        load_dotenv()

        # Check for required environment variables
        self.n8n_api_url = os.getenv("N8N_API_URL")
        self.n8n_api_key = os.getenv("N8N_API_KEY")

        if not self.n8n_api_url or not self.n8n_api_key:
            logger().error(
                "Missing required environment variables: N8N_API_URL and N8N_API_KEY"
            )
            click.echo("❌ Please set N8N_API_URL and N8N_API_KEY in your .env file")
            sys.exit(1)

    def analyze_workflow_with_ai(self, description: str) -> Dict[str, Any]:
        """Use AI to analyze the workflow description and generate appropriate nodes."""

        try:
            # Initialize Portia for AI analysis
            config = Config.from_default()
            portia = Portia(config=config)

            # Create AI prompt for workflow analysis
            analysis_prompt = f"""
            Analyze this workflow description and generate a complete n8n workflow JSON structure:
            
            Description: "{description}"
            
            Your task:
            1. Determine the appropriate trigger node based on the description
            2. Identify what action nodes are needed
            3. Create the complete n8n workflow JSON structure
            
            Available n8n node types and their use cases:
            - n8n-nodes-base.manualTrigger: For manual execution
            - n8n-nodes-base.cron: For scheduled execution (every X minutes/hours/days)
            - n8n-nodes-base.webhook: For HTTP webhook triggers
            - n8n-nodes-base.httpRequest: For making API calls
            - n8n-nodes-base.emailSend: For sending emails
            - n8n-nodes-base.telegram: For Telegram messages
            - n8n-nodes-base.slack: For Slack messages
            - n8n-nodes-base.set: For data processing and variables
            - n8n-nodes-base.if: For conditional logic
            
            Examples of parameters:
            - Cron trigger: {{"rule": {{"interval": [{{"field": "hours", "value": 2}}]}}}}
            - Telegram: {{"chatId": "YOUR_CHAT_ID", "text": "Your message here"}}
            - Email: {{"toEmail": "user@example.com", "subject": "Subject", "text": "Body"}}
            - HTTP: {{"url": "https://api.example.com", "method": "POST"}}
            - Set data: {{"values": {{"string": [{{"name": "key", "value": "value"}}]}}}}
            
            Generate ONLY a valid JSON workflow structure like this:
            {{
                "name": "descriptive-workflow-name",
                "nodes": [
                    {{
                        "parameters": {{}},
                        "type": "n8n-nodes-base.manualTrigger",
                        "typeVersion": 1,
                        "position": [20, 20],
                        "id": "trigger-node",
                        "name": "Manual Trigger"
                    }},
                    {{
                        "parameters": {{
                            "values": {{
                                "string": [
                                    {{
                                        "name": "message",
                                        "value": "Generated based on: {description}"
                                    }}
                                ]
                            }}
                        }},
                        "type": "n8n-nodes-base.set",
                        "typeVersion": 1,
                        "position": [240, 20],
                        "id": "action-node",
                        "name": "Process Data"
                    }}
                ],
                "connections": {{
                    "Manual Trigger": {{
                        "main": [
                            [
                                {{
                                    "node": "Process Data",
                                    "type": "main",
                                    "index": 0
                                }}
                            ]
                        ]
                    }}
                }},
                "settings": {{
                    "executionOrder": "v1"
                }}
            }}
            
            Important: 
            - Choose the RIGHT trigger type based on the description
            - Choose the RIGHT action nodes based on what the user wants to do
            - Use proper node parameters for the specific use case
            - Connect nodes properly
            - Return ONLY valid JSON, no explanations
            """

            # Run AI analysis using Portia with Google
            end_user = EndUser(external_id="workflow-analyzer")
            result = portia.run(query=analysis_prompt, end_user=end_user)

            if not result.outputs.final_output:
                raise ValueError("No AI response received from Portia")

            ai_response = result.outputs.final_output.get_value()

            if ai_response:
                # Try to parse the AI response as JSON
                try:
                    # Clean up the response (remove any markdown or extra text)
                    json_start = ai_response.find("{")
                    json_end = ai_response.rfind("}") + 1

                    if json_start >= 0 and json_end > json_start:
                        json_str = ai_response[json_start:json_end]
                        workflow_data = json.loads(json_str)

                        # Clean the workflow data for n8n API
                        cleaned_workflow = self.clean_workflow_for_n8n(workflow_data)

                        click.echo(
                            f"🤖 AI generated workflow with {len(cleaned_workflow.get('nodes', []))} nodes"
                        )
                        return cleaned_workflow
                    else:
                        raise ValueError("No valid JSON found in AI response")

                except (json.JSONDecodeError, ValueError) as e:
                    click.echo(f"⚠️ AI response parsing failed: {e}")
                    if ai_response:
                        click.echo(f"AI Response: {ai_response[:200]}...")
                    # Fallback to simple workflow
                    return self.create_fallback_workflow(description)
            else:
                click.echo("⚠️ No AI response received")
                return self.create_fallback_workflow(description)

        except Exception as e:
            click.echo(f"⚠️ AI analysis failed: {e}")
            return self.create_fallback_workflow(description)

    def clean_workflow_for_n8n(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean workflow data to remove properties that n8n API doesn't accept."""

        # Only keep the essential properties that n8n expects
        cleaned = {
            "name": workflow_data.get("name", "ai-workflow"),
            "nodes": [],
            "connections": {},
            "settings": workflow_data.get("settings", {"executionOrder": "v1"}),
        }

        # Clean each node to only include valid properties
        for node in workflow_data.get("nodes", []):
            cleaned_node = {
                "parameters": node.get("parameters", {}),
                "type": node.get("type", "n8n-nodes-base.set"),
                "typeVersion": node.get("typeVersion", 1),
                "position": node.get("position", [20, 20]),
                "id": node.get("id", f"node-{len(cleaned['nodes'])}"),
                "name": node.get("name", "Unnamed Node"),
            }
            cleaned["nodes"].append(cleaned_node)

        # Fix connections format - convert array to object if needed
        connections = workflow_data.get("connections", {})
        if isinstance(connections, list):
            # Convert array format to object format
            new_connections = {}
            for conn in connections:
                # Handle both simple format and nested format
                if "from" in conn and "to" in conn:
                    # Nested format: {"from": {"node": "X"}, "to": {"node": "Y"}}
                    from_node = (
                        conn["from"].get("node")
                        if isinstance(conn["from"], dict)
                        else conn["from"]
                    )
                    to_node = (
                        conn["to"].get("node")
                        if isinstance(conn["to"], dict)
                        else conn["to"]
                    )
                else:
                    # Simple format: {"from": "node1", "to": "node2"}
                    from_node = conn.get("from")
                    to_node = conn.get("to")

                from_handle = conn.get("fromHandle", "main")
                to_handle = conn.get("toHandle", "main")
                to_index = conn.get("toIndex", 0)

                if from_node and to_node:
                    if from_node not in new_connections:
                        new_connections[from_node] = {}
                    if from_handle not in new_connections[from_node]:
                        new_connections[from_node][from_handle] = []

                    new_connections[from_node][from_handle].append(
                        {"node": to_node, "type": to_handle, "index": to_index}
                    )

            cleaned["connections"] = new_connections
        else:
            cleaned["connections"] = connections

        return cleaned

    def create_fallback_workflow(self, description: str) -> Dict[str, Any]:
        """Create a simple fallback workflow when AI analysis fails."""
        return {
            "name": "ai-generated-workflow",
            "nodes": [
                {
                    "parameters": {},
                    "type": "n8n-nodes-base.manualTrigger",
                    "typeVersion": 1,
                    "position": [20, 20],
                    "id": "manual-trigger",
                    "name": "Manual Trigger",
                },
                {
                    "parameters": {
                        "values": {
                            "string": [
                                {"name": "description", "value": description},
                                {
                                    "name": "status",
                                    "value": "AI-generated workflow (fallback)",
                                },
                            ]
                        }
                    },
                    "type": "n8n-nodes-base.set",
                    "typeVersion": 1,
                    "position": [240, 20],
                    "id": "process-data",
                    "name": "Process Data",
                },
            ],
            "connections": {
                "Manual Trigger": {
                    "main": [[{"node": "Process Data", "type": "main", "index": 0}]]
                }
            },
            "settings": {"executionOrder": "v1"},
        }

    def create_workflow_via_api(self, workflow_data: Dict[str, Any]) -> bool:
        """Create workflow directly via n8n API."""

        headers = {
            "X-N8N-API-KEY": self.n8n_api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.n8n_api_url}/workflows",
                headers=headers,
                json=workflow_data,
                timeout=10,
            )

            if response.status_code in [200, 201]:
                workflow = response.json()
                click.echo(f"✅ Workflow created successfully!")
                click.echo(f"🎯 ID: {workflow.get('id')}")
                click.echo(f"📝 Name: {workflow.get('name')}")
                if self.n8n_api_url:
                    click.echo(
                        f"🌐 View: {self.n8n_api_url.replace('/api/v1', '')}/workflow/{workflow.get('id')}"
                    )
                return True
            else:
                click.echo(f"❌ API Error: {response.status_code}")
                click.echo(f"Response: {response.text}")
                return False

        except Exception as e:
            click.echo(f"❌ Error: {e}")
            return False

    def start_interactive_workflow(self) -> None:
        """Start the AI-powered interactive workflow creation process."""
        click.echo("🧠 Welcome to Smart Portia n8n Workflow Generator!")
        click.echo("=" * 60)
        click.echo("💡 Describe your workflow and I'll create the right nodes!")
        click.echo(
            "🔧 Supported: Cron/Schedule, Telegram, Email, Slack, HTTP, Webhooks"
        )

        # Get workflow name
        workflow_name = click.prompt(
            "📝 What would you like to name your workflow?",
            type=str,
            default="my-smart-workflow",
        )

        # Get workflow description with examples
        workflow_description = click.prompt(
            "📋 Describe what your workflow should do\n"
            + "💡 Examples:\n"
            + "   • 'Send me a Telegram message every 2 hours'\n"
            + "   • 'Make an HTTP request to my API daily'\n"
            + "   • 'Email me when webhook is triggered'\n"
            + "🎯 Your workflow",
            type=str,
        )

        click.echo("\n🤖 Analyzing your workflow requirements...")

        # Use AI to generate the workflow structure
        workflow_data = self.analyze_workflow_with_ai(workflow_description)

        # Update the name
        workflow_data["name"] = workflow_name

        # Show what will be created
        click.echo("📊 Workflow Structure:")
        click.echo(f"   🎯 Workflow: {workflow_data.get('name')}")
        click.echo(f"   📦 Nodes: {len(workflow_data.get('nodes', []))} total")
        for i, node in enumerate(workflow_data.get("nodes", [])):
            click.echo(
                f"      {i+1}. {node.get('name', 'Unnamed')} ({node.get('type', 'Unknown')})"
            )

        # Count connections safely
        connections = workflow_data.get("connections", {})
        if isinstance(connections, list):
            connection_count = len(connections)
        elif isinstance(connections, dict):
            connection_count = len(connections)
        else:
            connection_count = 0

        click.echo(f"   🔗 Connections: {connection_count} groups")

        # Ask for confirmation
        if not click.confirm("\n🚀 Create this workflow?", default=True):
            click.echo("❌ Workflow creation cancelled")
            return

        # Create via direct API call
        click.echo("\n🚀 Creating AI-generated workflow...")
        success = self.create_workflow_via_api(workflow_data)

        if success:
            click.echo("\n🎉 AI workflow creation complete!")
            click.echo(
                "💡 The AI analyzed your description and created appropriate nodes!"
            )
            click.echo("🔗 You can now run this workflow in n8n!")
        else:
            click.echo("\n❌ Failed to create AI workflow")
