# My Portia Hackathon Project

## Setup
Your environment is ready with:
- **Google Gemini 1.5 Flash** (working)
- **Portia API** (working)
- **OpenAI API** (has quota issues)

## Basic Usage

```python
from portia import Portia
from portia.tool_registry import ToolRegistry
from portia.config import Config, GenerativeModelsConfig
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.llm_tool import LLMTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Portia instance
tools = [CalculatorTool(), LLMTool()]
config = Config(models=GenerativeModelsConfig(default_model="google/gemini-1.5-flash"))
portia = Portia(config=config, tools=ToolRegistry(tools))

# Use it
result = portia.run("Your query here")
print(result.outputs.final_output)
```

## Available Tools
- **CalculatorTool** - Math calculations
- **LLMTool** - General reasoning and knowledge
- **FileReaderTool** - Read files
- **FileWriterTool** - Write files
- **WeatherTool** - Weather (needs OPENWEATHERMAP_API_KEY)
- **SearchTool** - Web search (needs TAVILY_API_KEY)

## Dashboard
Track your runs at: https://app.portialabs.ai

---
*Ready to build something amazing!*
