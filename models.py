from pydantic import BaseModel
from typing import List


class Task(BaseModel):
    """Task to be assigned to a subagent"""

    description: str
    focus_area: str


class SubagentTasks(BaseModel):
    """List of tasks to run subagents for"""

    tasks: List[Task]


class ResearchReport(BaseModel):
    """Final research report from lead agent"""

    executive_summary: str
    key_findings: List[str]
    detailed_analysis: str
    recommendations: List[str]


class SubagentFindings(BaseModel):
    """Research findings from a subagent"""

    task_description: str
    summary: str
    key_insights: List[str]
    sources_found: int
    confidence_level: str  # "high", "medium", "low"
