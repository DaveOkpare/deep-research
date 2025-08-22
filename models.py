from pydantic import BaseModel
from typing import List


class Task(BaseModel):
    """Task to be assigned to a subagent"""

    description: str
    focus_area: str


class SubagentTasks(BaseModel):
    """List of tasks to run subagents for"""

    tasks: List[Task]


class ResearchSection(BaseModel):
    """A section of the research report"""

    title: str
    content: str
    subsections: List["ResearchSection"] = []


class ResearchReport(BaseModel):
    """Final research report from lead agent in markdown format"""

    title: str
    executive_summary: str
    sections: List[ResearchSection]
    key_takeaways: List[str]

    def to_markdown(self) -> str:
        """Convert the research report to markdown format"""
        md = f"# {self.title}\n\n"
        md += f"## Executive Summary\n\n{self.executive_summary}\n\n"

        for section in self.sections:
            md += f"## {section.title}\n\n{section.content}\n\n"
            for subsection in section.subsections:
                md += f"### {subsection.title}\n\n{subsection.content}\n\n"

        md += "## Key Takeaways\n\n"
        for i, takeaway in enumerate(self.key_takeaways, 1):
            md += f"{i}. {takeaway}\n"

        return md


class SubagentFindings(BaseModel):
    """Research findings from a subagent"""

    task_description: str
    summary: str
    key_insights: List[str]
    sources_found: int
    confidence_level: str  # "high", "medium", "low"
