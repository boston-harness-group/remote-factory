"""Tests for DesignerAgent execution_strategy awareness."""

from __future__ import annotations

from factory.outer_loop.designer import DesignerAgent
from factory.workflow.primitives import (
    AgentNode,
)


class TestDesignerExecutorStrategy:
    """When execution_strategy='executor', Designer populates prompt_template and PROCEED edges."""

    def test_minimal_executor_has_prompt_template(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test", execution_strategy="executor")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"AgentNode '{nid}' missing prompt_template"

    def test_thorough_executor_has_prompt_template(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test", execution_strategy="executor")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"AgentNode '{nid}' missing prompt_template"

    def test_custom_executor_has_prompt_template(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_custom("test", {"max_nodes": 5}, execution_strategy="executor")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"AgentNode '{nid}' missing prompt_template"

    def test_prompt_template_includes_project_path(self) -> None:
        """Generated prompt_template should include {project_path} placeholder."""
        designer = DesignerAgent()
        wf = designer.design_minimal("test", execution_strategy="executor")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert "{project_path}" in node.prompt_template


class TestDesignerCeoStrategy:
    """Designer populates prompt_template for all execution strategies."""

    def test_minimal_ceo_skill_has_prompt_template(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test", execution_strategy="ceo-skill")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template != "", f"AgentNode '{nid}' should have non-empty prompt_template"

    def test_minimal_ceo_tool_has_prompt_template(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test", execution_strategy="ceo-tool")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template != "", f"AgentNode '{nid}' should have non-empty prompt_template"

    def test_thorough_ceo_skill_has_prompt_template(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test", execution_strategy="ceo-skill")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template != "", f"AgentNode '{nid}' should have non-empty prompt_template"


class TestDesignerDefaultStrategy:
    """Calling design_*() without execution_strategy defaults to 'executor'."""

    def test_default_is_executor_minimal(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"Default should populate prompt_template for '{nid}'"

    def test_default_is_executor_thorough(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test")
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"Default should populate prompt_template for '{nid}'"

    def test_default_is_executor_custom(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_custom("test", {"max_nodes": 5})
        for nid, node in wf.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"Default should populate prompt_template for '{nid}'"
