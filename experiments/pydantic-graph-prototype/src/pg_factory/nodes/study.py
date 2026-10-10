"""Study chain nodes — port of definitions.py _study_subgraph (lines 108-158).

Four BaseNode subclasses forming a linear chain:
    GraphUpdateNode → StudyNode → GraphExplorerNode → ConcatStudyNode

Structural difference from the original:
    definitions.py constructs a dict[str, Node] + list[Edge] where edges are explicit
    data objects wired by string IDs. In pydantic-graph, edges are implicit — each
    node's run() return type annotation declares its successor, and Graph infers the
    topology at construction time. This eliminates an entire class of wiring bugs
    (misspelled edge targets, dangling nodes) at the cost of requiring all successor
    types to be importable at definition time.
"""

import asyncio

from pydantic_graph import BaseNode, End, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


class ConcatStudyNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Concatenates observation files into study-combined.md.

    Maps from: FnNode(id="concat_study", command="cat ... > study-combined.md")
    """

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> End[HaltResult]:
        pp = ctx.deps.project_path
        obs = pp / ".factory" / "strategy" / "observations.md"
        graph_ctx = pp / ".factory" / "strategy" / "graph-context.md"
        combined = pp / ".factory" / "strategy" / "study-combined.md"

        if ctx.deps.dry_run:
            output = ""
            for f in (obs, graph_ctx):
                if f.exists():
                    output += f.read_text()
                else:
                    output += f"[mock] {f.name} content\n"
            combined.parent.mkdir(parents=True, exist_ok=True)
            combined.write_text(output)
        else:
            cmd = f"cat {obs} {graph_ctx} > {combined}"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

        ctx.state.node_outputs["ConcatStudyNode"] = str(combined)
        ctx.state.events.append(
            {"node": "ConcatStudyNode", "action": "concat", "output": str(combined)}
        )
        return End(HaltResult(reason="study_complete"))


class GraphExplorerNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Simulates agent invocation for graph exploration.

    Maps from: AgentNode(id="graph_explorer", role=RESEARCHER)
    In dry_run mode, writes mock graph-context output instead of invoking an agent.
    """

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> ConcatStudyNode:
        pp = ctx.deps.project_path
        output_path = pp / ".factory" / "strategy" / "graph-context.md"

        if ctx.deps.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "# Graph Context\n\nMock graph exploration output for dry_run mode.\n"
            )
            result = "dry_run: mock graph-context written"
        else:
            proc = await asyncio.create_subprocess_shell(
                f"factory agent researcher --task 'explore graph' --project {pp}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            result = stdout.decode()

        ctx.state.node_outputs["GraphExplorerNode"] = result
        ctx.state.events.append(
            {"node": "GraphExplorerNode", "action": "explore", "output": result}
        )
        return ConcatStudyNode()


class StudyNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Runs 'factory study {project_path}'.

    Maps from: Study(id="study", command="factory study {project_path}")
    """

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> GraphExplorerNode:
        pp = ctx.deps.project_path

        if ctx.deps.dry_run:
            obs_path = pp / ".factory" / "strategy" / "observations.md"
            obs_path.parent.mkdir(parents=True, exist_ok=True)
            obs_path.write_text("# Observations\n\nMock study output for dry_run mode.\n")
            result = "dry_run: mock observations written"
        else:
            proc = await asyncio.create_subprocess_shell(
                f"factory study {pp}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            result = stdout.decode()

        ctx.state.node_outputs["StudyNode"] = result
        ctx.state.events.append(
            {"node": "StudyNode", "action": "study", "output": result}
        )
        return GraphExplorerNode()


class GraphUpdateNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Runs 'factory graph update {project_path}'.

    Maps from: FnNode(id="graph_update", command="factory graph update {project_path}")
    """

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> StudyNode:
        pp = ctx.deps.project_path

        if ctx.deps.dry_run:
            result = "dry_run: graph update skipped"
        else:
            proc = await asyncio.create_subprocess_shell(
                f"factory graph update {pp}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            result = stdout.decode()

        ctx.state.node_outputs["GraphUpdateNode"] = result
        ctx.state.events.append(
            {"node": "GraphUpdateNode", "action": "graph_update", "output": result}
        )
        return StudyNode()
