"""Mechanical DataNode wrapping for Builder-produced subgraph workflows.

Takes a plain workflow (the per-item subgraph) and wraps it in a DataNode
+ JoinNode, making the DataNode the new start_node.  Real edges connect
DataNode → branch entry and branch exit → JoinNode.  The wrapping is
purely structural — no LLM involvement, no source resolution.  The data
source is either set via task_ref (when --task is provided) or left
late-bound (when --data-node is used alone).
"""

from __future__ import annotations

import structlog

from factory.workflow.primitives import DataNode, Edge, JoinNode, Workflow

log = structlog.get_logger()


def wrap_with_data_node(
    workflow: Workflow,
    *,
    task_ref: str | None = None,
    data_node_id: str = "data",
    join_node_id: str = "_join_data",
    parallelism: int = 1,
) -> Workflow:
    """Wrap a Builder-produced subgraph workflow with DataNode + JoinNode.

    Args:
        workflow: The Builder's output — a plain workflow with no DataNode.
                  Must have exactly one terminal node (no outgoing edges).
        task_ref: Optional task reference string (module:Class or name).
                  When provided, set as DataNode.task_ref.
                  When None, the DataNode is late-bound (no source — resolved at runtime).
        data_node_id: ID for the created DataNode. Defaults to "data".
                      If this collides with an existing node ID, "_data" is tried.
        join_node_id: ID for the created JoinNode. Defaults to "_join_data".
        parallelism: DataNode parallelism setting. Defaults to 1.

    Returns:
        A new Workflow with the DataNode as start_node, real edges connecting
        DataNode → branch entry → ... → branch exit → JoinNode.

    Raises:
        ValueError: If the workflow has zero nodes, multiple terminal nodes,
                    or the data_node_id collides after fallback attempts.
    """
    if not workflow.nodes:
        raise ValueError("Cannot wrap an empty workflow with a DataNode")

    # Step 1: Identify the branch entry point (the original start_node)
    branch_entry = workflow.start_node
    if branch_entry not in workflow.nodes:
        raise ValueError(
            f"start_node {branch_entry!r} not found in workflow nodes"
        )

    # Step 2: Find the terminal node (node with no outgoing edges)
    sources = {e.source for e in workflow.edges}
    all_ids = set(workflow.nodes.keys())
    terminal_candidates = all_ids - sources

    if len(terminal_candidates) == 0:
        raise ValueError(
            "No terminal node found (every node has outgoing edges). "
            "The subgraph must have exactly one terminal node."
        )
    if len(terminal_candidates) > 1:
        # If the entry node is among candidates (single-node workflow), it's the terminal
        if len(workflow.nodes) == 1:
            terminal_candidates = {branch_entry}
        else:
            raise ValueError(
                f"Multiple terminal nodes found: {terminal_candidates}. "
                "The subgraph must have exactly one terminal node."
            )
    branch_exit = next(iter(terminal_candidates))

    # Step 3: Handle ID collision
    chosen_id = data_node_id
    if chosen_id in workflow.nodes:
        chosen_id = f"_{data_node_id}"
        if chosen_id in workflow.nodes:
            raise ValueError(
                f"DataNode ID collision: both {data_node_id!r} and {chosen_id!r} "
                f"already exist in the workflow nodes."
            )
        log.info(
            "datanode_wrapping.id_collision_resolved",
            original=data_node_id,
            resolved=chosen_id,
        )

    # Step 4: Handle JoinNode ID collision
    chosen_join_id = join_node_id
    if chosen_join_id in workflow.nodes or chosen_join_id == chosen_id:
        chosen_join_id = f"_join_{chosen_id}"
        if chosen_join_id in workflow.nodes:
            raise ValueError(
                f"JoinNode ID collision: {chosen_join_id!r} "
                f"already exists in the workflow nodes."
            )

    # Step 5: Create DataNode and JoinNode
    data_node_kwargs: dict = {
        "id": chosen_id,
        "parallelism": parallelism,
    }
    if task_ref is not None:
        data_node_kwargs["task_ref"] = task_ref

    data_node = DataNode(**data_node_kwargs)
    join_node = JoinNode(id=chosen_join_id, sources=[branch_exit])

    # Step 6: Assemble the wrapped workflow with real edges
    new_nodes = dict(workflow.nodes)
    new_nodes[chosen_id] = data_node
    new_nodes[chosen_join_id] = join_node

    new_edges = list(workflow.edges)
    # DataNode → branch entry
    new_edges.append(Edge(source=chosen_id, target=branch_entry))
    # branch exit → JoinNode
    new_edges.append(Edge(source=branch_exit, target=chosen_join_id))

    wrapped = Workflow(
        name=workflow.name,
        nodes=new_nodes,
        edges=new_edges,
        start_node=chosen_id,
        task=task_ref,  # Set workflow.task for outer-loop compatibility
    )

    log.info(
        "datanode_wrapping.complete",
        data_node_id=chosen_id,
        join_node_id=chosen_join_id,
        branch_entry=branch_entry,
        branch_exit=branch_exit,
        task_ref=task_ref or "(late-bound)",
        node_count=len(new_nodes),
    )

    return wrapped
