"""Direct FeatureBench evaluator — runs agents on the host, verifies in Docker.

Three-step architecture:
1. Extract /testbed/ from Docker image to a local temp dir
2. Run factory agents DIRECTLY ON THE HOST against the extracted testbed
3. Copy the modified testbed into a fresh container via docker cp + exec
   (avoids bind-mount cross-platform issues with amd64 images on arm64 hosts)

This avoids installing agents inside Docker containers entirely.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

from factory.outer_loop.models import EvalResult
from factory.workflow.primitives import AgentNode, ForkNode, GateNode, JoinNode, Workflow

log = structlog.get_logger()

_FEATUREBENCH_DIR = Path(__file__).resolve().parents[2] / "featurebench"
_PYTEST_F2P_RE = re.compile(r"pytest\s+(.+?)\s*>\s*/tmp/f2p_output")
_PYTEST_P2P_RE = re.compile(r"pytest\s+(.+?)\s*>\s*/tmp/p2p_output")
_INSTALL_RE = re.compile(r"#\s*Repo-specific install[^\n]*\n(pip install[^\n]+)")


def _parse_from_line(dockerfile: Path) -> str:
    """Extract the base image from a Dockerfile's FROM line."""
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            return stripped.split()[1]
    raise ValueError(f"No FROM line found in {dockerfile}")


def _parse_deleted_files(patch_path: Path) -> list[str]:
    """Parse file paths deleted by a diff (--- a/path lines in deleted-file hunks)."""
    deleted: list[str] = []
    if not patch_path.exists():
        return deleted
    text = patch_path.read_text()
    in_delete_block = False
    for line in text.splitlines():
        if line.startswith("deleted file"):
            in_delete_block = True
        elif line.startswith("diff --git"):
            in_delete_block = False
        elif in_delete_block and line.startswith("--- a/"):
            deleted.append(line[6:])
    return deleted


def _parse_test_sh(test_sh: Path) -> tuple[str | None, str | None, str]:
    """Extract F2P test args, P2P test args, and install command from test.sh."""
    text = test_sh.read_text()

    f2p_match = _PYTEST_F2P_RE.search(text)
    f2p_args = f2p_match.group(1).strip() if f2p_match else None

    p2p_match = _PYTEST_P2P_RE.search(text)
    p2p_args = p2p_match.group(1).strip() if p2p_match else None

    install_match = _INSTALL_RE.search(text)
    install_cmd = install_match.group(1).strip() if install_match else "pip install -e . || true"

    return f2p_args, p2p_args, install_cmd


def _topo_sort_nodes(workflow: Workflow) -> list[str]:
    """Topological sort of workflow nodes using Kahn's algorithm."""
    adj: dict[str, list[str]] = {nid: [] for nid in workflow.nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in workflow.nodes}
    for edge in workflow.edges:
        if edge.source in adj and edge.target in in_degree:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while queue:
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order


class DirectFeatureBenchEvaluator:
    """Evaluates workflows on FeatureBench without installing agents in containers.

    Implements the ``EvaluatorFn`` protocol::

        __call__(workflow, project_dir, instances) -> EvalResult
    """

    def __init__(
        self,
        featurebench_dir: Path | None = None,
        agent_timeout: int = 1800,
    ) -> None:
        self._featurebench_dir = featurebench_dir or _FEATUREBENCH_DIR
        self._agent_timeout = agent_timeout

    def __call__(
        self,
        workflow: Workflow,
        project_dir: str,
        instances: list[str],
    ) -> EvalResult:
        total = len(instances)
        per_instance: dict[str, object] = {}
        total_score = 0.0

        for instance_id in instances:
            partial = self._eval_instance(workflow, instance_id)
            per_instance[instance_id] = {
                "score": partial,
                "resolved": partial >= 1.0,
            }
            total_score += partial

        score = total_score / max(total, 1)
        per_scores: dict[str, float] = {}
        for k, v in per_instance.items():
            if isinstance(v, dict):
                per_scores[k] = float(v.get("score", 0.0))
        log.info(
            "direct_eval_done",
            score=score,
            total=total,
            per_instance_scores=per_scores,
        )
        return EvalResult(
            score=score,
            benchmark_score=score,
            complexity=float(len(workflow.nodes)),
            details={"instances": per_instance},
        )

    def _eval_instance(self, workflow: Workflow, instance_id: str) -> float:
        """Evaluate a single FeatureBench instance. Returns partial credit [0.0, 1.0]."""
        task_dir = self._featurebench_dir / instance_id
        if not task_dir.exists():
            log.error("task_dir_missing", instance=instance_id)
            return 0.0

        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.exists():
            log.error("dockerfile_missing", instance=instance_id)
            return 0.0

        image = _parse_from_line(dockerfile)
        workdir = Path(tempfile.mkdtemp(prefix=f"fb-{instance_id[:30]}-", dir="/tmp"))

        try:
            # 1. Pull image if needed
            log.info("pulling_image", image=image, instance=instance_id)
            subprocess.run(
                ["docker", "pull", "--platform", "linux/amd64", image],
                capture_output=True,
                text=True,
                timeout=600,
            )

            # 2. Extract /testbed/ from Docker image
            log.info("extracting_testbed", instance=instance_id)
            cid_result = subprocess.run(
                ["docker", "create", "--platform", "linux/amd64", image],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if cid_result.returncode != 0:
                log.error("docker_create_failed", stderr=cid_result.stderr, instance=instance_id)
                return 0.0

            cid = cid_result.stdout.strip()
            try:
                cp_result = subprocess.run(
                    ["docker", "cp", f"{cid}:/testbed", str(workdir / "testbed")],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if cp_result.returncode != 0:
                    log.error("docker_cp_failed", stderr=cp_result.stderr, instance=instance_id)
                    return 0.0
            finally:
                subprocess.run(["docker", "rm", cid], capture_output=True, timeout=30)

            testbed = workdir / "testbed"

            # 3. Initialize git in testbed if not already a repo
            if not (testbed / ".git").exists():
                subprocess.run(["git", "init"], cwd=testbed, capture_output=True, timeout=30)
                subprocess.run(["git", "add", "."], cwd=testbed, capture_output=True, timeout=60)
                subprocess.run(
                    ["git", "commit", "-m", "initial"],
                    cwd=testbed,
                    capture_output=True,
                    timeout=60,
                    env={"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test",
                         "PATH": "/usr/bin:/bin:/usr/local/bin"},
                )

            # 4. Apply setup_patch (scramble the implementation)
            setup_patch = task_dir / "environment" / "setup_patch.diff"
            if setup_patch.exists() and setup_patch.stat().st_size > 0:
                log.info("applying_setup_patch", instance=instance_id)
                subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", str(setup_patch)],
                    cwd=testbed,
                    capture_output=True,
                    timeout=30,
                )

                # Delete test files listed in test_patch.diff (lv1)
                test_patch = task_dir / "environment" / "test_patch.diff"
                deleted_files = _parse_deleted_files(test_patch)
                for f in deleted_files:
                    target = testbed / f
                    if target.exists():
                        target.unlink()
                        log.debug("deleted_test_file", file=f, instance=instance_id)

            # 5. Copy instruction.md to testbed
            instruction = task_dir / "instruction.md"
            if instruction.exists():
                shutil.copy(instruction, testbed / "task-instruction.md")

            # 6. Create .factory dir for agent output
            factory_dir = testbed / ".factory"
            factory_dir.mkdir(exist_ok=True)
            (factory_dir / "reviews").mkdir(exist_ok=True)

            # 7. Run the workflow's agents on the testbed
            log.info("running_agents", instance=instance_id, nodes=len(workflow.nodes))
            self._run_workflow_agents(workflow, testbed)

            # 8. Verify: run test.sh inside Docker with the modified testbed mounted
            log.info("verifying_in_docker", instance=instance_id)
            partial_score = self._verify_in_docker(task_dir, image, testbed)
            log.info(
                "instance_result",
                instance=instance_id,
                partial_score=partial_score,
            )
            return partial_score

        except subprocess.TimeoutExpired:
            log.warning("instance_timeout", instance=instance_id)
            return 0.0
        except Exception as exc:
            log.error("instance_error", instance=instance_id, error=str(exc))
            return 0.0
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _run_workflow_agents(self, workflow: Workflow, testbed: Path) -> None:
        """Run workflow agents in topological order on the testbed."""
        order = _topo_sort_nodes(workflow)
        for node_id in order:
            node = workflow.nodes[node_id]
            if isinstance(node, AgentNode):
                timeout = node.timeout or self._agent_timeout
                prompt = node.prompt_template
                if not prompt:
                    continue

                log.info("running_agent", node=node_id, role=node.role.value, timeout=timeout)
                result = subprocess.run(
                    [
                        "factory",
                        "agent",
                        node.role.value,
                        "--task",
                        prompt,
                        "--project",
                        str(testbed),
                        "--timeout",
                        str(timeout),
                        "--disallowedTools",
                        "WebSearch,WebFetch",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout + 120,
                )
                log.info(
                    "agent_finished",
                    node=node_id,
                    returncode=result.returncode,
                    stdout_len=len(result.stdout),
                )
            elif isinstance(node, (GateNode, ForkNode, JoinNode)):
                pass

    def _verify_in_docker(
        self, task_dir: Path, image: str, testbed: Path
    ) -> float:
        """Run pytest via docker cp + exec — returns partial credit [0.0, 1.0]."""
        test_patch = task_dir / "environment" / "test_patch.diff"
        test_sh = task_dir / "tests" / "test.sh"

        f2p_args, p2p_args, install_cmd = (None, None, "pip install -e . || true")
        if test_sh.exists():
            f2p_args, p2p_args, install_cmd = _parse_test_sh(test_sh)

        # Restore deleted test files into the host testbed before copying to container
        test_files = _parse_deleted_files(test_patch)
        if test_files:
            if test_patch.exists() and test_patch.stat().st_size > 0:
                apply_result = subprocess.run(
                    ["git", "apply", "--reverse", "--whitespace=nowarn", str(test_patch)],
                    cwd=testbed,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if apply_result.returncode != 0:
                    log.warning(
                        "reverse_patch_failed",
                        stderr=apply_result.stderr,
                        task_dir=str(task_dir),
                    )
            f2p_cmd = f"pytest -rA --tb=short --color=no {' '.join(test_files)}"
        elif f2p_args:
            f2p_cmd = f"pytest -rA --tb=short --color=no {f2p_args}"
        else:
            log.error("no_test_target", task_dir=str(task_dir))
            return 0.0

        # 1. Create container (kept alive with sleep so we can exec into it)
        cid_result = subprocess.run(
            [
                "docker", "create", "--platform", "linux/amd64",
                image,
                "bash", "-c", "sleep 600",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if cid_result.returncode != 0:
            log.error("docker_create_verify_failed", stderr=cid_result.stderr)
            return 0.0
        cid = cid_result.stdout.strip()

        try:
            # 2. Copy only changed files into the container (avoids symlink conflicts
            #    where docker cp fails with "cannot overwrite directory with non-directory")
            diff_result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=testbed,
                capture_output=True,
                text=True,
                timeout=30,
            )
            changed_files: list[str] = []
            if diff_result.returncode == 0:
                changed_files.extend(f for f in diff_result.stdout.strip().splitlines() if f)

            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=testbed,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if untracked_result.returncode == 0:
                changed_files.extend(f for f in untracked_result.stdout.strip().splitlines() if f)

            log.info("copying_changed_files", count=len(changed_files), task_dir=str(task_dir))

            # Start container first so we can mkdir for new files
            start_result = subprocess.run(
                ["docker", "start", cid],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if start_result.returncode != 0:
                log.error("docker_start_failed", stderr=start_result.stderr)
                return 0.0

            parents_ensured: set[str] = set()
            for rel_path in changed_files:
                src = testbed / rel_path
                if not src.exists() or not src.is_file():
                    continue
                parent = str(Path(rel_path).parent)
                if parent and parent != "." and parent not in parents_ensured:
                    subprocess.run(
                        ["docker", "exec", cid, "mkdir", "-p", f"/testbed/{parent}"],
                        capture_output=True,
                        timeout=10,
                    )
                    parents_ensured.add(parent)
                cp_result = subprocess.run(
                    ["docker", "cp", str(src), f"{cid}:/testbed/{rel_path}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if cp_result.returncode != 0:
                    log.warning("docker_cp_file_failed", file=rel_path, stderr=cp_result.stderr)

            # 3. Exec the test inside the container
            script = (
                f"source /opt/miniconda3/bin/activate testbed; "
                f"cd /testbed; "
                f"{install_cmd} 2>&1 | tail -2; "
                f"{f2p_cmd}"
            )
            if p2p_args:
                script += f"; pytest -rA --tb=short --color=no {p2p_args}"

            result = subprocess.run(
                ["docker", "exec", cid, "bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=600,
            )

            log.info(
                "docker_verify_done",
                returncode=result.returncode,
                stdout_tail=result.stdout[-500:] if result.stdout else "",
                stderr_tail=result.stderr[-500:] if result.stderr else "",
            )

            if result.returncode == 0:
                return 1.0

            from factory.outer_loop.benchmark_evaluator import parse_pytest_stdout
            metrics = parse_pytest_stdout(result.stdout or "")
            return metrics.get("pass_rate", 0.0)
        finally:
            # 4. Cleanup: force-remove the container
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                timeout=30,
            )
