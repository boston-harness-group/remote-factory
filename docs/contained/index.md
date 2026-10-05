# Contained Runtimes

`factory contained` runs any factory command somewhere other than your shell — in a podman container
on your machine, in a pod on an OpenShift cluster, or in an OpenShell sandbox.

```bash
factory contained -- ceo ~/code/my-project
```

Two things make it worth using. The run happens against a **pinned toolchain** — a known Python, a
known set of agent CLIs, a known set of build tools — rather than whatever your machine has
accumulated. And it works on a **copy** of your project, so your working tree is never modified.

The copy is a git worktree of your repository, which means two things survive a run on purpose: the
copy itself, holding whatever the run produced, and a `contained/<name>` branch pointing at it.
`rm` prints the two commands that remove both once you are done with them.

Everything after `--` is handed inward **verbatim**. The runtime is a place to run the factory, not
a mode of it, so the host never parses what you pass and cannot break when the CLI grows.

!!! warning "Read the guarantees before trusting them"
    `contained` bounds *accidents* and gives runs a reproducible environment. It does **not** confine
    agent-authored code, it is **not** a multi-tenant boundary, and it does **not** replace review.
    See [What it does and does not protect you from](#what-it-does-and-does-not-protect-you-from).

---

## Quick start

```bash
factory contained setup                  # pull the image, check prerequisites
factory contained verify                 # report what's missing, with the fix for each
factory contained -- ceo ~/code/my-project
factory contained ls                     # what's running
factory contained attach <name>          # watch it; Ctrl-b d detaches, the run continues
factory contained sync <name>            # how to get the work back
factory contained rm <name>              # tear it down
```

You need `podman` (with its machine running on macOS), and inference credentials — an
`ANTHROPIC_API_KEY`, a Vertex configuration, or a credential profile in `~/.factory/config.toml`.

---

## Choosing a target

| | `--target local` | `--target k8s` | `--target openshell` |
|---|---|---|---|
| Where it runs | a podman container on your machine | a pod on a Kubernetes/OpenShift cluster | an OpenShell sandbox, driven by a gateway you run |
| Good for | everyday work; attaching and watching | long unattended runs; more CPU and memory than a laptop | runs whose **input you do not trust** |
| Needs | podman | a namespace, and a one-time setup you apply yourself | the `openshell` CLI, a local gateway, and a `claude-code` provider |
| Your project | a copy, bind-mounted from disk | a copy, uploaded to a volume that outlives the pod | a copy, uploaded into the sandbox |
| Credentials | taken from your shell, and they enter the container | a Secret you create in the namespace | **never enter the sandbox** — held by the gateway, attached as a provider |
| Egress | normal network access | normal network access from the pod | **deny-by-default**, kernel-enforced allowlist |
| Survives a laptop closing | no | yes | as long as the gateway's machine stays up |

### What it does and does not protect you from

`contained` exists to make runs **reproducible** and to keep them **off your working tree**. All
targets do that well.

The local and cluster targets are **not security sandboxes**, and it is worth being concrete about
what that means:

- The agent's code runs with normal network access and can reach anything your machine can. Nothing
  restricts what it writes or fetches.
- Locally, your inference credentials are inside the container, because the agent needs them to work.
- A contained run does not make its diff safe to merge. Review the result exactly as you would
  review any other change.
- Neither target is built for running code you do not trust, or for sharing a machine or namespace
  with people you do not trust.

The cluster target is the more constrained of the two — it runs under a restricted security context
with namespace-scoped permissions — but the point above still stands for both.

### The openshell target is different, by design

The openshell target (experimental) exists for the case the other two exclude: running the factory
on input you do not trust — an arbitrary GitHub issue, a spec from a stranger, a codebase you were
sent. Inside an OpenShell sandbox, agent-authored code is confined at the **kernel level**:

- **Filesystem**: Landlock allowlists. The workspace is read-write; nothing else is writable.
- **Network**: deny-by-default. Every connection is intercepted and decided by a supervisor against
  the policy — `uv` may reach PyPI, the `claude-code` provider allows the inference endpoint, and
  nothing else gets out.
- **Credentials**: your API key never enters the sandbox. The gateway holds it and resolves it only
  for requests the provider allows, so a key bound to `api.anthropic.com` cannot be exfiltrated
  anywhere else. What that does *not* mean: OpenShell's binary rules match parent processes, so
  every command the agent's Bash tool runs — a descendant of `claude` — inherits claude's egress
  and can *use* the placeholder against the provider's endpoint. The key cannot be extracted, but
  it can be proxied by confined code; that is the documented floor of the confinement, and the
  reason the provider profile the factory ships allows `api.anthropic.com` only (never the stock
  profile, whose `statsig`/`sentry` endpoints would be exfiltration channels) and the reason the
  allowlist stays small.

There is no skip-permissions flag to abuse: enforcement is in the kernel, not in the agent CLI's own
prompting, so it fails closed no matter what the model tries.

What it still does **not** do: the gateway host itself is trusted (it is part of the boundary, not
inside it); it is not a multi-tenant boundary between runs; and the run's diff is still yours to
review — a sandboxed run can still write a plausible-looking wrong change inside its workspace.

The default policy is deliberately small, and `--policy <file>` **replaces it entirely** — never
merges with it. The file you pass is the whole policy; nothing hidden can creep in through
default-and-override interaction. The default allowlist starts slightly loose (it includes read-only
PyPI access so eval environments can be built inside the sandbox) and is tightened based on how the
target gets used — loosening later is compatible, silently tightening would break running workflows.

---

## Command reference

```
factory contained [runtime flags] -- <any factory command>
factory contained {ls|attach|rm|sync|setup|verify|bundle|help} [name]
```

`help` prints the same text as `--help`, so whichever you reach for works.

**All targets**

| Flag | Default | Meaning |
|---|---|---|
| `--target local\|k8s\|openshell` | `local` | Which runtime |
| `--division` | off | Enable the container-manufacturing plane (local, k8s — not openshell) |
| `--name NAME` | derived | Runtime name |
| `--env KEY=VALUE` | — | Extra environment, repeatable |
| `--forward VAR` | — | Forward a named host variable, repeatable |
| `--image REF` | published default | Override the runtime image |
| `--yes` | off | Skip confirmations (`rm` of an active run, the secret-scan gate) |

**Local only**

| Flag | Meaning |
|---|---|
| `--mount PATH` | Additional host path bind-mounted in, repeatable |

**K8s only**

| Flag | Default | Meaning |
|---|---|---|
| `--namespace NS` | current context | Never hardcoded |
| `--context NAME` | your current one | Which kubeconfig context every cluster command uses |
| `--storage-class SC` | cluster default | Workspace PVC |

**Openshell only**

| Flag | Default | Meaning |
|---|---|---|
| `--policy PATH` | factory default | Replace the sandbox policy (full replacement, never merged) |
| `--gateway NAME` | the active one | Which registered OpenShell gateway to talk to |

A flag used against the wrong target fails at parse time naming the target it belongs to — never
silently ignored. Runtime flags go **before** the subcommand; anything flag-shaped after it is an
error rather than a name.

---

## Interaction examples

Transcripts below are from real runs, with `$HOME` shortened and the project renamed to
`my-project` throughout so nothing here reads as a required argument. They were captured
separately, so run names and ages differ between them.

### Checking prerequisites

`verify` reports; it changes nothing. Every failure carries the command that fixes it.

```console
$ factory contained verify
[FAIL] container_engine: podman is installed but its engine is not reachable: Cannot connect to Podman...
         fix: podman machine start
[FAIL] runtime_image: ghcr.io/akashgit/remote-factory/factory-runtime:latest is not present locally
         fix: factory contained setup   # pulls ghcr.io/akashgit/remote-factory/factory-runtime:latest
              or, if it is not published yet, point at one you have:
              export FACTORY_CONTAINED_IMAGE=<your-image>
[FAIL] inference: no inference configuration found: CLAUDE_CODE_USE_VERTEX is unset,
       ANTHROPIC_API_KEY is unset, and ~/.factory/config.toml defines no credential profiles
         fix: export ANTHROPIC_API_KEY=... and re-run with --forward ANTHROPIC_API_KEY, or
              configure Vertex (...), or add a [credentials.<name>] section to ~/.factory/config.toml

3 check(s) failed. `factory contained setup` can fix container_engine, runtime_image; the rest
need the fix shown above each one.
```

That is what a first run looks like on a machine with nothing set up. `setup` fixes the first two;
the third is yours, because the factory never handles credential material.

Inference is always reported by **shape** — which backend, which model, which variable or file
supplied it — and never by printing material:

`setup` runs as a numbered sequence, so it is always clear which step you are on and which one
stalled. On a terminal the step rules, the `[ ok ]` / `[FAIL]` marks and every resolved value are
coloured; piped or redirected, the same output is plain text.

```console
$ factory contained --target local setup

━━ 1/3  Container engine ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   The podman engine is not reachable. Starting the podman machine...

━━ 2/3  Runtime image ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Image already present: ghcr.io/akashgit/remote-factory/factory-runtime:latest

━━ 3/3  Result ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ ok ] container_engine: podman reachable (5.7.1, rootful)
[ ok ] runtime_image: ghcr.io/akashgit/remote-factory/factory-runtime:latest present locally
[ ok ] inference: Vertex, project my-project in us-east5, model <unset — pass --model in the
       payload>, credential from Application Default Credentials at ~/.config/gcloud

All checks passed. Start a run with `factory contained -- ceo <path>`.
```

Colour is navigation, not decoration, so it obeys the conventions you already have configured:
`NO_COLOR` turns it off, `FORCE_COLOR` turns it on through a pipe, and `TERM=dumb` is respected.

!!! note "If the image cannot be pulled"
    The runtime image is published by CI. If the pull fails, `setup` prints two ways forward: point
    `FACTORY_CONTAINED_IMAGE` at an image you already have, or build one from a checkout of the
    repository — the Containerfile ships in git, not in the installed package.

Run without `--target`, and at a terminal, `setup` asks which runtime you are preparing first:

```console
$ factory contained setup

━━ What are you setting up? ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Pass --target local, k8s or openshell to skip this question.

  1) local  a podman container on this machine
  2) k8s    a pod on a cluster
  3) both  local and cluster
  4) openshell  a policy-governed sandbox, for runs whose input you do not trust

Choice [1]:
```

`openshell` is option 4, appended rather than inserted, so `3` has always meant `both` and
still does.

Pass `--target local`, `--target k8s` or `--target openshell` to skip the question. `setup` is
idempotent — re-running changes nothing that is already correct, and it is the supported way to
repair a partial setup.

### Starting a run

The runtime's identifier is printed **first**, before any long-running work. A run whose name you
cannot see is a run you cannot manage.

```console
$ factory contained --name my-run -- backlog-list ~/code/my-project
Warning: no inference credentials are configured, so every agent call in this run will fail.
  Set one of these before running, and pass it inward:
    export ANTHROPIC_API_KEY=...   then add:  --forward ANTHROPIC_API_KEY
  Run `factory contained verify` to check.
Starting my-run
  attach:  factory contained attach my-run
  result:  factory contained sync my-run
  stop:    factory contained rm my-run

my-run is running.
```

That is the whole output. The command returns as soon as the run is going; the run itself continues
in tmux inside the container. Set `FACTORY_LOG_LEVEL=debug` if you want to see every command the
runtime issued.

### Watching, detaching, coming back

```console
$ factory contained ls
NAME                              TARGET  PROJECT       AGE   STATE
my-run                           local   e06e95065606  1s    running

$ factory contained attach my-run
```

`ls` covers both targets, but it only asks the cluster once you have actually used one — otherwise
a laptop that has only ever run locally would wait on a network timeout and then be told about a
cluster it never set up. `--target k8s ls` always asks.

That drops you into the live session. `Ctrl-b d` detaches and **leaves the run going** — the tmux
prefix, because the run lives in tmux precisely so that detaching is safe.

Typing `exit` is safe too. It ends the shell inside the session and returns you to your own
terminal; the session and everything it printed stay, and attaching again gives you a fresh shell in
the same window. `ls` shows such a run as `finished` rather than `running`, because the container
deliberately outlives the run inside it.

### Getting the work back

Nothing is ever merged for you.

```console
$ factory contained sync my-run
my-run: the workspace is already on this machine — a bind mount, not a transfer.
Work is on branch contained/my-run in ~/.factory-contained/my-run/my-project.
  Review:  git -C ~/.factory-contained/my-run/my-project status && git -C ... diff
  Merge:   git -C ~/code/my-project merge contained/my-run
```

### Tearing down

```console
$ factory contained rm my-run
my-run: deleted. Your work is kept — it is not removed with the runtime.
Work is on branch contained/my-run in ~/.factory-contained/my-run/my-project.
  Review:  git -C ~/.factory-contained/my-run/my-project status && git -C ... diff
  Merge:   git -C ~/code/my-project merge contained/my-run

This run left a git worktree and a branch in your repository. Remove them with:
  git -C ~/code/my-project worktree remove ~/.factory-contained/my-run/my-project
  git -C ~/code/my-project branch -D contained/my-run
```

The container **persists** until you remove it. Nothing is auto-reaped, because a failed run is
exactly when its state is worth reading. A launch that fails *before* the container exists cleans
its own workspace up, so only runs that actually started leave anything behind.

### When the workspace is wrong

Five assertions run between provisioning and the first agent call. A failure aborts **before** any
tokens are spent, names the likely cause, and leaves the runtime up so you can look:

```console
$ factory contained --name my-run -- ceo ~/code/my-project
contained: step 'assert:git_usable' failed
  The workspace is not a usable git repository inside the runtime.
  Most likely the repository this project belongs to was not mounted — a git worktree's .git is a
  file pointing at a directory elsewhere.
  Try:  factory contained --mount <path-to-that-repository> -- <your command>
  The container is still there for inspection:
    podman exec -it my-run sh
    factory contained rm my-run

This run left a git worktree and a branch in your repository. Remove them with:
  git -C ~/code/my-project worktree remove ~/.factory-contained/my-run/my-project
  git -C ~/code/my-project branch -D contained/my-run
```

Each hint names the likely cause and what to try. The container is left running so you can look
inside it before removing it.

### Composing without provisioning

`FACTORY_CONTAINED_DRY_RUN=1` prints the exact commands the real path would run, and provisions
nothing:

```console
$ FACTORY_CONTAINED_DRY_RUN=1 factory contained -- study ~/code/my-project
DRY RUN — my-run (ghcr.io/…/factory-runtime:latest); nothing is provisioned.
[create] podman run -d --init --name my-run --label factory.contained=true …
[assert:project_present] podman exec my-run sh -lc '[ -d "…" ] && [ -n "$(ls -A "…")" ]'
[assert:git_usable] podman exec my-run sh -lc 'git -C "…" status --porcelain >/dev/null 2>&1'
[assert:factory_state] podman exec my-run test -f …/.factory/config.json
[assert:writable] podman exec my-run sh -lc 'touch "…/.factory-write-probe" && rm -f …'
[assert:content_hash] podman exec my-run sh -lc 'sha256sum "…" | grep -q "^<digest> "'
[run] podman exec my-run sh -lc 'tmux new-session -d -s factory -c … '
      […the run line is ~45 lines: it embeds the Claude Code state seeding verbatim…]
```

Which assertions appear depends on what your project actually has: `factory_state` only when the
project has a `.factory/config.json`, `git_usable` only when it is a git repository.

The `[run]` line really is that long, and it will look like line noise. Dry-run's contract is to
print *the same commands the real path runs*, so it is not trimmed — a tidier rendering could drift
from what actually executes, which would defeat the point of previewing.

Secret-looking values are redacted anywhere a command is printed:

```console
$ FACTORY_CONTAINED_DRY_RUN=1 factory contained --forward GH_TOKEN -- study ~/code/my-project
… --env GH_TOKEN=<redacted> …
```

---

## The local division

`--division` gives the contained agent your **host's** podman engine, so it can build an image, run
it, read the failure and iterate.

Builds happen on your machine rather than inside the container: the container has no container
engine of its own, and nesting one inside it is not workable on macOS. That is why this is a
separate flag rather than something always on.

```console
$ factory contained --division --name buildcycle -- ceo ~/code/my-project --focus "add a Containerfile"

  ┌─ Container builds enabled (--division) ───────────────────────────────────────
  │ Started podman-mcp-server so the agent can build and run container images.
  │ The run reaches it at http://host.containers.internal:8430/mcp
  │
  │ It listens on 0.0.0.0:8430 — every network interface, not just this
  │ machine — and it has no authentication. For as long as the run lasts, anyone
  │ who can reach that port can build and run containers as you.
  │
  │ Avoid --division on untrusted networks.
  │ It stops when the run is removed:
  │     factory contained rm buildcycle
  └───────────────────────────────────────────────────────────────────────────────

Starting buildcycle
  attach:  factory contained attach buildcycle
  result:  factory contained sync buildcycle
  stop:    factory contained rm buildcycle

buildcycle is running.
```

The endpoint lives as long as the run, not as long as the launching command — the launch returns
immediately while the run continues for minutes or hours. `factory contained rm` stops it.

It cannot be bound to loopback instead: the container reaches your machine through a gateway
address rather than through localhost, so a loopback bind would make the build tools unreachable
rather than make them safer.

`FACTORY_CONTAINED_DRY_RUN=1` shows the same banner, marked as not started, so you can see what
`--division` would do before doing it.

The agent gets the podman tool surface plus a brief telling it these are capabilities it already
has. Asked to name its tools, it answers with them rather than proposing to build a CLI wrapper:

```console
$ factory contained --division -- agent builder \
    --task "List the container tools available to you. Do not write code." --project ~/code/my-project

I have access to the following Podman/Docker container management tools:
**Container Operations:**
- `mcp__podman__container_list` — List running containers
- `mcp__podman__container_run` — Run a container from an image
- `mcp__podman__container_logs` — Display container logs
…
**Image Operations:**
- `mcp__podman__image_build` — Build an image from a Dockerfile/Containerfile
…
```

Without the flag, nothing is started, no `.mcp.json` is written, and the agent has no container
tools. The division is genuinely opt-in.

Requires `npx` on `PATH`.

---

## The cluster target

`--target k8s` runs the factory unattended on hardware your laptop is not: real CPU, real memory,
amd64, and a workspace that survives the pod.

### One-time namespace setup

`bundle` prints plain namespace-scoped YAML and never applies it. `setup` asks which namespace to
prepare, **checks what is already there**, then walks you through only the objects that are missing
or wrong — one at a time, each explained — and applies what you accept **with your own
credentials**:

```console
$ factory contained --target k8s setup

━━ 1/3  Cluster and namespace ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Clusters in your kubeconfig:

   1) 'default/api-my-cluster-example-com:443/you@example.com' (current)
      https://api.my-cluster.example.com:443
   2) 'factory/api-lab-cluster:443/you'
      https://api.lab-cluster.example.com:443

Which cluster? [1] 2

   This is where the factory's ServiceAccount, Role, RoleBinding and workspace
   PVC will live. If it does not exist yet, you will be offered the chance to
   create it.

   Cluster:   https://api.my-cluster.example.com:443
   User:      you@example.com
   Context:   default/api-my-cluster-example-com:443/you@example.com
   Namespace: 'default'  (the default below)

Namespace to prepare [default] factory-contained
   Namespace 'factory-contained' does not exist on this cluster.
Create namespace 'factory-contained' now?  [y]es  [n]o  (y/N): y
   $ oc new-project factory-contained
   Created factory-contained.
   `oc new-project` also made it your current project.

━━ 2/3  Review and apply ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Comparing 5 object(s) against namespace 'factory-contained' on
   'https://api.my-cluster.example.com:443':

   [ ok ] serviceaccount/factory       already present and matches what the factory needs
   [diff] role/factory-runtime         present, but not what the factory needs
   [new ] rolebinding/factory-runtime  not in this namespace — it would be created
   [new ] rolebinding/factory-scc      not in this namespace — it would be created
   [new ] pvc/factory-workspace        not in this namespace — it would be created

   1 already correct and will be skipped; 4 need(s) your decision.

── 1 of 4 · role/factory-runtime  (present, but not what the factory needs) ───
   What that identity may do, and the whole of it: create, watch and delete
   pods in this namespace, and read their logs. That is what a run needs to
   launch a validation pod and see why it failed. `pods/exec` is absent on
   purpose — the build sidecar is a boundary only because the agent cannot
   exec into it.

   What would change in factory-contained:
  rules:
-   verbs: ["get"]
+   verbs: ["create", "get", "list", "watch", "delete"]
Apply this? (1 of 4)  [y]es  [n]o  [a]ll remaining  [q]uit  (Enter or Esc = skip/stop): y
   role.rbac.authorization.k8s.io/factory-runtime configured

── 2 of 4 · rolebinding/factory-runtime  (would be created) ───────────────────
   Grants the Role above to the ServiceAccount above. Without it the Role
   exists and applies to nobody, and the run fails on its first cluster call.

apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
…
Apply this? (2 of 4)  [y]es  [n]o  [a]ll remaining  [q]uit  (Enter or Esc = skip/stop): a
   Applying this and the 2 after it.
   rolebinding.rbac.authorization.k8s.io/factory-runtime created
   rolebinding.rbac.authorization.k8s.io/factory-scc created
   persistentvolumeclaim/factory-workspace created

The credentials Secret is yours to create — the factory never handles the material:
  oc create secret generic factory-credentials -n factory-contained \
      --from-literal=ANTHROPIC_API_KEY=...
```

Six details that are deliberate.

The **cluster** is asked too, not just the namespace. A kubeconfig usually holds several, and `oc
config use-context` is the only way most people know to move between them — so picking the wrong
one meant Ctrl-C, a context switch and a restart. Whatever you choose is applied as `--context` on
**every** command this invocation issues, including the applies and the diffs; it does not rewrite
your kubeconfig, because deciding where *this* run goes must not silently change where your next
unrelated `oc get pods` goes. At the end, if you prepared a cluster that is not your default, you
are offered the switch and given the exact command either way. `--context NAME` skips the question.

The namespace is **asked**, not assumed — `--namespace` skips the question, and without it the
current context supplies the default rather than the answer, so a shared `default` never quietly
acquires a ServiceAccount, a Role and a 10Gi PVC. It is then **checked**, including when you passed
it explicitly: a typo would otherwise surface as five separate `NotFound` errors from the apply. If
it is not there you are offered the chance to create it, via `oc new-project` rather than `create
namespace`, because a regular user is usually denied the second and permitted the first. A
namespace you are simply *not allowed to read* — routine on OpenShift for a project you own — is
reported as unconfirmed rather than treated as missing.

The **cluster** is named alongside it. A namespace on its own identifies nothing — `default` exists
on every cluster you have ever logged into — so the API server URL is the field that actually
answers "am I about to apply RBAC to the right place?". Every namespace and server is printed
quoted and highlighted, because "in namespace default" gives a reader no way to tell the name from
the sentence. Only *names* are read from your kubeconfig: the context, the user's name, the server
URL, the namespace. Nothing from its `users` section, which is where credential material lives.

The **current state comes first**. Each object is checked against the namespace before anything is
asked, and the summary covers all of them including the ones already correct — "4 of 5 are already
there" is the most useful thing to know before deciding whether this is about to do something
drastic. Comparison is `oc diff`, done server-side, so a field the cluster defaults in does not
read as a change you are about to make.

Then it **walks only the difference**, one object at a time, each with what it is for and what
would change:

| State | What you see | Asked about? |
|---|---|---|
| already correct | one summary line | no — a prompt whose only sane answer is "yes" teaches people to stop reading prompts |
| would be created | its manifest | yes |
| present but differs | the **diff**, not the manifest | yes |
| could not be compared | a warning, then the manifest | yes — never silently skipped |

Each option is spelled out rather than abbreviated to `[y/n/a/q]`, which is readable only to
whoever wrote it. `y` applies one, `n` skips it, `a` applies everything remaining, and `q` **or
Escape** stops. A bare Enter skips, because the default has to be the answer that changes nothing;
anything unrecognised re-asks and never counts as yes. Where the terminal allows it these are
single keypresses — no Enter — which is also what makes Escape work at all, since a line-buffered
prompt can only ever see it as the `^[` characters it inserts.

**Escape backs out of any prompt in the flow**, not just this one — the cluster chooser and the
namespace prompt included, and the namespace prompt is a typed line, which is why it is read
character by character rather than with `input()`. **Ctrl-C** exits with a message and status 130
rather than a stack trace: changing your mind at question three is ordinary, not a crash.

Finally, each object is **applied the moment you accept it**, not batched until the end. You see
`role.rbac.../factory-runtime configured` before deciding the next one, and stopping halfway is
reported honestly:

```console
Apply this? (2 of 4)  [y]es  [n]o  [a]ll remaining  [q]uit  (Enter or Esc = skip/stop): q

   Stopped. 1 object(s) were applied before you stopped and remain applied; the rest were not.
```

That sentence is the whole reason for applying per object: batching would have said "nothing was
applied" to someone who had already said yes once. A skipped or unapplied object stays as it is,
and `verify` at the end reports it — nothing goes quiet. A single object failing to apply names
itself and does not stop the walk, since the rest may still be worth doing.

There is no second, blanket "are you sure?": every object was confirmed a moment earlier, and a
prompt on top of that is the friction that teaches people to hit `y` without reading. `--yes`
applies everything pending without walking, for automation.

Then `verify` checks every object, every verb the ServiceAccount needs, the Secret's **keys** (never
its values), and that inference is reachable from a pod *inside* the namespace. Results print **as
each one lands**, not at the end — several are a cluster round trip and the in-cluster inference
probe launches a pod and waits on it, so a step that stayed silent until the last check finished
was reported as a hang:

```console
$ factory contained --target k8s --namespace factory-contained verify
[ ok ] cluster_cli: oc, context factory-contained/api-…:443/you@example.com,
       server https://api.my-cluster.example.com:6443
[ ok ] namespace: factory-contained exists and is accessible
[ ok ] bundle:serviceaccount/factory: serviceaccount/factory present
…
[ ok ] permissions: serviceaccount/factory has every verb the run needs
[ ok ] no_pods_exec: serviceaccount/factory cannot exec into pods, which is what makes the build
       sidecar a boundary
[ ok ] credentials_secret: secret/factory-credentials carries the Anthropic API key
[ ok ] inference_from_cluster: a pod in this namespace reached the configured inference backend
[ ok ] secret_scanner: gitleaks present; workspaces are scanned before they leave this machine

All checks passed. Start a run with `factory contained --target k8s --namespace factory-contained -- ceo <path>`.
```

The inference check is the slow one: it creates a short-lived pod, with the same image and Secret
a real run uses, and asks it to make one request — because a host-side check proves nothing about
the *pod's* egress. It is announced before it starts, and **skipped entirely when the credentials
Secret is missing**, since the probe pod mounts that Secret and could only spend its 180-second
timeout rediscovering what the check above already said. That is the state a freshly prepared
namespace is in, because creating the Secret is deliberately left to you.

Before setup, the same command lists what is missing with the command that restores each — e.g.
`factory contained --namespace factory-contained bundle | oc apply -f -`.

### Running

```console
$ factory contained --target k8s --namespace factory-contained -- run ~/code/my-project --loop
k8srun
  attach:  factory contained --target k8s attach k8srun
  result:  factory contained --target k8s sync k8srun
  logs:    oc logs -f k8srun -n factory-contained -c factory
```

The workspace is packed into one tarball, streamed into an initContainer that is waiting for it, and
unpacked onto the PVC before the factory container starts. `oc cp` of a directory is one API round
trip per file, which is painfully slow on a repository.

### The secret scan

Nothing leaves your machine unscanned. Gitleaks runs over the workspace before the upload:

```console
$ factory contained --target k8s -- study ~/code/my-project
gitleaks: 1 finding(s)
  .env:1  [github-pat] Uncovered a GitHub Personal Access Token, potentially leading to
          unauthorized repository access and sensitive content exposure.

This workspace is about to be copied onto cluster storage. Anything above goes with it.
Refusing to upload without confirmation. Re-run with --yes to proceed non-interactively.
```

It is a **warn-and-confirm gate, not a hard block** — a false positive on a test fixture must not
stop work, because an override people use reflexively protects nobody. `--yes` proceeds, and says so
rather than passing silently. If gitleaks is not installed, the upload warns that it is unscanned
rather than quietly going ahead.

### Getting the work back, and tearing down

```console
$ factory contained --target k8s sync k8srun
k8srun: workspace fetched to ~/.factory-contained/k8srun/workspace.tar.gz.
  Review:  tar tzf ~/.factory-contained/k8srun/workspace.tar.gz
  Unpack:  mkdir -p <dir> && tar xzf ~/.factory-contained/k8srun/workspace.tar.gz -C <dir>
Nothing is merged automatically.

$ factory contained --target k8s rm k8srun
k8srun: pod deleted.
  The workspace is still on PVC factory-workspace in factory-contained. Fetch it with
  `factory contained --target k8s sync k8srun` before deleting the claim.
```

The PVC is deliberately left alone: it may hold the only copy of a long run's work.

### The cluster division

`--target k8s --division` is **OpenShift only**, refused at launch by API presence rather than by
whether `oc` happens to be installed. Builds go through OpenShift `Build` objects, submitted by a
**sidecar container** that is the only holder of `oc` and the ServiceAccount token — the agent's
container has neither, and cannot exec into the sidecar because the Role excludes `pods/exec`.
`verify` asserts that verb's absence; it is the one check that fails when something *succeeds*.

The agent gets one tool for building — `start_build(dockerfile, tag)` — plus namespace-scoped
cluster tools for launching validation pods and reading logs.

---

## Checks the runtime runs for you

Before the first agent call, the runtime asserts that the workspace it is about to use is the one
you meant — that it is present and non-empty, that git works in it, that `.factory/` arrived if your
project has one, that it is writable, and that a file's contents match the copy on your machine.

Each of these can fail silently otherwise: a read-only workspace looks like an agent whose edits
keep vanishing, and a stale copy produces a plausible result from the wrong code. A failed check
stops the run before any tokens are spent and leaves the container up so you can look inside it.

---

## The openshell target

`--target openshell` runs the factory inside an [OpenShell](https://github.com/NVIDIA/OpenShell)
sandbox — the target to reach for when the run's **input is not trusted**: an arbitrary issue, a
spec from a stranger, a codebase you were sent. See [Choosing a target](#choosing-a-target) for
what changes about the security story; this section is how it works day to day.

```bash
factory contained setup --target openshell     # guided: CLI, SDK extra, gateway, provider
factory contained verify --target openshell    # each failure with its fix
factory contained --target openshell -- ceo ~/code/untrusted-project --headless
```

The `--headless` on `ceo` is required: a sandbox has no terminal (tmux needs a PTY; a sandbox
cannot allocate one), so `ceo` is refused at parse time without `--headless` (or `--bg`, or
`--auto-approve` — design mode needs `--auto-approve` as it does everywhere else). The check
is per command, because the commands differ in what makes them interactive: `resume` and
`tmux` are refused outright (they *are* a terminal), while `run` and `agent` are headless by
default and only refused with `--tmux-persist`. The run is a detached process; `attach`
follows its log.

Setup automates nothing — every step either installs software (yours to choose) or touches
credential material (the `claude-code` provider, which the factory describes but never creates
for you). One trap it does name, because the failure names neither the gateway nor the VM: on
macOS/WSL with Docker Desktop, the sandbox supervisor runs in the VM's host network and cannot
reach a gateway bound to `127.0.0.1` — sandboxes then die at startup with
`ControlSupervisorStartFailed`. Point the docker driver at the host instead, in
`~/.config/openshell/gateway.toml`:

```toml
[openshell.drivers.docker]
grpc_endpoint = "https://host.docker.internal:17670"
```

The provider step is a **single import of the profile the factory ships** — the stock profile's
own header asks for a copy-edit ("drop `statsig.anthropic.com` and `sentry.io`, name the
resolved claude path"), and `factory/contained/openshell_claude_code.yaml` is that edit, made
once in the repo instead of by hand on every machine. Binary rules match parent processes, so
untrusted code descended from `claude` could otherwise reach the telemetry endpoints —
sentry.io is a multi-tenant ingest service, i.e. an exfiltration channel — and the kernel
matches `/proc/<pid>/exe`, which follows the npm symlink, so the stock `/usr/local/bin/claude`
binary path never matches and inference egress is silently denied. `verify` reads the
gateway's profile back and fails unless it matches the shipped one, so an import of the stock
profile does not pass:

```console
$ factory contained setup --target openshell

━━ OpenShell runtime ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   openshell CLI present; nothing to do.
   openshell SDK importable; nothing to do.
   Register a local gateway, if you have not:
     openshell gateway add --name local --local
     openshell gateway select local
   macOS/WSL with Docker Desktop: the sandbox supervisor cannot reach a
   gateway bound to 127.0.0.1. If sandboxes die at startup, point the
   docker driver at the host instead, in ~/.config/openshell/gateway.toml:
     [openshell.drivers.docker]
     grpc_endpoint = "https://host.docker.internal:17670"
   Inference is never automated — it touches credential material:
     openshell profile import -f <factory install>/factory/contained/openshell_claude_code.yaml --global
     openshell provider create --name claude-code --type claude-code --from-existing
```

The Python SDK is an optional dependency: `uv sync --extra contained-openshell`. Without it, the
target fails with that fix rather than a traceback, and `verify` reports it as a prerequisite.

A run is the same sequence as the cluster target — self-contained workspace copy, secret scan,
one tarball upload, provenance assertions before the first agent call, then the run starts and
the CLI returns. One difference from the other targets: the run is **not** in tmux, because a
sandbox cannot allocate the pseudo-terminal every tmux window needs (denied by the Landlock
filesystem allowlist). Instead the run is a detached process that writes its output to
`.factory/run.log` inside the project, its pid to `.factory/run.pid`, and its exit code to
`.factory/run.exit` — so `attach` follows the log (`Ctrl-C` detaches, and cannot disturb the
run); `sync` downloads the workspace back, log included; `rm` deletes the sandbox and keeps your
local workspace copy.

**The policy.** The default grants the workspace read-write, read-only access to the runtime
image's `/opt/factory` install, `uv` access to PyPI — and nothing else. (`pip` is deliberately
absent: OpenShell matches binaries by the kernel-resolved executable, and pip runs as the
Python interpreter, so a pip path never matches — install with `uv pip`.) Inference egress
comes from the attached `claude-code` provider, not the policy: the gateway synthesizes its own
rule for the provider's endpoints, and restating them in the policy is a create-time error.
`--policy <file>` replaces it entirely, in OpenShell's documented YAML format; the file you pass
is the whole policy. The default allowlist is code (a builder in
`factory/contained/openshell.py`, snapshot-tested), so changing what untrusted code can reach is
a reviewed diff, not a config edit nobody saw.

Only PyPI is allowlisted, and only for `uv`. npm, GitHub, and every other registry are denied —
a non-Python project (or a run that needs to push branches and open PRs) needs a `--policy`
file that says so, which is the point: egress for untrusted code should be a reviewed decision.

---

## Environment

| Variable | Purpose |
|---|---|
| `FACTORY_CONTAINED_IMAGE` | Override the runtime image |
| `FACTORY_CONTAINED_SIDECAR_IMAGE` | Override the k8s build sidecar's `oc` image |
| `FACTORY_CONTAINED_HOME` | Where workspace copies live (default `~/.factory-contained`) |
| `FACTORY_CONTAINED_DRY_RUN=1` | Print what would run; provision nothing |
| `FACTORY_LOG_LEVEL=debug` | Show every command the runtime issues (quiet by default) |

**Nothing crosses into the runtime that you did not ask for.** Variables starting with `FACTORY_`
go in, along with whatever `--forward` names and the variables your inference backend needs — and
nothing else. Your `~/.factory/` is mounted read-write, so config, credential profiles, the project
registry and evolved playbooks work exactly as they do outside. Anything else you want in there —
`~/.claude/projects/`, `GH_TOKEN`, `FACTORY_MANAGED_DIRS`, `FACTORY_VAULT_PATH` — you pass explicitly
with `--mount` or `--forward`.

---

## Troubleshooting

**"podman is installed but its engine is not reachable"** — on macOS the machine stops quietly.
`podman machine start`, or `factory contained setup`, which does it for you.

**"The workspace is read-only inside the runtime"** — the container runs as a user that does not own
your files. Check that the project is owned by you, and that `factory contained verify` is green.

**"could not read ... from inside a container"** — usually the podman machine does not share that
path. On macOS it shares your home directory; a project elsewhere is not mounted at all rather than
mounted empty. Move it under your home directory, or add the path with `podman machine set --volume`
and restart the machine. The launch warns about this before it happens.

**"is not a path the podman machine shares"** — same cause, caught at launch. The message lists the
paths that *are* shared.

**A wall of output instead of three lines** — that is `FACTORY_LOG_LEVEL=debug`. Unset it.

**"container 'x' already exists"** — a previous run left it. Attach to it, `rm` it, or pass `--name`.
A container that is no longer running is reaped automatically and the run retried once.

**"is already running a session — this is the same run, not a new one"** (k8s) — the pod is mid-run.
Attach, or `rm` and start again.

**"the division port 8430 is already held by the run 'x'"** — one port, one server. Finish or remove
that run first, or run this one without `--division`.

**Vertex 429s on every call** — pass an explicit `--model`. `MAX_THINKING_TOKENS=0` is pinned for you.

---

## Implementation

| Concern | Module |
|---|---|
| All podman CLI knowledge | `factory/podman.py` |
| All cluster CLI knowledge | `factory/contained/k8s.py` |
| Workspace copy | `factory/contained/workspace.py` |
| Provenance assertions | `factory/contained/provenance.py` |
| Container identity probe | `factory/contained/identity.py` |
| Credential shape | `factory/contained/credentials.py` |
| Local division | `factory/contained/division.py` |
| Pre-answering Claude Code's first-run prompts | `factory/contained/claude_state.py` |
| Cluster division | `factory/contained/k8s_division.py` |
| All OpenShell SDK/CLI knowledge, default policy | `factory/contained/openshell.py` |
| OpenShell prerequisite checks | `factory/contained/openshell_prereq.py` |
| Prereq bundle | `factory/contained/bundle.py` |
| Object-by-object review | `factory/contained/k8s_review.py` |
| Secret scan | `factory/contained/secrets.py` |
| Terminal colour and wizard steps | `factory/contained/style.py` |
| CLI front door | `factory/cli/contained.py` |
| Reading the command line | `factory/cli/contained_args.py` |
| One local container | `factory/cli/contained_local.py` |
| One cluster pod | `factory/cli/contained_k8s.py` |
| One OpenShell sandbox | `factory/cli/contained_openshell.py` |

The CLI modules **compose** commands and do not execute them, which is what makes
`FACTORY_CONTAINED_DRY_RUN=1` print the same argv the real path runs rather than a separate
rendering that drifts.

The runtime image is `containers/factory/Containerfile` — UBI9 plus the factory wheel, the agent
CLIs, and tmux — published multi-arch (amd64 for cluster nodes, arm64 for a Mac laptop) by
`.github/workflows/runtime-image.yml`. `factory contained setup` pulls it; it does not build.

It publishes on three events. A push to `main` moves `:latest`; a **published release** builds that
release's commit and publishes `:<release tag>`, moving `:latest` too unless the release is a
prerelease; `workflow_dispatch` publishes whatever tag you name. Every event also tags the short
SHA. The release trigger is the one that has to be automatic — `setup` pulls a published image and
does not build, so a release whose image was never built leaves a new user's first command failing
on a manifest that does not exist. Nightlies are prereleases and therefore never move `:latest`.
