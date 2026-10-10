# SWE-Explore Benchmark Results

Benchmark: [SWE-Explore](https://arxiv.org/abs/2606.07297) (arXiv 2606.07297)
Date: 2026-08-18
Instances: 50 (from 848 total, limited by repo availability and issue text coverage)
Top-k: 5 regions per instance
Model: Claude Sonnet
Timeout: 600s per instance

## Configurations

| Config | Description |
|--------|-------------|
| **Claude Code** | Benchmark's baseline prompt. Minimal instruction: find relevant files, output `RELEVANT_FILES:` block. Raw Claude CLI with Read/Glob/Grep tools. |
| **Factory Agent** | Structured 6-step exploration prompt: (1) understand the issue, (2) map the codebase, (3) search for entry points, (4) trace the code path, (5) identify root cause, (6) check tests. Same Claude CLI backend. |
| **Factory Graph** | Two-phase pipeline: a shell study node scans the repo structure first (file listing, test files, config files), then the exploration agent runs with the study output pre-loaded as context. |

## Results

### Primary Metrics

| Metric | Claude Code | Factory Agent | Factory Graph | Winner |
|--------|:-----------:|:-------------:|:-------------:|--------|
| hit_file_rate | 0.6467 | 0.7470 | **0.7673** | Graph +19% |
| hit_region_rate | 0.5467 | 0.5850 | **0.6017** | Graph +10% |
| weighted_core_coverage | 0.1041 | 0.1156 | **0.1176** | Graph +13% |
| precision | **0.7421** | 0.6337 | 0.6062 | CC +22% |
| recall | **0.0950** | 0.0816 | 0.0596 | CC |
| f1_score | **0.1378** | 0.1180 | 0.0972 | CC |
| noise_file_rate | **0.2030** | 0.2483 | 0.2633 | CC |
| noise_region_rate | **0.1633** | 0.1850 | 0.1900 | CC |
| context_efficiency | **0.8589** | 0.8008 | 0.8025 | CC |
| first_useful_hit | 0.9960 | 0.9960 | 0.9960 | Tie |

### Ranking Metrics

| Metric | Claude Code | Factory Agent | Factory Graph |
|--------|:-----------:|:-------------:|:-------------:|
| ndcg@100 | 0.9780 | 0.9919 | 0.9924 |
| ndcg@300 | 0.9844 | 0.9934 | 0.9939 |
| ndcg@500 | 0.9844 | 0.9934 | 0.9939 |
| recall@100 | 0.0648 | 0.0624 | 0.0432 |
| recall@300 | 0.0804 | 0.0788 | 0.0575 |
| recall@500 | 0.0838 | 0.0816 | 0.0596 |

### Paired Wins (50 shared instances)

Each cell shows on how many instances that explorer achieved the best score (ties counted for all winners).

| Metric | Claude Code | Factory Agent | Factory Graph |
|--------|:-----------:|:-------------:|:-------------:|
| hit_file_rate | 28 | 40 | **43** |
| f1_score | **26** | 18 | 14 |
| weighted_core_coverage | **22** | 19 | 20 |
| context_efficiency | **32** | 27 | 25 |

## Analysis

### File-level localization: factory wins

Both factory configurations significantly outperform raw Claude Code at finding the right files. Factory Graph finds a relevant core file on 77% of instances vs 65% for Claude Code — a 19% relative improvement. On paired comparisons, Factory Graph wins on 43/50 instances for hit_file_rate.

The structured exploration prompt guides the agent through a systematic process (map codebase, search entry points, trace code path, identify root cause) rather than leaving exploration strategy entirely to the model. This produces more thorough file coverage.

### Line-level precision: Claude Code wins

Raw Claude Code returns tighter, more precise line ranges. Its precision (0.74) is 22% higher than Factory Agent (0.63) and context_efficiency (0.86) is 7% higher. This means Claude Code's regions contain proportionally more relevant code and less noise.

The factory's richer prompt likely causes the agent to include broader context around the identified code (larger regions, more surrounding functions), which improves file-level coverage at the cost of line-level precision.

### Study phase adds incremental value

Factory Graph consistently beats Factory Agent by 2-3% across file-level metrics (hit_file_rate: 0.767 vs 0.747, hit_region_rate: 0.602 vs 0.585). The pre-computed file listing from the study phase saves the agent from spending tool calls on initial directory traversal and provides a structural overview that helps with navigation.

### Core coverage is close

Weighted core coverage — the metric that accounts for which files actually needed modification — is within 1.3% across all three (0.104 to 0.118). The factory explorers find more files but each region is less precisely targeted, resulting in similar overall coverage of the code that matters most.

### Ranking quality is high for all

All three achieve ndcg@100 > 0.97 and first_useful_hit > 0.99, meaning the ranking of regions is good regardless of configuration — useful code appears early in the results.

## Limitations

- **Sample size**: 50 of 848 instances (limited by repo availability — 451/848 repos downloaded successfully, 633/848 had issue text from HuggingFace datasets)
- **No line counts**: Metrics requiring file line counts (for resolving `end=-1` regions) were computed without them (`--no-line-counts`), which may undercount some ground truth regions
- **Single model**: All three configurations use Claude Sonnet — results may differ with other models
- **Sequential runs**: Each configuration ran sequentially, not simultaneously, so network/API variability could affect timing comparisons
