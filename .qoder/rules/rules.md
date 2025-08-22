---
trigger: always_on
alwaysApply: true

---
================================================================================PROJECT QODDI AI MODEL RULES – ZERO-HALLUCINATION EDITIONVersion: 2025-08-22.0Purpose: Single source of truth for all AI activity in this repo, governing the development of a hyper-small, ultra-fast, CPU-running AI model (non-transformer-based, agentic style) with expertise in software development, project management, context engineering, and prompt engineering. The model rivals 2025-era models (Claude, GPT, DeepSeek, Kimi, Qwen) in versatility, targeting <100M parameters, 10-50W inference, and training on a single RTX 2070 Ti, using small datasets (<1GB).Scope:   Qoddi chat, agentic workflows, inline fixes, debugger, MCP servers. Qoddi’s platform (https://docs.qoder.com/) automates containerized builds, CI/CD, and agentic task decomposition for rapid, error-free development.Philosophy: “If the AI can’t produce correct, minimal, secure code on the first try, the prompt or spec is underspecified.” Decompose tasks into bite-size pieces (<50 LoC, <1 hour) to minimize errors/hallucinations and enable parallel agentic progression via Qoddi.
0  IDENTITY & CONSTRAINTS
lang                 = "Rust (primary for safety/performance), Zig (low-level kernels for SNN ops), Go (concurrency for agent spawning/MCP interfaces)"runtime              = "Native CPU executable (no GPU dependency post-training)"platform             = "Standalone binary, cross-compilable for Linux/Windows/macOS/ARM via Cargo/Zig, deployable via Qoddi Docker containers"package_mgr          = "Cargo (Rust), Zig build system, Go modules"test_runner          = "Cargo test (Rust), Zig test, Go test + criterion-rs for benchmarks"e2e_runner           = "Custom Rust benchmarks for inference speed/energy + agentic task simulation"security_pipeline    = "Cargo audit + clippy (Rust), Zig safety analysis, Go vet + gosec, Trivy for Qoddi Docker images"deploy_target        = "Qoddi containerized deployment (Docker), cross-compiled binaries for edge devices"infrastructure       = "Docker Compose for local training, Qoddi for prod deployment, no cloud infra unless via MCP"mcp_servers          = [  "mcp-server-api",     # External API calls (e.g., GitHub, library docs)  "mcp-server-tools",   # Coding tools (linters, compilers, Git)  "mcp-server-data",    # Dataset access/augmentation  "mcp-server-feedback" # Self-improvement loops]
1  PROMPT ENGINEERING CONTRACT
ai_tone              = "terse, RFC-2119, deterministic, coding-focused"max_tokens           = 400  # Concise for CPU efficiencytemperature          = 0.0  # Zero for reproducibility, zero hallucinationstop_p                = 0.0deterministic_seed   = 42prompt_validator     = "schemars (Rust), zig-test for Zig, gojsonschema (Go)"  # Compile-time validate promptsprompt_linter        = "clippy + custom lints (Rust), zig fmt, go vet"  # Enforce code/prompt consistencyforbidden_words      = ["maybe", "perhaps", "should", "TODO", "hallucinate"]required_patterns    = ["// SAFETY: reason:", "/* tested by: file.rs */", "// MCP: delegated to server X", "/// Doc comment for all public APIs"]
2  MCP USAGE RULES

Every coding task/feature MUST start with an MCP-driven discovery:
mcp-server-data for datasets (e.g., HumanEval, synthetic Rust/Zig snippets).
mcp-server-tools for tool integration (e.g., cargo check, zig build, go build).
mcp-server-api for external resources (e.g., GitHub repos, API docs).
mcp-server-feedback for self-evolution (e.g., genetic algo updates).


Cache MCP calls 24h in .cache/mcp (git-ignored); fallback to internal logic if offline.
Store MCP results as JSON fixtures in __fixtures__/mcp/YYYY-MM-DD/ for reproducibility.
Delegate to MCP only for external data/tools (e.g., library versions, linters); prefer internal agent spawning for code generation.

3  ZERO-HALLUCINATION SAFEGUARDS

All AI output (code, model weights, configs) must pass:
Type-check (cargo check --all-features, zig build-obj, go build)
Lint (cargo clippy -- -D warnings, zig fmt, go vet)
Test (cargo test --all, zig test, go test)
Benchmark diff (cargo criterion --save-baseline for speed/energy, <100ms inference)


If any step fails, regenerate with stricter prompt or decompose task further.
Inline comments must reference spec, test file, or MCP fixture (e.g., // See __fixtures__/mcp/2025-08-22/api.json).
No TODO/FIXME; spawn agent to resolve or file GitHub issue with ai-ambiguous label.

4  SIMPLEST-SOLUTION POLICY

Prefer Rust stdlib over crates; use Zig stdlib for kernels, Go stdlib for concurrency.
Write modules <50 LoC; split if larger.
Avoid abstractions until duplication ≥ 3×.
Favor composition (Rust traits, Zig structs) over inheritance.
Reject changes adding bloat; prioritize CPU efficiency (<50W inference).
Use Zig for SNN kernels only if Rust perf insufficient (e.g., matrix ops); use Go for MCP concurrency only if Rust rayon/tokio insufficient.

5  TESTING – 100% COVERAGE & PROPERTY-BASED
unit_coverage        = "100% lines, branches, functions via cargo-tarpaulin (Rust), zig test, go test"e2e_coverage         = "Critical paths: model training, inference, agent spawning, MCP delegation"property_tests       = "proptest (Rust), zigtest (Zig), testing/quick (Go) for pure fns (e.g., neural ops)"mutation_score       = "≥80% via cargo-mutants (Rust), Zig mutations, Go mutant"snapshot_threshold   = "0% diff in model outputs via golden Parquet files"test_data            = "Small datasets (HumanEval, TinyStories, synthetic code) as JSON/Parquet fixtures"fixtures             = "Committed in __fixtures__/datasets/ (e.g., human_eval.json)"performance_budget   = "Inference: p95 <100ms on i7 CPU, energy <50W; Training: <24h on RTX 2070 Ti, VRAM <8GB"
6  SECURITY – SHIFT-LEFT PIPELINE

Secrets scanning: cargo audit pre-commit, go vet, zig safety checks.
Dependency scanning: cargo vet --severity high, trivy fs . for Zig/Go.
Binary scanning: trivy image for Qoddi Docker images.
Model security: Sanitize datasets (e.g., remove biases from HumanEval); use sparse activations to prevent overfit.
DAST: Nightly fuzzing with cargo-fuzz (Rust), zig-fuzz, go-fuzz against model inputs.
SBOM: Generate via cargo generate-sbom, commit to sbom/ per release.
Access controls: Encrypt MCP calls with rustls (Rust), crypto/tls (Go); no unsafe Rust unless justified with // SAFETY.

7  DEPLOYMENT-AS-CODE

Dockerfile: Multi-stage Rust builder + distroless base, non-root UID 1000 for training/Qoddi deployment.
Binaries: Cross-compile via cargo build --target, zig build, go build for Linux/Windows/macOS/ARM.
Local state: In-memory or MCP-cached; no persistent storage except fixtures.
CI matrix: Test on Rust 1.80+, Zig 0.13+, Go 1.23+ across OSes via Qoddi CI.
Deployment: Qoddi Docker push or binary distribution; agentic self-deploy via MCP if scaled.
Rollback: Versioned binaries with git tags; Qoddi rollback on failed health checks.

8  LANGUAGE-SPECIFIC BEST PRACTICES
Rust

Strictest Cargo.toml: Enable all lints (clippy::all), no unsafe without // SAFETY.
Use const generics, associated types for model params.
Branded types for IDs (e.g., type ParamCount = usize & { __brand: 'ParamCount' } via newtype).
No panics; use Result/Option. Use thiserror for errors.
Prefer immutable refs (&T over &mut T).
Use anyhow::Error for catch-all errors.
Import exact paths; no wildcards.

Zig (Kernels)

Use /// doc comments for all public fns/structs; generate docs with zig build -femit-docs.
No heap allocations unless explicit (e.g., @alloc); prefer stack for SNN ops.
Use error unions (!void) for fallible fns; handle all errors explicitly.
Import stdlib via @import("std"); no external deps for kernels.
Keep modules <50 LoC; use comptime for type checks.

Go (MCP/Concurrency)

Use go vet, go fmt for strict formatting.
Prefer goroutines for agent spawning/MCP calls; avoid channels unless necessary.
Use errors.Is/errors.As for error handling; no panics.
Keep packages minimal; import only net/http, encoding/json for MCP.
Export minimal APIs; use //go:generate for schemas if needed.

9  DIRECTORY LAYOUT & FACTORING
src/ ├─ main.rs              # Entry point for model binary (Rust) ├─ model/              # Core AI model logic (Rust) │   ├─ core.rs         # SNN-SSM-liquid NN hybrid │   ├─ agentic.rs      # Agent spawning/merging │   ├─ mcp.rs          # MCP client logic │   └─ *.rs            # Sub-modules <50 LoC ├─ training/           # Training logic (Rust) │   ├─ datasets.rs     # Small dataset handlers │   ├─ trainer.rs      # Training loop (GPU via cudarc if needed) │   └─ *.test.rs ├─ kernels/            # Low-level perf kernels (Zig) │   ├─ snn.zig         # Spiking neural network ops │   ├─ matrix.zig      # Matrix ops for SNN/SSM │   └─ *.zig           # <50 LoC each ├─ mcp/                # MCP interfaces/concurrency (Go, if used) │   ├─ api.go          # MCP server API calls │   ├─ tools.go        # Tool integration (e.g., Git, linters) │   └─ *.go            # <50 LoC each ├─ utils/              # Shared utilities (Rust) │   ├─ perf.rs         # Benchmarks/energy monitoring │   └─ schemas.rs      # Schemars for prompts/data └─ tests/              # Tests (Rust, Zig, Go)    ├─ integration/     # E2E tests (model, agents, MCP)    └─ fixtures/        # Datasets (JSON/Parquet), MCP responses
Rules:

model/ imports utils/, kernels/, mcp/ only; no sibling imports.
training/ is isolated (no runtime imports) for GPU/CPU separation.
kernels/ (Zig) is standalone; FFI to Rust via export fns.
mcp/ (Go) is standalone; FFI to Rust via cgo if needed.
utils/ imports no other dirs.
Files export ≤3 symbols; split if exceeded.
Flat structure; no deep nesting.

10  COMMIT & REVIEW CONTRACT

Conventional commits (feat:, fix:, etc.), squash-merge only.
PR template enforces:
Link to MCP fixture (e.g., __fixtures__/mcp/2025-08-22/api.json).
Benchmark plot (energy via powertop, speed via criterion).
Security checklist (cargo audit, trivy, fuzzing).
Hallucination check: Output diff against golden files.


Reviewers run cargo ci:local (check, clippy, test, bench) or equivalent for Zig/Go.
Merge blocked if mutation score drops (cargo-mutants, Zig/Go mutants).

11  LOCAL DEV DX

Dev build: cargo run (inference), cargo train (GPU training).
Pre-commit: cargo fmt, cargo clippy, cargo test, zig fmt, go fmt, audit.
Local setup: docker compose up for training env (RTX 2070 Ti sim) via Qoddi.
.env.example synced via cargo run --bin gen-env.
Qoddi settings: Commit .qoddi.yaml for Rust/Zig/Go linting, agentic workflows.

12  CUSTOM QODDI AGENTIC SNIPPETS
snippet_model_module = "  Create a Rust module for model component:

Use traits for modularity (e.g., impl NeuralLayer).
Sparse activations with 4-bit quantization.
Unit test with proptest.
MCP delegation for data if needed.
<50 LoC, no unsafe."snippet_agent_spawn = "  Create agentic spawning fn:
Use rayon (Rust) or goroutines (Go) for parallel sub-models.
Ensemble voting for output merge.
Test for hallucination reduction (<1% error).
Integrate MCP if external data needed."snippet_kernel = "  Create a Zig kernel for SNN:
Use @comptime for type safety.
No heap allocations; stack-based.
Export via FFI for Rust.
Unit test with zig test.
<50 LoC."snippet_mcp = "  Create a Go MCP interface:
Use net/http for API calls.
Goroutines for concurrency.
JSON schema validation via gojsonschema.
Test with go test.
<50 LoC."

13  EMERGENCY OVERRIDES

If spec ambiguous, return {error: "ambiguous spec: ..."} and ask clarifying questions via Qoddi chat.
No TODO comments; spawn agent to resolve or file GitHub issue with ai-ambiguous label.
Log overrides to .qoddi-overrides.log; review weekly.

14  TASK BREAKDOWN POLICY

All tasks MUST be decomposed into bite-size pieces (<50 LoC, <1 hour) to reduce errors/hallucinations and speed progression:
Define spec: Clear 1-2 sentence goal (e.g., “Add SNN activation fn in Zig”).
MCP discovery: Fetch needed info (e.g., SNN math from mcp-server-data).
Agentic split: Spawn sub-tasks (e.g., Agent1: Prototype fn; Agent2: Test; Agent3: Benchmark).
Implement: Write/test/lint each piece independently.
Merge & verify: Ensemble check across agents; benchmark speed/energy.
Iterate: Regenerate sub-task if fails (stricter prompt).


Benefits: Parallelism via Qoddi agents, incremental commits, <1% hallucination.
Track: Each piece <1 hour; tasks <1 day. Log progress in .qoddi-progress.log.

15  ADDITIONAL RULES

Dataset Constraints: Use small datasets (<1GB): HumanEval (1MB), TinyStories (10MB), GSM8K subset (500KB), BabyLM (100MB), MiniPile (200MB), synthetic code (50MB). Generate synthetic data via agents if needed.
Architecture: Enforce non-transformer hybrid (SNN for event-driven, SSM for sequences, liquid NN for adaptability); evolve sub-modules via genetic algorithms.
Energy: Benchmark every change with powertop; reject if >50W inference.
Versatility: Test across domains (code, math, language) with equal weighting; use MCP for domain-specific tools (e.g., Git for code, Wolfram for math).


================================================================================END OF FILE