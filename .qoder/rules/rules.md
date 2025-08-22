---
trigger: always_on
alwaysApply: true



---
Universal Project Rules
Version: 2025-08-22.0Purpose: Define a standardized set of rules applicable to any project, ensuring consistency, quality, and efficiency across all development activities.Scope: All project-related activities, including coding, testing, deployment, and collaboration.Philosophy: “Produce correct, minimal, and secure outputs on the first attempt, with clear specifications driving all decisions.”
0. IDENTITY & CONSTRAINTS

Project Scope: Clearly define the project’s purpose, deliverables, and boundaries at the outset.
Team Roles: Assign and document roles (e.g., developers, testers, reviewers) to avoid ambiguity.
Tooling: Use standardized, industry-accepted tools for development, testing, and deployment, ensuring compatibility across environments.
Version Control: Use a single source of truth (e.g., Git) with a defined branching strategy (e.g., Gitflow, trunk-based development).
Environment: Define development, staging, and production environments with consistent configurations.

1. SPECIFICATION & PLANNING CONTRACT

Tone: Use clear, precise, and deterministic language in all specifications and documentation (e.g., RFC-2119 style: MUST, SHOULD, MAY).
Validation: All specifications must be validated at creation using predefined schemas or templates.
Forbidden Practices: Avoid vague terms (e.g., “maybe,” “possibly,” “later”) in requirements or planning documents.
Required Documentation: Every feature or change must include:
A clear description of functionality.
Acceptance criteria.
Reference to related specs or tests.



2. DISCOVERY & RESEARCH RULES

Research Phase: Every new feature or change MUST begin with a discovery phase to gather requirements, constraints, and dependencies.
Data Sources: Use reliable, up-to-date sources (e.g., official documentation, verified libraries) for research.
Caching: Cache external research results (e.g., API docs, library references) for a defined period to ensure reproducibility.
Versioning: Store research outputs (e.g., JSON, screenshots) in a versioned directory for traceability.

3. ZERO-DEFECT SAFEGUARDS

Validation Pipeline: All outputs (code, configs, docs) must pass:
Static analysis (e.g., linting, formatting checks).
Unit tests with 100% coverage for critical paths.
Integration or end-to-end tests for key user journeys.
Visual or functional snapshot tests to detect regressions.


Failure Handling: If any validation fails, revise the specification or implementation with stricter constraints.
Comments: Inline comments must reference specific requirements, tests, or issues—no generic placeholders (e.g., TODO, FIXME).
Commit Hygiene: No untested or incomplete changes may be committed to the main branch.

4. SIMPLEST-SOLUTION POLICY

Minimalism: Favor the simplest implementation that meets requirements.
Modularity: Keep modules small (e.g., < 100 lines where feasible) and focused on a single responsibility.
Avoid Premature Abstraction: Do not introduce abstractions until duplication occurs at least three times.
Composition: Prefer composition over inheritance or complex hierarchies.
Review: Reject changes that introduce unnecessary complexity or abstractions.

5. TESTING – COMPREHENSIVE & PROPERTY-BASED

Coverage: Aim for 100% coverage of critical code paths (lines, branches, functions).
End-to-End Tests: Cover critical user journeys with automated tests.
Property-Based Testing: Use property-based testing for all stateless, pure functions to ensure robustness.
Mutation Testing: Achieve a high mutation score (e.g., ≥ 80%) to validate test effectiveness.
Snapshots: Use snapshot testing for UI or output consistency, with a zero-tolerance threshold for unintended changes.
Test Data: Use data factories or mocks to generate consistent, realistic test data.
Performance: Define and enforce performance budgets (e.g., p95 response time < 300 ms in production).

6. SECURITY – SHIFT-LEFT PIPELINE

Secrets Management: Scan for secrets (e.g., API keys, credentials) before commits.
Dependency Scanning: Check dependencies for known vulnerabilities in CI pipelines.
Infrastructure Scanning: Validate infrastructure-as-code (IaC) configurations for security issues.
Dynamic Testing: Run dynamic application security tests (DAST) regularly against staging environments.
SBOM: Generate and commit a Software Bill of Materials (SBOM) for each release.
Security Headers: Enforce strict security policies (e.g., Content-Security-Policy) without unsafe practices.

7. DEPLOYMENT-AS-CODE

Containerization: Use minimal, secure container images with non-root users.
Infrastructure as Code: Define all infrastructure in code, stored in version control with validated schemas.
CI/CD: Automate builds, tests, and deployments with matrix testing across supported environments.
Canary Deploys: Use canary or gradual rollouts with automated rollback on failure (e.g., based on error rates).
Monitoring: Define and monitor service-level objectives (SLOs) with automated alerts.

8. CODE BEST PRACTICES

Type Safety: Use strict typing or schema validation where applicable to prevent runtime errors.
Immutability: Prefer immutable data structures for critical data contracts.
Explicitness: Avoid implicit behaviors (e.g., type coercion, default fallbacks) in favor of explicit checks.
Modularity: Limit exports per module to a small, focused set (e.g., ≤ 3 exports).
No Magic: Avoid “magic” values or behaviors; document all assumptions explicitly.

9. PROJECT STRUCTURE

Directory Layout:project/
├─ src/               # Core source code
│  ├─ features/       # Feature-specific modules
│  ├─ shared/         # Reusable utilities, components, or schemas
│  └─ tests/          # Unit, integration, and E2E tests
├─ infra/             # Infrastructure-as-code definitions
└─ docs/              # Documentation and fixtures


Rules:
Features may not depend on other features to avoid tight coupling.
Shared code may not import from feature-specific code.
Each file should have a single, clear responsibility.
Avoid deep nesting of directories or exports.



10. COMMIT & REVIEW CONTRACT

Commit Style: Use a standardized commit message format (e.g., Conventional Commits).
PR Requirements: Every pull request must include:
Links to relevant specifications, tests, or research fixtures.
Evidence of testing (e.g., screenshots, test reports).
Security and performance checklists.


Review Process: Reviewers must validate changes locally (e.g., lint, test, build).
Merge Policy: Block merges until all tests pass and quality metrics are maintained or improved.

11. LOCAL DEVELOPMENT EXPERIENCE

Dev Setup: Provide a single command to start the development environment.
Pre-Commit Hooks: Enforce linting, testing, and security scans before commits.
Local Infrastructure: Support a one-command setup for local infrastructure (e.g., via containers).
Environment Config: Provide and sync example environment files (e.g., .env.example).
Editor Config: Commit editor settings to ensure consistent formatting across the team.

12. CUSTOM TEMPLATES

API Template: Create secure, validated API endpoints with:
Input validation.
Rate limiting.
Security headers.
Comprehensive tests.


Component Template: Create modular, reusable components with:
Typed inputs.
Test coverage (unit, visual).
Minimal styling dependencies.
Small footprint (e.g., ≤ 100 lines).



13. EMERGENCY OVERRIDES

Ambiguity Handling: If a specification is unclear, return an error object with details (e.g., { error: "ambiguous spec: ..." }).
Issue Tracking: Log ambiguities as issues in the project’s issue tracker, not as comments in deliverables.
Override Logging: Log all manual overrides to a dedicated file and review them periodically.
