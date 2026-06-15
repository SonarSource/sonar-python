# Quick-Fix Feasibility Criteria

This file defines how the quick-fix feasibility score is assigned in `current-analyzer-rules.md`.

## What The Score Means

The score is **not** "could a developer fix this manually?"

The score is:

**Can the analyzer apply a safe automatic rewrite, with one canonical edit, without asking the user to make a design, security, naming, or business decision?**

## Conditions That Favor A Quick Fix

The score should be high when most of the following are true:

1. The issue is **local** to one token, expression, statement, or very small AST fragment.
2. The replacement is **deterministic**: one clear target rewrite exists.
3. The rewrite can be derived from **syntax or local semantic information** already available to the analyzer.
4. The analyzer does **not** need to invent a business value, timeout, password, name, API contract, or algorithm.
5. The rewrite preserves intent in a **predictable** way, or the rule itself defines the exact replacement.

Typical high-feasibility patterns:

- Exact token/operator replacement
- Exact API substitution
- Deterministic removal of redundant syntax
- Deterministic insertion of a known missing token/argument
- Deterministic rename when the target name is already known
- Small AST-local rewrite with one canonical result

## Conditions That Work Against A Quick Fix

The score should be low when one or more of the following are true:

1. Fixing the issue requires a **refactor**, not a rewrite.
2. The analyzer must choose between **multiple valid solutions** with no single canonical one.
3. The fix needs a **human decision** about naming, architecture, algorithm, control flow, or public behavior.
4. The fix requires choosing a **security-sensitive or environment-specific value** such as a password, timeout, salt, hostname policy, encryption mode, or access policy.
5. The issue is mainly a **review/hotspot/policy** problem rather than a deterministic code transformation.
6. The fix would require broad **cross-file or project-level reasoning** rather than a local rewrite.

Typical low-feasibility patterns:

- Complexity / size / refactor rules
- Security hotspot rules requiring manual review
- "Use a more specific X" rules where the analyzer cannot choose the right replacement
- Missing-value rules where the value is domain-specific
- Naming rules where several compliant names are possible

## Practical Scale

- `10/10`: Exact deterministic rewrite. Example: replace one operator by another, remove one useless token, add a missing newline.
- `8-9/10`: Deterministic local rewrite, but it depends on a little local context. Example: rename to a known derived name, replace a call with a known API alternative, add a known missing parameter.
- `5-7/10`: Algorithmic fix is plausible, but only for constrained cases, or it relies on a placeholder / narrow inference / guarded transformation.
- `2-4/10`: A rewrite can be imagined, but it would be heuristic, noisy, or often wrong because several valid edits exist.
- `0-1/10`: No responsible automatic fix. The issue needs refactoring, human judgment, or a security/business decision.

## Sanity Checks

- `InequalityUsage`: `10/10`
  Reason: exact token replacement from `<>` to `!=`.

- `ClassComplexity`: `0-1/10`
  Reason: this is a refactor problem, not a deterministic edit.

- Hard-coded credentials / secure configuration / timeout value selection:
  Usually `0-2/10`
  Reason: the analyzer cannot safely choose the replacement value.

- Placeholder docstring / TODO / simple local boilerplate insertion:
  Usually `6-8/10`
  Reason: algorithmically easy, but often weaker than a semantic rewrite.
