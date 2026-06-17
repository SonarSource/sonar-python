# Quick-Fix Feasibility Criteria

This file defines how the quick-fix feasibility score is assigned in `current-analyzer-rules.md`.

## What The Score Means

The score is **not** "could a developer fix this manually?"

The score is:

**Can the analyzer offer one or more safe automatic quick fixes for the reported issue shape?**

That can be:

- one exact rewrite
- a small bounded menu of rewrites
- a deterministic rename
- a small local structural refactor

It does **not** need to be a single canonical edit if the safe alternatives are few and mechanically derivable.

## Conditions That Favor A Quick Fix

The score should be high when most of the following are true:

1. The issue is local to one token, expression, statement, decorator list, argument list, or a very small AST fragment.
2. The analyzer can derive the rewrite from syntax, local semantic information, or a small amount of already available symbol usage data.
3. The replacement is deterministic, or the analyzer can offer a bounded set of safe alternatives (roughly fewer than 5).
4. The edit stays within one file or one local scope and does not require project-wide coordination.
5. The rule already implies the target text, target API, target flag, target threshold, or target rename pattern.

Typical high-feasibility patterns:

- Exact token/operator replacement
- Exact API substitution
- Exact flag insertion or boolean flip
- Exact threshold bump when the minimum secure value is known
- Deterministic rename from an existing pattern
- Small local refactor such as wrapping, reordering, extracting a simple expression form, or replacing one syntactic construct with another equivalent one

## Conditions That Work Against A Quick Fix

The score should be low when one or more of the following are true:

1. The analyzer must invent a business value, secret, hostname list, timeout value, schema, field set, status code, or other open-ended domain choice.
2. The edit needs project-wide changes such as file renames, import updates across files, caller updates, or public API migration.
3. The issue requires open-ended design, algorithm, architecture, or behavioral decisions.
4. The rule suggests a review or policy judgment more than a code transformation.
5. The rewrite would likely be incomplete with the current text-edit model, even if a human could perform it safely.

Typical low-feasibility patterns:

- Complexity / size / broad refactor rules
- Rules that require inventing a missing value
- Security hotspot rules where the secure configuration is not uniquely derivable
- Contract changes that propagate outside the local scope
- Fixes that depend on application-specific behavior, not just syntax or local semantics

## Practical Scale

- `10/10`: Exact deterministic rewrite. One obvious safe edit.
- `8-9/10`: Deterministic local rewrite or a small bounded menu of safe edits.
- `5-7/10`: Feasible local rewrite, but guarded, API-specific, broader in scope, or dependent on a few supported shapes.
- `2-4/10`: Partially automatable, but often incomplete, heuristic, or risky because scope or semantics escape the local shape.
- `0-1/10`: No responsible automatic fix. The issue needs open-ended human judgment or project-wide work.

## Sanity Checks

- `InequalityUsage`: `10/10`
  Reason: exact token replacement from `<>` to `!=`.

- `ClassComplexity`: `0-1/10`
  Reason: this is an open-ended refactor problem, not a bounded rewrite.

- `S117` local variable / parameter naming: high
  Reason: the rename transform is deterministic and the scope is local.

- `S1578` module naming: low
  Reason: a file rename spills into imports and project structure.

- `S2092` / `S3330` cookie flags: high
  Reason: the secure rewrite is a local flag addition or boolean change.

- Missing timeout value rules: low
  Reason: the analyzer cannot safely invent the timeout value.

- `S7505` map + lambda to comprehension: high
  Reason: this is a bounded local structural rewrite.

- `S7484` event instead of polling loop: low
  Reason: it is a broader behavioral refactor, not a small local rewrite.
