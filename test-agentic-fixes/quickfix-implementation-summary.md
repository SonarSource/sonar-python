# Quick-Fix Implementation Summary

This note summarizes the quick-fix work completed for the selected Python rules.

## Implemented Quick Fixes

The following rules now have quick-fix support implemented or verified in the analyzer:

- `python:S101` - Class names should comply with a naming convention
  - Feasibility: `4/10`
  - Feasibility justification: The rename is deterministic, but class names often escape the file through imports and public APIs.
  - Implemented a targeted rename quick fix for the default naming pattern.
  - The fix derives a PascalCase name and renames all local usages through the existing rename helper.

- `python:S117` - Local variable and function parameter names should comply with a naming convention
  - Feasibility: `8/10`
  - Feasibility justification: The rename transform is deterministic and the check already works from local-scope symbol usages.
  - Implemented a targeted rename quick fix for the default naming pattern.
  - The fix derives a snake_case name and renames all local usages.

- `python:S1542` - Function names should comply with a naming convention
  - Feasibility: `5/10`
  - Feasibility justification: The rename transform is mechanical, but function names can leak across callers and public APIs.
  - Implemented a targeted rename quick fix for the default naming pattern.
  - The fix derives a snake_case name and renames all local usages.

- `python:S1656` - Variables should not be self-assigned
  - Feasibility: `9/10`
  - Feasibility justification: For the reported shapes the assignment is redundant, so removing it is a safe local edit.
  - Implemented quick fixes for bounded local shapes:
    - remove a useless self-assignment statement
    - collapse a useless assignment expression to its underlying expression

- `python:S1045` - All `except` blocks should be able to catch exceptions
  - Feasibility: `3/10`
  - Feasibility justification: A fix is imaginable, but it is not clearly safe or canonical enough for automation.
  - Implemented targeted quick fixes for bounded local shapes:
    - remove an unreachable duplicate `except` clause
    - remove a redundant exception from a two-element exception tuple

- `python:S1066` - Mergeable `if` statements should be combined
  - Feasibility: `3/10`
  - Feasibility justification: A fix is imaginable, but it is not clearly safe or canonical enough for automation.
  - Implemented a quick fix for the supported nested-`if` shape already reported by the rule.
  - The fix merges the conditions with `and` and shifts the nested body left.

- `python:S108` - Nested blocks of code should not be left empty
  - Feasibility: `7/10`
  - Feasibility justification: Adding a TODO comment is algorithmic, but it is only a placeholder fix.
  - Quick fix already existed.
  - Verified during the investigation; not reimplemented.

- `python:S1244` - Floating point numbers should not be tested for equality
  - Feasibility: `8/10`
  - Feasibility justification: The analyzer can rewrite to `isclose`, but it depends on local call context.
  - Quick fix already existed.
  - Verified during the investigation; not reimplemented.

- `python:S1481` - Unused local variables should be removed
  - Feasibility: `8/10`
  - Feasibility justification: The existing fix is local, but only safe in guarded assignment shapes.
  - Quick fix already existed.
  - Verified during the investigation; not reimplemented.

- `python:S1192` - String literals should not be duplicated
  - Feasibility: `3/10`
  - Feasibility justification: A fix is imaginable, but it is not clearly safe or canonical enough for automation.
  - Implemented a targeted constant-extraction quick fix.
  - The fix is offered when all duplicated occurrences belong to the same statement list and the literal source can be reinserted safely on one line.
  - The implementation derives an upper-snake-case constant name from the literal text, avoids local/global name collisions, inserts the assignment in the right scope, and preserves module/function docstrings by inserting after them.
  - Current supported shapes include:
    - repeated literals inside one function body
    - repeated top-level literals in one module block
  - Current non-goals for this batch:
    - duplicates spread across different scopes
    - multiline literal sources that cannot be safely reinserted as one-line assignments

- `python:S1226` - Function parameters initial values should not be ignored
  - Feasibility: `4/10`
  - Feasibility justification: Several plausible edits exist, so there is no single canonical rewrite.
  - Implemented a targeted rename-based quick fix.
  - The fix preserves the original parameter value by renaming the first reassignment and all subsequent usages of the reassigned value to a fresh local name such as `p_value`.
  - It currently targets bounded local shapes where the first reassignment is one of:
    - a standard assignment target
    - an assignment expression target
    - a loop target
    - a class declaration reusing the parameter name
  - The implementation also resolves name collisions by suffixing the generated local name when needed.

## Not Implemented

The following rules were intentionally left without a new quick fix:

- `python:S112` - `"Exception"` and `"BaseException"` should not be raised
  - Feasibility: `1/10`
  - Feasibility justification: The rule still needs human judgment rather than one canonical edit.
  - No safe canonical replacement can be derived automatically.
  - The analyzer would have to invent a more specific exception type.

- `python:S1134` - Track uses of `FIXME` tags
  - Feasibility: `1/10`
  - Feasibility justification: The rule still needs human judgment rather than one canonical edit.
  - This is not a code transformation problem.
  - A responsible fix requires human judgment about the unfinished work.

- `python:S1135` - Track uses of `TODO` tags
  - Feasibility: `1/10`
  - Feasibility justification: The rule still needs human judgment rather than one canonical edit.
  - Same issue as `S1134`.
  - The analyzer cannot complete the missing task automatically.

## Test Workflow Used

For the implemented batch, the workflow was:

1. Add a reproducer / quick-fix test.
2. Run the targeted tests and confirm failure.
3. Implement the quick fix.
4. Re-run the targeted tests until green.

Targeted verification command:

```bash
mvn -pl python-checks -am -Dtest=ClassNameCheckTest,LocalVariableAndParameterNameConventionCheckTest,FunctionNameCheckTest,SelfAssignmentCheckTest,UnreachableExceptCheckTest,CollapsibleIfStatementsCheckTest,StringLiteralDuplicationCheckTest,IgnoredParameterCheckTest -Dsurefire.failIfNoSpecifiedTests=false test
```

This targeted slice passed after the implementation changes.
