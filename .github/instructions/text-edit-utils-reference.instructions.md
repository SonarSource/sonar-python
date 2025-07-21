---
description: Rule should be used anytime quickfixes are mentionned
applyTo: "**"
---
# TextEditUtils Reference

This rule provides a quick reference for all public methods in [TextEditUtils.java](../../python-frontend/src/main/java/org/sonar/python/quickfix/TextEditUtils.java). For best practices on creating quick-fixes, see [Python Plugin Quick-Fix Creation Guidelines](./quickfix-guidelines.instructions.md).

## Methods

### insertLineBefore(Tree tree, String textToInsert): PythonTextEdit
Insert a line with the same indent as `tree` before it (applies offset to multiline insertions).
Example:
```java
quickFix.addTextEdit(TextEditUtils.insertLineBefore(node, "import os"));
```

### insertLineAfter(Tree tree, Tree indentReference, String textToInsert): PythonTextEdit
Insert a line with the same indent as `indentReference` after `tree` (applies offset to multiline insertions).
Example:
```java
quickFix.addTextEdit(TextEditUtils.insertLineAfter(statement, referenceNode, "print('Done')"));
```

### insertBefore(Tree tree, String textToInsert): PythonTextEdit
Insert `textToInsert` at the start position of `tree`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.insertBefore(expression, "0 and "));
```

### insertAfter(Tree tree, String textToInsert): PythonTextEdit
Insert `textToInsert` immediately after the first token of `tree`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.insertAfter(identifier, "_suffix"));
```

### insertAtPosition(PythonLine line, int column, String textToInsert): PythonTextEdit
Insert `textToInsert` at the given line and column.
Example:
```java
quickFix.addTextEdit(TextEditUtils.insertAtPosition(line, 4, "# TODO"));
```

### replace(Tree toReplace, String replacementText): PythonTextEdit
Replace the code represented by `toReplace` with `replacementText`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.replace(oldCall, "new_call()"));
```

### replaceRange(Tree start, Tree end, String replacementText): PythonTextEdit
Replace the code from the start of `start` to the end of `end` with `replacementText`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.replaceRange(nodeStart, nodeEnd, "combined()"));
```

### shiftLeft(StatementList stmtList): List<PythonTextEdit>
Shift each statement in `stmtList` one level left (decrease indent).
Example:
```java
quickFix.addTextEdits(TextEditUtils.shiftLeft(block));
```

### shiftLeft(Tree tree, int offset): List<PythonTextEdit>
Remove `offset` spaces from all lines of `tree` tokens.
Example:
```java
quickFix.addTextEdits(TextEditUtils.shiftLeft(statement, indentDelta));
```

### removeRange(PythonLine startLine, int startColumn, PythonLine endLine, int endColumn): PythonTextEdit
Remove code from `startLine:startColumn` to `endLine:endColumn`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.removeRange(start, 0, end, 2));
```

### removeUntil(Tree start, Tree until): PythonTextEdit
Remove code from the start of `start` until the first token of `until`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.removeUntil(firstStmt, lastStmt));
```

### removeStatement(Statement stmt): PythonTextEdit
Remove or replace `stmt` according to context:
- Single in block: replaces with `pass`
- First on line: strips tokens including separators
- Last on line: strips trailing whitespace and preserves line break
- Only statement: removes entire line
Example:
```java
quickFix.addTextEdit(TextEditUtils.removeStatement(stmt));
```

### remove(Tree toRemove): PythonTextEdit
Remove the code represented by `toRemove`.
Example:
```java
quickFix.addTextEdit(TextEditUtils.remove(node));
```

### renameAllUsages(HasSymbol node, String newName): List<PythonTextEdit>
Rename all usages of the symbol defined by `node` to `newName`.
Example:
```java
quickFix.addTextEdits(TextEditUtils.renameAllUsages(varNode, "newVar"));
```
