/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.quickfix.TextEditUtils.insertLineBefore;

@Rule(key = "S1186")
public class EmptyFunctionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a nested comment explaining why this %s is empty, or complete the implementation.";
  private static final List<String> ABC_DECORATORS = List.of("abstractmethod", "abstractstaticmethod", "abstractproperty", "abstractclassmethod");

  private static final List<String> BINARY_MAGIC_METHODS = List.of("__add__", "__and__", "__cmp__", "__divmod__",
    "__div__", "__eq__", "__floordiv__", "__ge__", "__gt__", "__iadd__", "__iand__", "__idiv__", "__ifloordiv__",
    "__ilshift__", "__imod__", "__imul__", "__ior__", "__ipow__", "__irshift__", "__isub__", "__ixor__", "__le__",
    "__lshift__", "__lt__", "__mod__", "__mul__", "__ne__", "__or__", "__pow__", "__radd__", "__rand__", "__rdiv__",
    "__rfloordiv__", "__rlshift__", "__rmod__", "__rmul__", "__ror__", "__rpow__", "__rrshift__", "__rshift__", "__rsub__",
    "__rxor__", "__sub__", "__xor__");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (functionDef.decorators().stream()
        .map(d -> TreeUtils.decoratorNameFromExpression(d.expression()))
        .filter(Objects::nonNull)
        .flatMap(s -> Arrays.stream(s.split("\\.")))
        .anyMatch(ABC_DECORATORS::contains)) {
        return;
      }

      if (functionDef.body().statements().size() == 1 && functionDef.body().statements().get(0).is(Tree.Kind.PASS_STMT)) {
        if (TreeUtils.tokens(functionDef).stream().anyMatch(t -> !t.trivia().isEmpty())) {
          return;
        }
        if (hasCommentAbove(functionDef)) {
          return;
        }
        String type = functionDef.isMethodDefinition() ? "method" : "function";
        PreciseIssue issue = ctx.addIssue(functionDef.name(), String.format(MESSAGE, type));
        addQuickFixes(issue, functionDef, type);
      }
    });
  }

  private static void addQuickFixes(PreciseIssue issue, FunctionDef functionDef, String functionType) {
    Statement passStatement = functionDef.body().statements().get(0);
    issue.addQuickFix(PythonQuickFix.newQuickFix("Insert placeholder comment",
      insertLineBefore(passStatement, "# TODO document why this method is empty")));

    if (functionType.equals("method") && BINARY_MAGIC_METHODS.contains(functionDef.name().name())) {
      issue.addQuickFix(PythonQuickFix.newQuickFix("Return NotImplemented constant",
        insertLineBefore(passStatement, "return NotImplemented")));
    } else {
      issue.addQuickFix(PythonQuickFix.newQuickFix("Raise NotImplementedError()",
        insertLineBefore(passStatement, "raise NotImplementedError()")));
    }
  }

  private static boolean hasCommentAbove(FunctionDef functionDef) {
    Tree parent = functionDef.parent();
    List<Token> tokens = TreeUtils.tokens(parent);
    Token defKeyword = functionDef.defKeyword();
    int index = tokens.indexOf(defKeyword);
    if (index == 0) {
      parent = parent.parent();
      tokens = TreeUtils.tokens(parent);
      index = tokens.indexOf(defKeyword);
    }
    return index > 0 && !tokens.get(index - 1).trivia().isEmpty();
  }
}
