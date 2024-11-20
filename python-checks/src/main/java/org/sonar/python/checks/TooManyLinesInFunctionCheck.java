/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.metrics.FileLinesVisitor.countDocstringLines;
import static org.sonar.python.metrics.FileLinesVisitor.tokenLineNumbers;

@Rule(key = "S138")
public class TooManyLinesInFunctionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "This %1$s \"%2$s\" has %3$d lines of code, " +
    "which is greater than the %4$d authorized. Split it into smaller %1$ss.";

  private static final int DEFAULT = 100;

  @RuleProperty(
    key = "max",
    description = "Maximum authorized lines of code in a function",
    defaultValue = "" + DEFAULT)
  public int max = DEFAULT;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      FunctionLineVisitor visitor = new FunctionLineVisitor();
      visitor.scan(functionDef.body());

      Set<Integer> linesOfCode = visitor.linesOfCode;
      Set<Integer> linesOfDocstring = countDocstringLines(functionDef.docstring());

      for (Integer line : linesOfDocstring) {
        linesOfCode.remove(line);
      }

      if (linesOfCode.size() > max) {
        ctx.addIssue(functionDef.name(), getIssueMessage(functionDef, linesOfCode.size()));
      }
    });
  }

  private String getIssueMessage(FunctionDef functionDef, int numberOfLines) {
    String type = functionDef.isMethodDefinition() ? "method" : "function";
    return String.format(MESSAGE, type, functionDef.name().name(), numberOfLines, max);
  }

  private static class FunctionLineVisitor {

    private final Set<Integer> linesOfCode = new HashSet<>();

    private void scan(Tree element) {
      Deque<Tree> stack = new ArrayDeque<>();
      stack.push(element);
      while (!stack.isEmpty()) {
        Tree currentElement = stack.pop();
        if (currentElement.is(Tree.Kind.TOKEN)) {
          linesOfCode.addAll(tokenLineNumbers((Token) currentElement));
        }

        currentElement.children()
          .stream().filter(Objects::nonNull)
          .forEach(stack::push);
      }
    }
  }
}
