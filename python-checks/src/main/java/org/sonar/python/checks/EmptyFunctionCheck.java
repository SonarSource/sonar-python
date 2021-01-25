/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1186")
public class EmptyFunctionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a nested comment explaining why this %s is empty, or complete the implementation.";
  private static final List<String> ABC_DECORATORS = Arrays.asList("abstractmethod", "abstractstaticmethod", "abstractproperty", "abstractclassmethod");

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
        ctx.addIssue(functionDef.name(), String.format(MESSAGE, type));
      }
    });
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
