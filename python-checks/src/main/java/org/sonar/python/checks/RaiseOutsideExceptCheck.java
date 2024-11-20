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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_GROUP_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FINALLY_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FUNCDEF;
import static org.sonar.plugins.python.api.tree.Tree.Kind.RAISE_STMT;

@Rule(key = "S5747")
public class RaiseOutsideExceptCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(RAISE_STMT, ctx -> {
      RaiseStatement raiseStatement = (RaiseStatement) ctx.syntaxNode();
      if (!raiseStatement.expressions().isEmpty()) {
        return;
      }
      if (!isInsideExceptOrFinally(raiseStatement)) {
        ctx.addIssue(raiseStatement, "Remove this \"raise\" statement or move it inside an \"except\" block.");
      }

    });
  }

  private static boolean isInsideExceptOrFinally(Tree tree) {
    Tree parent = tree.parent();
    while (parent != null) {
      if ((parent.is(FUNCDEF))) {
        // __exit__ special method is handled by S5706
        return ((FunctionDef) parent).name().name().equals("__exit__") || isFunctionCalledInsideExceptBlock(((FunctionDef) parent));
      }
      if (parent.is(EXCEPT_CLAUSE, EXCEPT_GROUP_CLAUSE, FINALLY_CLAUSE)) {
        return true;
      }
      parent = parent.parent();
    }
    return false;
  }

  private static boolean isFunctionCalledInsideExceptBlock(FunctionDef functionDef) {
    Symbol symbol = functionDef.name().symbol();
    return symbol != null && symbol.usages().stream()
      .filter(usage -> usage.tree() != functionDef.name())
      .anyMatch(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), EXCEPT_CLAUSE, EXCEPT_GROUP_CLAUSE, FINALLY_CLAUSE) != null);
  }
}
