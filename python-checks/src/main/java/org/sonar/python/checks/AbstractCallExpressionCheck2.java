/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.semantic.Symbol;

public abstract class AbstractCallExpressionCheck2 extends PythonSubscriptionCheck {

  protected abstract Set<String> functionsToCheck();

  protected abstract String message();

  protected boolean isException(PyCallExpressionTree callExpression) {
    return false;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  public void visitNode(SubscriptionContext ctx) {
    PyCallExpressionTree node = (PyCallExpressionTree) ctx.syntaxNode();
    Symbol symbol = ctx.symbolTable().getSymbol(node);
    if (!isException(node) && symbol != null && functionsToCheck().contains(symbol.qualifiedName())) {
      ctx.addIssue(node, message());
    }
  }

  protected static boolean isWithinImport(Tree tree) {
    Tree parent = tree.parent();
    while (parent != null) {
      if (parent.is(Tree.Kind.IMPORT_NAME) || parent.is(Tree.Kind.IMPORT_FROM)) {
        return true;
      }
      parent = parent.parent();
    }
    return false;
  }

  protected static <T> Set<T> immutableSet(T...args) {
    return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(args)));
  }
}
