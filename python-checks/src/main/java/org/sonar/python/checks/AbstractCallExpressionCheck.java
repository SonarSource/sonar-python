/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;

public abstract class AbstractCallExpressionCheck extends PythonSubscriptionCheck {

  protected abstract Set<String> functionsToCheck();

  protected abstract String message();

  protected boolean isException(CallExpression callExpression) {
    return false;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  public void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Symbol symbol = node.calleeSymbol();
    if (!isException(node) && symbol != null && functionsToCheck().contains(symbol.fullyQualifiedName())) {
      ctx.addIssue(node.callee(), message());
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
