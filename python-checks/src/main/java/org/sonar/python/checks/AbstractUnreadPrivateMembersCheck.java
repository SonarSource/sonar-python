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

import java.util.Optional;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CLASSDEF;
import static org.sonar.python.tree.TreeUtils.getClassSymbolFromDef;

public abstract class AbstractUnreadPrivateMembersCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    String memberPrefix = memberPrefix();
    context.registerSyntaxNodeConsumer(CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      Optional.ofNullable(getClassSymbolFromDef(classDef)).ifPresent(classSymbol -> classSymbol.declaredMembers().stream()
        .filter(s -> s.name().startsWith(memberPrefix) && !s.name().endsWith("__") && equalsToKind(s) && isNeverRead(s))
        .forEach(symbol -> reportIssue(ctx, symbol)));
    });
  }

  private boolean equalsToKind(Symbol symbol) {
    if (symbol.kind().equals(Symbol.Kind.AMBIGUOUS)) {
      return ((AmbiguousSymbol) symbol).alternatives().stream().allMatch(s -> s.kind() == kind());
    }
    return symbol.kind() == kind();
  }

  private void reportIssue(SubscriptionContext ctx, Symbol symbol) {
    PreciseIssue preciseIssue = null;
    for (int i = 0; i < symbol.usages().size(); i++) {
      Usage usage = symbol.usages().get(i);
      if (i == 0) {
        preciseIssue = ctx.addIssue(usage.tree(), message(symbol.name()));
      } else {
        preciseIssue.secondary(usage.tree(), secondaryMessage());
      }
    }
  }

  private static boolean isNeverRead(Symbol symbol) {
    return symbol.usages().stream().allMatch(Usage::isBindingUsage);
  }

  abstract String memberPrefix();

  abstract Symbol.Kind kind();

  abstract String message(String memberName);

  abstract String secondaryMessage();
}
