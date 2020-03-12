/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.plugins.python.api.tree.Tree.Kind.CLASSDEF;
import static org.sonar.python.tree.TreeUtils.getClassSymbolFromDef;

@Rule(key = "S4487")
public class UnreadPrivateAttributesCheck extends PythonSubscriptionCheck {

  private static final boolean DEFAULT_ENABLE_SINGLE_UNDERSCORE_ISSUES = false;

  @RuleProperty(
    key = "enableSingleUnderscoreIssues",
    description = "Enable issues on unread attributes with a single underscore prefix",
    defaultValue = "" + DEFAULT_ENABLE_SINGLE_UNDERSCORE_ISSUES)
  public boolean enableSingleUnderscoreIssues = DEFAULT_ENABLE_SINGLE_UNDERSCORE_ISSUES;

  @Override
  public void initialize(Context context) {
    String memberPrefix = enableSingleUnderscoreIssues ? "_" : "__";
    context.registerSyntaxNodeConsumer(CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      Optional.ofNullable(getClassSymbolFromDef(classDef)).ifPresent(classSymbol -> classSymbol.declaredMembers().stream()
        .filter(s -> s.name().startsWith(memberPrefix) && !s.name().endsWith("__") && s.kind() != FUNCTION && isNeverRead(s))
        .forEach(symbol -> reportIssue(ctx, symbol)));
    });
  }

  private static void reportIssue(SubscriptionContext ctx, Symbol symbol) {
    PreciseIssue preciseIssue = null;
    for (int i = 0; i < symbol.usages().size(); i++) {
      Usage usage = symbol.usages().get(i);
      if (i == 0) {
        preciseIssue = ctx.addIssue(usage.tree(), "Remove this unread private attribute '" + symbol.name() + "' or refactor the code to use its value.");
      } else {
        preciseIssue.secondary(usage.tree(), null);
      }
    }
  }

  private static boolean isNeverRead(Symbol symbol) {
    return symbol.usages().stream().allMatch(Usage::isBindingUsage);
  }
}
