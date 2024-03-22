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
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S5709")
public class ExceptionSuperClassDeclarationCheck extends PythonSubscriptionCheck {

  private static final Set<String> FORBIDDEN_SUPER_CLASS_FQNS = new HashSet<>(Arrays.asList(
    "BaseException",
    "GeneratorExit",
    "KeyboardInterrupt",
    "SystemExit"
  ));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      ArgList args = classDef.args();
      if (args != null) {
        args.arguments().stream()
          .filter(RegularArgument.class::isInstance)
          .map(RegularArgument.class::cast)
          .forEach(arg -> checkSuperClass(arg, ctx));
      }
    });
  }

  private static void checkSuperClass(RegularArgument arg, SubscriptionContext ctx) {
    Expression expression = arg.expression();
    if (expression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol != null && FORBIDDEN_SUPER_CLASS_FQNS.contains(symbol.fullyQualifiedName())) {
        ctx.addIssue(arg, String.format("Derive this class from \"Exception\" instead of \"%s\".", symbol.fullyQualifiedName()));
      }
    }
  }

}
