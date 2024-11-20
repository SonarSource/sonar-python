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
package org.sonar.python.checks.hotspots;

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

@Rule(key = "S2257")
public class NonStandardCryptographicAlgorithmCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Make sure using a non-standard cryptographic algorithm is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, NonStandardCryptographicAlgorithmCheck::checkCreatingCustomHasher);
  }

  private static String getQualifiedName(Expression node) {
    if (node instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      return symbol != null ? symbol.fullyQualifiedName() : "";
    }
    return "";
  }

  private static void checkCreatingCustomHasher(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();
    String qualifiedName = getQualifiedName(classDef.name());
    if (qualifiedName != null && qualifiedName.startsWith("django.contrib.auth.hashers")) {
      return;
    }
    ArgList argList = classDef.args();
    if (argList != null) {
      argList.arguments()
        .stream()
        .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .filter(arg -> "django.contrib.auth.hashers.BasePasswordHasher".equals(getQualifiedName(arg.expression())))
        .forEach(arg -> ctx.addIssue(arg, MESSAGE));
    }
  }
}
