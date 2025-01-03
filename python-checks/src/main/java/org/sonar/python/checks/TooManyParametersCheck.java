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

import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.FunctionDefImpl;

@Rule(key = TooManyParametersCheck.CHECK_KEY)
public class TooManyParametersCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S107";
  private static final String MESSAGE = "%s has %s parameters, which is greater than the %s authorized.";

  private static final int DEFAULT_MAX = 13;

  @RuleProperty(
    key = "max",
    description = "Maximum authorized number of parameters",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, this::checkFunctionDef);

    context.registerSyntaxNodeConsumer(Kind.LAMBDA, ctx -> {
      LambdaExpression tree = (LambdaExpression) ctx.syntaxNode();
      ParameterList parameters = tree.parameters();
      if (parameters != null) {
        int nbParameters = parameters.all().size();
        if (nbParameters > max) {
          String name = "Lambda";
          String message = String.format(MESSAGE, name, nbParameters, max);
          ctx.addIssue(parameters, message);
        }
      }
    });
  }

  private void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    ParameterList parameters = functionDef.parameters();
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    if (parameters != null && functionSymbol != null) {
      long nbParameters = functionSymbol.parameters().size();
      boolean isMethod = functionDef.isMethodDefinition();
      if (isMethod && functionSymbol.decorators().stream().noneMatch(d -> d.contains("staticmethod"))) {
        // First parameter is implicitly passed: either "self" or "cls"
        nbParameters -= 1;
      }
      if (nbParameters > max) {
        if (isMethod && isAlreadyReportedInParent(functionSymbol)) {
          return;
        }
        String typeName = isMethod ? "Method" : "Function";
        String name = String.format("%s \"%s\"", typeName, functionDef.name().name());
        String message = String.format(MESSAGE, name, nbParameters, max);
        ctx.addIssue(parameters, message);
      }
    }
  }

  private boolean isAlreadyReportedInParent(FunctionSymbol functionSymbol) {
    // If the rule would already raise an issue on a parent class, don't raise the issue twice
    return SymbolUtils.getOverriddenMethods(functionSymbol).stream().findFirst().map(f -> {
      int nbParameters = f.parameters().size();
      if (f.decorators().stream().anyMatch(d -> d.contains("staticmethod"))) {
        nbParameters -= 1;
      }
      return nbParameters > max;
    }).orElse(false);
  }
}
