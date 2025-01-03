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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key="S5706")
public class NoReRaiseInExitCheck extends PythonSubscriptionCheck {

  private static class RaiseVisitor extends BaseTreeVisitor {
    private Symbol caughtException;
    private Symbol packedParameter;

    private List<RaiseStatement> nonCompliantRaises = new ArrayList<>();

    public RaiseVisitor(Symbol caughtException, Symbol packedParameter) {
      this.caughtException = caughtException;
      this.packedParameter = packedParameter;
    }

    @Override
    public void visitExceptClause(ExceptClause exceptClause) {
      // Intentionally empty - we do not want to enter except blocks.
    }

    @Override
    public void visitRaiseStatement(RaiseStatement pyRaiseStatementTree) {
      if (pyRaiseStatementTree.expressions().isEmpty()) {
        nonCompliantRaises.add(pyRaiseStatementTree);
        return;
      }

      Expression raisedException = pyRaiseStatementTree.expressions().get(0);
      if (raisedException instanceof HasSymbol hasSymbol && Objects.equals(hasSymbol.symbol(), caughtException)) {
        nonCompliantRaises.add(pyRaiseStatementTree);
      }

      if (raisedException.is(Tree.Kind.SUBSCRIPTION)) {
        SubscriptionExpression subscriptionExpression = (SubscriptionExpression) raisedException;
        Expression objectExpression = subscriptionExpression.object();

        if (objectExpression instanceof HasSymbol hasSymbol && Objects.equals(hasSymbol.symbol(), packedParameter)) {
          nonCompliantRaises.add(pyRaiseStatementTree);
        }
      }
    }
  }

  private static Symbol extractPackedParameter(ParameterList parameterList) {
    if (parameterList.nonTuple().size() != 2) {
      return null;
    }

    Parameter parameter = parameterList.nonTuple().get(1);
    if (parameter.starToken() == null) {
      return null;
    }

    Name name = parameter.name();
    if (name == null) {
      return null;
    }
    return name.symbol();
  }

  private static Symbol extractCaughtExceptionParameter(ParameterList parameterList) {
    List<Parameter> regularParams = parameterList.nonTuple();
    if (regularParams.size() != 4 || !regularParams.equals(parameterList.all())) {
      // A valid signature has 4 parameters here: self, exc_type, exc_value, trace_back.
      // Bail out early if the __exit__ method is declared differently.
      return null;
    }

    Parameter parameter = regularParams.get(2);
    Name name = parameter.name();
    if (name == null) {
      return null;
    }
    return name.symbol();
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      ParameterList parameterList = functionDef.parameters();
      if (!functionDef.name().name().equals("__exit__") || parameterList == null) {
        return;
      }

      Symbol caughtException = extractCaughtExceptionParameter(parameterList);
      Symbol packedParameter = extractPackedParameter(parameterList);

      RaiseVisitor visitor = new RaiseVisitor(caughtException, packedParameter);
      functionDef.accept(visitor);

      for (RaiseStatement bareRaise : visitor.nonCompliantRaises) {
        ctx.addIssue(bareRaise, "Remove this \"raise\" statement and return \"False\" instead.");
      }
    });
  }
}
