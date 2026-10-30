/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key="S5706")
public class NoReRaiseInExitCheck extends PythonSubscriptionCheck {

  private static class RaiseVisitor extends BaseTreeVisitor {
    private final SymbolV2 caughtException;
    private final SymbolV2 packedParameter;

    private final List<RaiseStatement> nonCompliantRaises = new ArrayList<>();

    public RaiseVisitor(SymbolV2 caughtException, SymbolV2 packedParameter) {
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
      if (caughtException != null && raisedException instanceof Name name && caughtException.equals(name.symbolV2())) {
        nonCompliantRaises.add(pyRaiseStatementTree);
      }

      if (raisedException.is(Tree.Kind.SUBSCRIPTION)) {
        SubscriptionExpression subscriptionExpression = (SubscriptionExpression) raisedException;
        Expression objectExpression = subscriptionExpression.object();

        if (packedParameter != null && objectExpression instanceof Name name && packedParameter.equals(name.symbolV2())) {
          nonCompliantRaises.add(pyRaiseStatementTree);
        }
      }
    }
  }

  private static SymbolV2 extractPackedParameter(ParameterList parameterList) {
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
    return name.symbolV2();
  }

  private static SymbolV2 extractCaughtExceptionParameter(ParameterList parameterList) {
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
    return name.symbolV2();
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      ParameterList parameterList = functionDef.parameters();
      if (!"__exit__".equals(functionDef.name().name()) || parameterList == null) {
        return;
      }

      SymbolV2 caughtException = extractCaughtExceptionParameter(parameterList);
      SymbolV2 packedParameter = extractPackedParameter(parameterList);

      RaiseVisitor visitor = new RaiseVisitor(caughtException, packedParameter);
      functionDef.accept(visitor);

      for (RaiseStatement bareRaise : visitor.nonCompliantRaises) {
        ctx.addIssue(bareRaise, "Remove this \"raise\" statement and return \"False\" instead.");
      }
    });
  }
}
