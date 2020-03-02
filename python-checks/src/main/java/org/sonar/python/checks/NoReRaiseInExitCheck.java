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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

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
      if (raisedException instanceof HasSymbol && Objects.equals(((HasSymbol) raisedException).symbol(), caughtException)) {
        nonCompliantRaises.add(pyRaiseStatementTree);
      }

      if (raisedException.is(Tree.Kind.SUBSCRIPTION)) {
        SubscriptionExpression subscriptionExpression = (SubscriptionExpression) raisedException;
        Expression objectExpression = subscriptionExpression.object();

        if (objectExpression instanceof HasSymbol && Objects.equals(((HasSymbol) objectExpression).symbol(), packedParameter)) {
          nonCompliantRaises.add(pyRaiseStatementTree);
        }
      }
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (!functionDef.name().name().equals("__exit__") || functionDef.parameters() == null) {
        return;
      }

      Symbol caughtException = null;
      Symbol packedParameter = null;
      if (functionDef.parameters().nonTuple().size() == 2) {
        AnyParameter possiblyPackedParam = functionDef.parameters().nonTuple().get(1);
        if (possiblyPackedParam.is(Tree.Kind.PARAMETER)) {
          Parameter parameter = (Parameter) possiblyPackedParam;
          if (parameter.starToken() == null) {
            return;
          }

          Name name = parameter.name();
          if (name != null) {
            packedParameter = parameter.name().symbol();
          }
        }
      } else if (functionDef.parameters().nonTuple().size() >= 3) {
        AnyParameter exceptionParam = functionDef.parameters().nonTuple().get(2);
        if (exceptionParam.is(Tree.Kind.PARAMETER)) {
          Name name = ((Parameter) exceptionParam).name();
          if (name != null) {
            caughtException = ((Parameter) exceptionParam).name().symbol();
          }
        }
      } else {
        return;
      }

      RaiseVisitor visitor = new RaiseVisitor(caughtException, packedParameter);
      functionDef.accept(visitor);

      for (RaiseStatement bareRaise : visitor.nonCompliantRaises) {
        ctx.addIssue(bareRaise, "Remove this \"raise\" statement and return \"False\" instead.");
      }
    });
  }
}
