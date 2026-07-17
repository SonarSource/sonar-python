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
package org.sonar.python.checks.tests;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FinallyClause;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S8994")
public class PytestFixtureMultipleYieldCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Pytest fixtures should contain at most one yield statement.";
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN("pytest.fixture");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, PytestFixtureMultipleYieldCheck::checkFunctionDef);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isPytestFixture(functionDef, ctx)) {
      return;
    }

    YieldPathAnalyzer yieldPathAnalyzer = new YieldPathAnalyzer();
    yieldPathAnalyzer.analyze(functionDef.body());
    for (YieldStatement yieldStatement : yieldPathAnalyzer.violations()) {
      ctx.addIssue(yieldStatement, MESSAGE);
    }
  }

  private static boolean isPytestFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream().anyMatch(decorator -> matchesPytestFixtureDecorator(decorator, ctx));
  }

  private static boolean matchesPytestFixtureDecorator(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return PYTEST_FIXTURE_MATCHER.isTrueFor(callExpression.callee(), ctx);
    }
    return PYTEST_FIXTURE_MATCHER.isTrueFor(expression, ctx);
  }

  private static final class YieldPathAnalyzer {

    private final List<YieldStatement> violations = new ArrayList<>();

    void analyze(StatementList body) {
      analyzeStatements(body.statements(), 0);
    }

    List<YieldStatement> violations() {
      return violations;
    }

    private int analyzeStatements(List<Statement> statements, int yieldsBefore) {
      int yieldsSoFar = yieldsBefore;
      for (Statement statement : statements) {
        yieldsSoFar = analyzeStatement(statement, yieldsSoFar);
      }
      return yieldsSoFar;
    }

    private int analyzeStatement(Statement statement, int yieldsBefore) {
      if (statement instanceof YieldStatement yieldStatement) {
        if (yieldsBefore >= 1) {
          violations.add(yieldStatement);
        }
        return yieldsBefore + 1;
      }
      if (statement instanceof IfStatement ifStatement) {
        return analyzeIfStatement(ifStatement, yieldsBefore);
      }
      if (statement instanceof TryStatement tryStatement) {
        return analyzeTryStatement(tryStatement, yieldsBefore);
      }
      if (statement instanceof ForStatement forStatement) {
        return analyzeLoopBody(forStatement.body(), forStatement.elseClause(), yieldsBefore);
      }
      if (statement instanceof WhileStatement whileStatement) {
        return analyzeLoopBody(whileStatement.body(), whileStatement.elseClause(), yieldsBefore);
      }
      if (statement instanceof WithStatement withStatement) {
        return analyzeStatements(withStatement.statements().statements(), yieldsBefore);
      }
      if (statement instanceof FunctionDef) {
        return yieldsBefore;
      }
      return yieldsBefore;
    }

    private int analyzeIfStatement(IfStatement ifStatement, int yieldsBefore) {
      if (ifStatement.elseBranch() != null || !ifStatement.elifBranches().isEmpty()) {
        return analyzeExclusiveBranches(ifStatement, yieldsBefore);
      }
      BranchResult ifBranch = analyzeBranch(ifStatement.body(), yieldsBefore);
      return YieldPathAnalyzer.mergeBranchOutcomes(yieldsBefore, List.of(ifBranch));
    }

    private int analyzeExclusiveBranches(IfStatement ifStatement, int yieldsBefore) {
      List<BranchResult> branchResults = new ArrayList<>();
      branchResults.add(analyzeBranch(ifStatement.body(), yieldsBefore));
      for (IfStatement elifBranch : ifStatement.elifBranches()) {
        branchResults.add(analyzeBranch(elifBranch.body(), yieldsBefore));
      }
      ElseClause elseBranch = ifStatement.elseBranch();
      if (elseBranch != null) {
        branchResults.add(analyzeBranch(elseBranch.body(), yieldsBefore));
      }
      return YieldPathAnalyzer.mergeBranchOutcomes(yieldsBefore, branchResults);
    }

    private static int mergeBranchOutcomes(int yieldsBefore, List<BranchResult> branchResults) {
      int maxAfter = yieldsBefore;
      for (BranchResult branchResult : branchResults) {
        if (branchResult.exits()) {
          maxAfter = Math.max(maxAfter, yieldsBefore);
        } else {
          maxAfter = Math.max(maxAfter, branchResult.yieldsAfter());
        }
      }
      return maxAfter;
    }

    private int analyzeTryStatement(TryStatement tryStatement, int yieldsBefore) {
      List<BranchResult> branchResults = new ArrayList<>();
      branchResults.add(analyzeBranch(tryStatement.body(), yieldsBefore));
      for (ExceptClause exceptClause : tryStatement.exceptClauses()) {
        branchResults.add(analyzeBranch(exceptClause.body(), yieldsBefore));
      }
      int maxAfter = YieldPathAnalyzer.mergeBranchOutcomes(yieldsBefore, branchResults);
      ElseClause elseClause = tryStatement.elseClause();
      if (elseClause != null) {
        maxAfter = analyzeStatements(elseClause.body().statements(), maxAfter);
      }
      FinallyClause finallyClause = tryStatement.finallyClause();
      if (finallyClause != null) {
        maxAfter = analyzeStatements(finallyClause.body().statements(), maxAfter);
      }
      return maxAfter;
    }

    private int analyzeLoopBody(StatementList body, @Nullable ElseClause elseClause, int yieldsBefore) {
      int maxAfter = analyzeStatements(body.statements(), yieldsBefore);
      if (elseClause != null) {
        maxAfter = Math.max(maxAfter, analyzeStatements(elseClause.body().statements(), yieldsBefore));
      }
      return maxAfter;
    }

    private BranchResult analyzeBranch(StatementList body, int yieldsBefore) {
      int yieldsAfter = analyzeStatements(body.statements(), yieldsBefore);
      return new BranchResult(yieldsAfter, branchExits(body));
    }

    private static boolean branchExits(StatementList body) {
      for (Statement statement : body.statements()) {
        if (statement instanceof ReturnStatement || statement instanceof RaiseStatement) {
          return true;
        }
        if (statement instanceof IfStatement ifStatement) {
          if (branchExits(ifStatement.body()) || ifStatement.elifBranches().stream().anyMatch(elif -> branchExits(elif.body()))) {
            return true;
          }
          ElseClause elseBranch = ifStatement.elseBranch();
          if (elseBranch != null && branchExits(elseBranch.body())) {
            return true;
          }
        }
      }
      return false;
    }

    private record BranchResult(int yieldsAfter, boolean exits) {}
  }
}
