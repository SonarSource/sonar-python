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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;

@Rule(key = SameConditionCheck.CHECK_KEY)
public class SameConditionCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S1862";
  private static final String MESSAGE = "This branch duplicates the one on line %s.";

  private List<IfStatement> ignoreList;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> this.ignoreList = new ArrayList<>());

    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> {
      IfStatement ifStatement = (IfStatement) ctx.syntaxNode();
      if (ignoreList.contains(ifStatement)) {
        return;
      }
      List<Expression> conditions = getConditionsToCompare(ifStatement);
      findSameConditions(conditions, ctx);
    });
  }

  private List<Expression> getConditionsToCompare(IfStatement ifStatement) {
    List<Expression> conditions = new ArrayList<>();
    conditions.add(ifStatement.condition());
    conditions.addAll(ifStatement.elifBranches().stream().map(IfStatement::condition).toList());
    ElseClause elseClause = ifStatement.elseBranch();
    if (elseClause != null) {
      lookForElseIfs(conditions, elseClause);
    }
    return conditions;
  }

  private void lookForElseIfs(List<Expression> conditions, ElseClause elseBranch) {
    IfStatement singleIfChild = singleIfChild(elseBranch.body());
    if (singleIfChild != null) {
      ignoreList.add(singleIfChild);
      conditions.addAll(getConditionsToCompare(singleIfChild));
    }
  }

  private static void findSameConditions(List<Expression> conditions, SubscriptionContext ctx) {
    for (int i = 1; i < conditions.size(); i++) {
      compareConditions(conditions, i, ctx);
    }
  }

  private static void compareConditions(List<Expression> conditions, int index, SubscriptionContext ctx) {
    for (int j = 0; j < index; j++) {
      if (CheckUtils.areEquivalent(conditions.get(j), conditions.get(index))) {
        String message = String.format(MESSAGE, conditions.get(j).firstToken().line());
        ctx.addIssue(conditions.get(index), message).secondary(conditions.get(j), "Original");
        return;
      }
    }
  }

  private static IfStatement singleIfChild(StatementList statementList) {
    List<Statement> statements = statementList.statements();
    if (statements.size() == 1 && statements.get(0).is(Tree.Kind.IF_STMT)) {
      return (IfStatement) statements.get(0);
    }
    return null;
  }
}
