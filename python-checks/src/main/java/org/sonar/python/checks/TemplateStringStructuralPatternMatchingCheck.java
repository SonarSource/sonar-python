/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7945")
public class TemplateStringStructuralPatternMatchingCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use structural pattern matching (match/case) instead of isinstance() checks for template string processing.";
  private static final String SECONDARY_MESSAGE = "Replace this isinstance with the appropriate pattern matching case.";
  private TypeCheckMap<Object> templateType = new TypeCheckMap<>();
  private TypeCheckBuilder isInstanceCheck;
  private boolean isPython314OrGreater;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::checkForStatement);
  }

  private void initializeState(SubscriptionContext ctx) {
    isInstanceCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("isinstance");
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);

    Object marker = new Object();
    templateType.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("str"), marker);
    templateType.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("string.templatelib.Interpolation"), marker);
  }

  private void checkForStatement(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }

    ForStatement forStmt = (ForStatement) ctx.syntaxNode();
    StatementList body = forStmt.body();
    if (body == null || body.statements().isEmpty()) {
      return;
    }

    List<String> iterationVariables = extractIterationVariables(forStmt.expressions());
    if (iterationVariables.isEmpty()) {
      return;
    }

    List<IsInstanceCallAndType> isInstanceChecks = findIsInstanceChecks(body.statements(), iterationVariables);
    if (isInstanceChecks.size() >= 2 && hasTemplateTypeChecks(isInstanceChecks)) {
      PreciseIssue issue = ctx.addIssue(isInstanceChecks.get(0).callExpr.callee(), MESSAGE);
      isInstanceChecks.stream().skip(1).forEach(isInstance -> issue.secondary(isInstance.callExpr.callee(), SECONDARY_MESSAGE));
    }
  }

  private record IsInstanceCallAndType(CallExpression callExpr, Name typeName) {
  }

  private static List<String> extractIterationVariables(List<Expression> expressions) {
    List<String> variables = new ArrayList<>();
    for (Expression expr : expressions) {
      if (expr instanceof Name name) {
        variables.add(name.name());
      }
    }
    return variables;
  }

  private List<IsInstanceCallAndType> findIsInstanceChecks(List<Statement> statements, List<String> iterationVariables) {
    List<IsInstanceCallAndType> isInstanceChecks = new ArrayList<>();
    for (Statement stmt : statements) {
      if (stmt instanceof IfStatement ifStmt) {
        isInstanceChecks.addAll(extractIsInstanceChecks(ifStmt.condition(), iterationVariables));
        for (IfStatement elif : ifStmt.elifBranches()) {
          isInstanceChecks.addAll(extractIsInstanceChecks(elif.condition(), iterationVariables));
        }
        ElseClause elseClause = ifStmt.elseBranch();
        if (elseClause != null && elseClause.body() != null) {
          isInstanceChecks.addAll(findIsInstanceChecks(elseClause.body().statements(), iterationVariables));
        }
      }
    }
    return isInstanceChecks;
  }

  private List<IsInstanceCallAndType> extractIsInstanceChecks(Expression condition, List<String> iterationVariables) {
    List<IsInstanceCallAndType> checks = new ArrayList<>();
    if (condition instanceof CallExpression callExpr) {
      isIsInstanceCheck(callExpr, iterationVariables).ifPresent(checks::add);
    } else if (condition instanceof BinaryExpression binaryExpr) {
      checks.addAll(extractIsInstanceChecks(binaryExpr.leftOperand(), iterationVariables));
      checks.addAll(extractIsInstanceChecks(binaryExpr.rightOperand(), iterationVariables));
    }
    return checks;
  }

  private Optional<IsInstanceCallAndType> isIsInstanceCheck(CallExpression callExpr, List<String> iterationVariables) {
    if (!isInstanceCheck.check(callExpr.callee().typeV2()).isTrue()) {
      return Optional.empty();
    }

    List<RegularArgument> args = callExpr.arguments().stream()
      .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .toList();

    if (args.size() != 2) {
      return Optional.empty();
    }

    Expression firstArg = args.get(0).expression();
    if (!(firstArg instanceof Name varName)) {
      return Optional.empty();
    }
    if (!iterationVariables.contains(varName.name())) {
      return Optional.empty();
    }
    Expression secondArg = args.get(1).expression();
    if (!(secondArg instanceof Name typeName)) {
      return Optional.empty();
    }

    return Optional.of(new IsInstanceCallAndType(callExpr, typeName));
  }

  private boolean hasTemplateTypeChecks(List<IsInstanceCallAndType> isInstanceChecks) {
    long templateTypeCount = isInstanceChecks.stream()
      .map(instanceCallAndType -> instanceCallAndType.typeName.typeV2())
      .filter(templateType::containsForType)
      .count();
    return templateTypeCount >= 2;
  }

}
