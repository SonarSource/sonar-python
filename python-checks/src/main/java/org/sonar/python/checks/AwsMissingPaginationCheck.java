/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7622")
public class AwsMissingPaginationCheck extends PythonSubscriptionCheck {

  private static final Set<String> SENSITIVE_METHODS_FQNS = Set.of(
    "botocore.client.BaseClient.list_objects_v2",
    "botocore.client.BaseClient.scan"
  );

  private TypeCheckMap<Boolean> sensitiveTypesCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initializeCheck(SubscriptionContext ctx) {
    sensitiveTypesCheckMap = new TypeCheckMap<>();
    SENSITIVE_METHODS_FQNS.stream()
      .map(fqn -> ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn))
      .forEach(check -> sensitiveTypesCheckMap.put(check, true));

  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    if (sensitiveTypesCheckMap.containsForType(TreeUtils.inferSingleAssignedExpressionType(callExpression.callee()))) {
      ctx.addIssue(callExpression, "Use a paginator to retrieve all results from this boto3 operation.");
    }
  }


}
