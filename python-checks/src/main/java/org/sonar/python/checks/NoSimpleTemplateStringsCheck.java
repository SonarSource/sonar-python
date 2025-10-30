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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7944")
public class NoSimpleTemplateStringsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Template strings should not be used for simple string formatting.";

  private boolean isPython314OrGreater = false;
  private TypeCheckBuilder isStrCallCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Kind.STRING_ELEMENT, this::checkTemplateString);
  }

  private void initializeState(SubscriptionContext ctx) {
    isStrCallCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("str");
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);
  }

  private void checkTemplateString(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    StringElement stringElement = (StringElement) ctx.syntaxNode();
    if (stringElement.isTemplate() && isWithinStrCall(stringElement)) {
      ctx.addIssue(stringElement, MESSAGE);
    }
  }

  private boolean isWithinStrCall(Tree currentNode) {
    Tree callAncestor = TreeUtils.firstAncestorOfKind(currentNode, Kind.CALL_EXPR);
    if (callAncestor instanceof CallExpression callExpr) {
      return isStrCallCheck.check(callExpr.callee().typeV2()).isTrue();
    }
    return false;
  }   
}
