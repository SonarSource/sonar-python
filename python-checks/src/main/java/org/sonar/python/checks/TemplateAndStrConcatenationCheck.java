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
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree.Kind;

@Rule(key = "S7943")
public class TemplateAndStrConcatenationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Template strings should not be concatenated with regular strings.";

  private boolean isPython314OrGreater = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Kind.PLUS, this::checkStringConcatenation);
  }

  private void initializeState(SubscriptionContext ctx) {
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);
  }

  private void checkStringConcatenation(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    
    BinaryExpression binaryExpression = (BinaryExpression) ctx.syntaxNode();
    Expression leftOperand = binaryExpression.leftOperand();
    Expression rightOperand = binaryExpression.rightOperand();
    
    if (leftOperand.is(Kind.STRING_LITERAL) && rightOperand.is(Kind.STRING_LITERAL)) {
      StringLiteral leftString = (StringLiteral) leftOperand;
      StringLiteral rightString = (StringLiteral) rightOperand;
      
      boolean isLeftTemplate = isTemplateString(leftString);
      boolean isRightTemplate = isTemplateString(rightString);
      
      if (isLeftTemplate != isRightTemplate) {
        ctx.addIssue(binaryExpression, MESSAGE)
          .secondary(leftString, isLeftTemplate ? "Template string" : "Regular string")
          .secondary(rightString, isRightTemplate ? "Template string" : "Regular string");
      }
    }
  }
  
  private static boolean isTemplateString(StringLiteral stringLiteral) {
    return stringLiteral.stringElements().stream()
      .anyMatch(StringElement::isTemplate);
  }
}
