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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

@Rule(key = "S3403")
public class SillyIdentityCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IS, ctx -> {
      IsExpression isExpression = (IsExpression) ctx.syntaxNode();
      InferredType left = isExpression.leftOperand().type();
      InferredType right = isExpression.rightOperand().type();
      if (!left.isIdentityComparableWith(right) && !left.canOnlyBe(NONE_TYPE) && !right.canOnlyBe(NONE_TYPE)) {
        Token notToken = isExpression.notToken();
        String operator = notToken == null ? "is" : "is not";
        String result = notToken == null ? "False" : "True";
        Token lastToken = notToken == null ? isExpression.operator() : notToken;
        ctx.addIssue(isExpression.operator(), lastToken, String.format("Remove this \"%s\" check; it will always be %s.", operator, result));
      }
    });
  }

}
