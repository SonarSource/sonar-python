/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
