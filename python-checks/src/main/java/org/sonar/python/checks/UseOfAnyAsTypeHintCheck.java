/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6542")
public class UseOfAnyAsTypeHintCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a more specific type than `Any` for this type hint.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      TypeAnnotation typeAnnotation = functionDef.returnTypeAnnotation();
      if (isTypeAny(typeAnnotation)) {
        ctx.addIssue(typeAnnotation.expression(), MESSAGE);
      }
      checkForAnyInParameters(ctx, functionDef.parameters());
    });
  }

  private static void checkForAnyInParameters(SubscriptionContext ctx, @Nullable ParameterList parameterList) {
    if (parameterList != null) {
      List<Parameter> parameters = parameterList.nonTuple();
      parameters.forEach(parameter -> {
        TypeAnnotation typeAnnotation = parameter.typeAnnotation();
        if (isTypeAny(typeAnnotation)) {
          ctx.addIssue(typeAnnotation.expression(), MESSAGE);
        }
      });
    }
  }

  private static boolean isTypeAny(@Nullable TypeAnnotation typeAnnotation) {
    return typeAnnotation != null && "typing.Any".equals(TreeUtils.fullyQualifiedNameFromExpression(typeAnnotation.expression()));
  }
}
