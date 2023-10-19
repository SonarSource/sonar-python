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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6796")
public class GenericFunctionTypeParameterCheck extends PythonSubscriptionCheck {

  private static String MESSAGE = "Use a generic type parameter for this function instead of a TypeVar.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_TYPE_ANNOTATION, GenericFunctionTypeParameterCheck::checkUseOfTypeVarGeneric);
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_TYPE_ANNOTATION, GenericFunctionTypeParameterCheck::checkUseOfTypeVarGeneric);
  }

  private static void checkUseOfTypeVarGeneric(SubscriptionContext ctx) {
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    TypeAnnotation typeAnnotation = (TypeAnnotation) ctx.syntaxNode();
    Optional.of(typeAnnotation.expression())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Expressions::singleAssignedValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("typing.TypeVar"::equals)
      .ifPresent(fqn -> ctx.addIssue(typeAnnotation.expression(), MESSAGE));
  }

  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    PythonVersionUtils.Version required = PythonVersionUtils.Version.V_312;

    // All versions must be greater than or equal to the required version.
    return ctx.sourcePythonVersions().stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }
}
