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
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;

@Rule(key = "S6794")
public class TypeAliasAnnotationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a \"type\" statement instead of this \"TypeAlias\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.VARIABLE_TYPE_ANNOTATION, TypeAliasAnnotationCheck::checkTypeAliasVariableAnnotation);
  }

  public static void checkTypeAliasVariableAnnotation(SubscriptionContext ctx) {
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    TypeAnnotation typeAnnotation = (TypeAnnotation) ctx.syntaxNode();
    Optional.of(typeAnnotation.expression())
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("typing.TypeAlias"::equals)
      .ifPresent(fqn -> ctx.addIssue(typeAnnotation.parent(), MESSAGE));
  }

  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    PythonVersionUtils.Version required = PythonVersionUtils.Version.V_312;

    // All versions must be greater than or equal to the required version.
    return ctx.sourcePythonVersions().stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }
}
