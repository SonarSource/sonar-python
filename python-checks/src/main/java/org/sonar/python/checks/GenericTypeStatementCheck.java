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

import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6795")
public class GenericTypeStatementCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a generic type parameter instead of a \"TypeVar\" in this type statement.";
  private static final String SECONDARY_MESSAGE_USE = "Use of \"TypeVar\" here.";
  private static final String SECONDARY_MESSAGE_ASSIGNMENT = "\"TypeVar\" is assigned here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TYPE_ALIAS_STMT, GenericTypeStatementCheck::checkUseOfGenerics);
  }

  private static void checkUseOfGenerics(SubscriptionContext ctx) {
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    TypeAliasStatement typeStatement = (TypeAliasStatement) ctx.syntaxNode();
    Set<Tree> typeVarAsTypeParameter = Optional.ofNullable(typeStatement.expression())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(SubscriptionExpression.class))
      .map(SubscriptionExpression::subscripts)
      .map(ExpressionList::expressions)
      .stream()
      .flatMap(Collection::stream)
      .filter(Expressions::isGenericTypeAnnotation)
      .collect(Collectors.toSet());

    if (!typeVarAsTypeParameter.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(typeStatement.name(), MESSAGE);
      typeVarAsTypeParameter.forEach(loc -> issue.secondary(loc, SECONDARY_MESSAGE_USE));
      getAssignmentLocations(typeVarAsTypeParameter).forEach(loc -> issue.secondary(loc, SECONDARY_MESSAGE_ASSIGNMENT));
    }
  }

  private static Set<Tree> getAssignmentLocations(Set<Tree> secondaryLocations) {
    return secondaryLocations.stream()
      .map(Name.class::cast)
      .map(Expressions::singleAssignedValue)
      .collect(Collectors.toSet());
  }

  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    return PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_312);
  }
}
