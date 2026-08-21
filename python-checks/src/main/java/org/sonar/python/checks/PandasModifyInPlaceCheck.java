/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.matchers.InternalTypeMatchers;

@Rule(key = "S6734")
public class PandasModifyInPlaceCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not use \"inplace=True\" when modifying a dataframe.";

  private static final Set<String> METHOD_NAMES = Set.of(
    "drop",
    "dropna",
    "drop_duplicates",
    "sort_values",
    "sort_index",
    "eval",
    "query");

  private static final TypeMatcher IS_DATAFRAME = TypeMatchers.isObjectOfType("pandas.core.frame.DataFrame");
  private static final TypeMatcher IS_DATAFRAME_METHOD = TypeMatchers.isFunctionOwnerSatisfying(
    TypeMatchers.isType("pandas.core.frame.DataFrame"));
  private static final TypeMatcher IS_OR_CONTAINS_DATAFRAME = TypeMatchers.any(
    IS_DATAFRAME,
    InternalTypeMatchers.isAnyTypeInUnionSatisfying(IS_DATAFRAME));
  private static final TypeMatcher IS_OR_CONTAINS_DATAFRAME_METHOD = TypeMatchers.any(
    IS_DATAFRAME_METHOD,
    InternalTypeMatchers.isAnyTypeInUnionSatisfying(IS_DATAFRAME_METHOD));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasModifyInPlaceCheck::checkInplaceParameter);
  }

  private static void checkInplaceParameter(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!(callExpression.callee() instanceof QualifiedExpression qualifiedExpression)) {
      return;
    }
    if (!METHOD_NAMES.contains(qualifiedExpression.name().name())) {
      return;
    }
    if (!IS_OR_CONTAINS_DATAFRAME_METHOD.isTrueFor(callExpression.callee(), ctx)
      && !IS_OR_CONTAINS_DATAFRAME.isTrueFor(qualifiedExpression.qualifier(), ctx)) {
      return;
    }
    RegularArgument inplaceArgument = TreeUtils.argumentByKeyword("inplace", callExpression.arguments());
    Optional.ofNullable(inplaceArgument)
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::name)
      .filter("True"::equals)
      .ifPresent(unused -> ctx.addIssue(inplaceArgument, MESSAGE));
  }
}
