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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CollectionBulkAdditionUtils;
import org.sonar.python.checks.utils.CollectionBulkAdditionUtils.BulkAddition;

@Rule(key = "S9409")
public class ListExtendOverIndividualAppendsCheck extends PythonSubscriptionCheck {

  private static final BulkAddition LIST_EXTEND = new BulkAddition(
    "list",
    TypeMatchers.isObjectOfType("builtins.list"),
    TypeMatchers.isType("list.append"),
    "append",
    "extend");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, ctx -> CollectionBulkAdditionUtils.checkForStatement(ctx, LIST_EXTEND));
    context.registerSyntaxNodeConsumer(Tree.Kind.STATEMENT_LIST, ctx -> CollectionBulkAdditionUtils.checkStatementList(ctx, LIST_EXTEND));
  }
}
