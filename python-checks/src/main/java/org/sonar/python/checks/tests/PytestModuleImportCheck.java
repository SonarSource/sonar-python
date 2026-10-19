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
package org.sonar.python.checks.tests;

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S9084")
public class PytestModuleImportCheck extends PythonSubscriptionCheck {

  private static final String PYTEST = "pytest";
  private static final String MESSAGE_FROM_IMPORT = "Import \"pytest\" as a module.";

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, PytestModuleImportCheck::checkImportFrom);
  }

  private static void checkImportFrom(SubscriptionContext ctx) {
    ImportFrom importFrom = (ImportFrom) ctx.syntaxNode();
    DottedName module = importFrom.module();
    if (module != null && importFrom.dottedPrefixForModule().isEmpty() && isPytestModule(module)) {
      ctx.addIssue(importFrom, MESSAGE_FROM_IMPORT);
    }
  }
  private static boolean isPytestModule(DottedName dottedName) {
    List<Name> names = dottedName.names();
    return !names.isEmpty() && PYTEST.equals(names.get(0).name());
  }
}
