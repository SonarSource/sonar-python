/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S2208")
public class WildcardImportCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Import only needed names or import the module and then use its members.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, ctx -> {
      if (ctx.pythonFile().fileName().equals("__init__.py")) {
        // Ignore __init__.py files, as wildcard import are commonly used to populate those.
        return;
      }

      ImportFrom importFrom = (ImportFrom) ctx.syntaxNode();
      if (importFrom.isWildcardImport()) {
        ctx.addIssue(importFrom, MESSAGE);
      }
    });
  }
}
