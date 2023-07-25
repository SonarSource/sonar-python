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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;

import static org.sonar.plugins.python.api.tree.Tree.Kind.IMPORT_FROM;


@Rule(key = "S1128")
public class UnusedImportCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this unused import.";
  private static final Set<String> ALLOWED_MODULES = Set.of("__future__", "typing", "typing_extensions");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(IMPORT_FROM, ctx -> checkImportFrom(((ImportFrom) ctx.syntaxNode()), ctx));
  }

  private static void checkImportFrom(ImportFrom importFrom, SubscriptionContext ctx) {
    // The rule should not raise on __init__ files as they are often used as a facade for packages
    if ("__init__.py".equals(ctx.pythonFile().fileName())) return;
    DottedName module = importFrom.module();
    if (module != null && module.names().size() == 1 && ALLOWED_MODULES.contains(module.names().get(0).name())) return;
    for (AliasedName aliasedName : importFrom.importedNames()) {
      Name alias = aliasedName.alias();
      var importedName = alias != null ? alias : aliasedName.dottedName().names().get(0);
      var importedSymbol = importedName.symbol();
      // defensive programming: imported symbol should never be null, because it always binds a name
      if (importedSymbol != null && importedSymbol.usages().stream().filter(u -> !u.isBindingUsage()).findFirst().isEmpty()) {
        ctx.addIssue(importedName, MESSAGE);
      }
    }
  }
}
