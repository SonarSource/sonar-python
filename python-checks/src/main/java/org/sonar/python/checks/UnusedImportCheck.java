/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

@Rule(key = "S1128")
public class UnusedImportCheck extends PythonVisitorCheck {

  private static final String MESSAGE = "Remove this unused import.";
  private static final Set<String> ALLOWED_MODULES = Set.of("__future__", "typing", "typing_extensions");
  private static final Set<String> ALLOWED_FQN_PREFIX = Set.of("sklearn.experimental.");

  private final Map<String, Name> unusedImports = new HashMap<>();

  @Override
  public void visitFileInput(FileInput fileInput) {
    unusedImports.clear();
    // The rule should not raise on __init__ files as they are often used as a facade for packages
    if ("__init__.py".equals(getContext().pythonFile().fileName())) return;
    super.visitFileInput(fileInput);
    removeImportedNamesUsedInCommentsOrLiterals(fileInput);
    unusedImports.values().forEach(unusedImport -> addIssue(unusedImport, MESSAGE));
  }


  /**
   * To reduce FPs, we go through every comment and every string literal to check if any detected unused import has been used there.
   * This is useful to avoid raising FPs when symbols are exported via `__all__` global variable or when they are used inside type hints comments.
   */
  private void removeImportedNamesUsedInCommentsOrLiterals(Tree tree) {
    if (tree.is(Tree.Kind.TOKEN)) {
      for (Trivia trivia : ((Token) tree).trivia()) {
        unusedImports.values().removeIf(name -> trivia.value().contains(name.name()));
      }
    } else if (tree.is(Tree.Kind.STRING_LITERAL)) {
      unusedImports.remove(((StringLiteral) tree).trimmedQuotesValue());
    } else {
      for (Tree child : tree.children()) {
        removeImportedNamesUsedInCommentsOrLiterals(child);
      }
    }
  }

  @Override
  public void visitImportFrom(ImportFrom importFrom) {
    DottedName module = importFrom.module();
    if (module != null && module.names().size() == 1 && ALLOWED_MODULES.contains(module.names().get(0).name())) return;
    for (AliasedName aliasedName : importFrom.importedNames()) {
      Name alias = aliasedName.alias();
      var importedName = alias != null ? alias : aliasedName.dottedName().names().get(0);
      Optional.ofNullable(importedName.symbol())
        .filter(symbol -> symbol.usages().stream().filter(u -> !u.isBindingUsage()).findFirst().isEmpty())
        .filter(symbol ->
          Optional.ofNullable(symbol.fullyQualifiedName())
            .map(fqn -> ALLOWED_FQN_PREFIX.stream().noneMatch(fqn::startsWith))
            .orElse(true)
        )
        .ifPresent(symbol -> unusedImports.put(importedName.name(), importedName));
    }
    super.visitImportFrom(importFrom);
  }
}
