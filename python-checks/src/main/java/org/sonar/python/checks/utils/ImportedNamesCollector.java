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
package org.sonar.python.checks.utils;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Tree;

public class ImportedNamesCollector extends BaseTreeVisitor {

  private final Set<String> importedNames = new HashSet<>();

  public void collect(Tree tree) {
    importedNames.clear();
    tree.accept(this);
  }

  @Override
  public void visitImportFrom(ImportFrom pyImportFromTree) {
    pyImportFromTree.importedNames().forEach(this::addImportedName);
  }

  @Override
  public void visitImportName(ImportName pyImportNameTree) {
    pyImportNameTree.modules().forEach(this::addImportedName);
  }

  private void addImportedName(AliasedName aliasedName) {
    Optional.of(aliasedName)
      .map(AliasedName::alias)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .ifPresent(importedNames::add);
  }

  public boolean anyMatches(Predicate<String> predicate) {
    return importedNames.stream().anyMatch(predicate);
  }
}
