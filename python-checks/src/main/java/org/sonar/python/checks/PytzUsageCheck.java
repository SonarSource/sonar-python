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
package org.sonar.python.checks;

import java.util.Collection;
import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6890")
public class PytzUsageCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Don't use `pytz` module with Python 3.9 and later.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, PytzUsageCheck::checkImport);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, PytzUsageCheck::checkImport);
  }

  private static boolean isRelevantPythonVersion(SubscriptionContext context) {
    return PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(context.sourcePythonVersions(), PythonVersionUtils.Version.V_39);
  }

  private static void checkImport(SubscriptionContext context) {
    if (!isRelevantPythonVersion(context)) {
      return;
    }

    Stream.of(
      Optional.of(context.syntaxNode())
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(ImportFrom.class))
        .map(ImportFrom::importedNames),
      Optional.of(context.syntaxNode())
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(ImportName.class))
        .map(ImportName::modules))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .flatMap(Collection::stream)
      .map(AliasedName::dottedName)
      .map(DottedName::names)
      .filter(list -> !list.isEmpty())
      .map(names -> names.get(0))
      .filter(name -> "pytz".equals(name.name())
        || Optional.ofNullable(name.symbol())
          .map(Symbol::fullyQualifiedName)
          .filter(fqn -> fqn.startsWith("pytz")).isPresent())
      .forEach(name -> raiseIssue(context, name));
  }

  private static void raiseIssue(SubscriptionContext context, Tree tree) {
    if (context.syntaxNode().is(Tree.Kind.IMPORT_FROM)) {
      ImportFrom importFrom = (ImportFrom) context.syntaxNode();
      Optional.ofNullable(importFrom.module()).ifPresent(module -> context.addIssue(module, MESSAGE));
    } else {
      context.addIssue(tree, MESSAGE);
    }
  }
}
