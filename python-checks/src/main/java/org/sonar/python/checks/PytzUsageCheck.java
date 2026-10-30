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

import java.util.Collection;
import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6890")
public class PytzUsageCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Don't use `pytz` module with Python 3.9 and later.";
  private static final TypeMatcher PYTZ_PREFIX_MATCHER = TypeMatchers.withFQNPrefix("pytz.");

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

    if (context.syntaxNode() instanceof ImportFrom importFrom) {
      if (importFrom.dottedPrefixForModule().isEmpty()) {
        Optional.ofNullable(importFrom.module())
          // V1 associated the imported symbol with the alias rather than the imported name.
          // Preserve that behavior: an aliased "from pytz import ..." import is not reported.
          .ifPresent(module -> importFrom.importedNames().stream()
            .filter(name -> name.alias() == null && (isPytzModule(module) || isPytzImportedName(name)))
            .forEach(name -> context.addIssue(module, MESSAGE)));
      }
      return;
    }

    Stream.of(
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
        || PYTZ_PREFIX_MATCHER.isTrueFor(name, context))
      .forEach(name -> context.addIssue(name, MESSAGE));
  }

  private static boolean isPytzModule(DottedName module) {
    return !module.names().isEmpty()
      && "pytz".equals(module.names().get(0).name());
  }

  private static boolean isPytzImportedName(AliasedName name) {
    return !name.dottedName().names().isEmpty()
      && "pytz".equals(name.dottedName().names().get(0).name());
  }
}
