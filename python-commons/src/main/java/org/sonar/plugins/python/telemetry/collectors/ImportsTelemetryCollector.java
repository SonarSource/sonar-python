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
package org.sonar.plugins.python.telemetry.collectors;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.dependency.model.Dependency;

/**
 * Collects top-level module names from Python import statements across all analyzed files.
 * Thread-safe: may be called from parallel file scans.
 */
public class ImportsTelemetryCollector {

  private static final int NAME_LENGTH_LIMIT = 100;

  private final Set<String> importedModules = ConcurrentHashMap.newKeySet();

  public void collect(FileInput rootTree) {
    var visitor = new CollectorVisitor();
    rootTree.accept(visitor);
    importedModules.addAll(visitor.getCollected());
  }

  public ImportsTelemetry getTelemetry() {
    return new ImportsTelemetry(importedModules);
  }

  private static class CollectorVisitor extends BaseTreeVisitor {
    private final Set<String> collected = new HashSet<>();

    @Override
    public void visitImportName(ImportName importName) {
      // import X.Y.Z  ->  top-level is X
      for (AliasedName aliasedName : importName.modules()) {
        List<Name> names = aliasedName.dottedName().names();
        if (!names.isEmpty()) {
          addIfShortEnough(names.get(0).name());
        }
      }
      super.visitImportName(importName);
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      // Skip relative imports (from . import foo, from ..utils import bar)
      if (!importFrom.dottedPrefixForModule().isEmpty()) {
        super.visitImportFrom(importFrom);
        return;
      }
      // module() can be null for bare relative imports
      if (importFrom.module() == null) {
        super.visitImportFrom(importFrom);
        return;
      }
      // Collect the top-level module name — works for both regular and wildcard imports
      List<Name> names = importFrom.module().names();
      if (!names.isEmpty()) {
        addIfShortEnough(names.get(0).name());
      }
      super.visitImportFrom(importFrom);
    }

    private void addIfShortEnough(String rawName) {
      // Reuse Dependency normalization (lowercase + [._-]+ -> -)
      String normalized = new Dependency(rawName).name();
      if (normalized.length() < NAME_LENGTH_LIMIT) {
        collected.add(normalized);
      }
    }

    Set<String> getCollected() {
      return collected;
    }
  }
}
