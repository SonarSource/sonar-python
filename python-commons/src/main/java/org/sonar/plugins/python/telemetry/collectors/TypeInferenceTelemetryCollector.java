/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.telemetry.collectors;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;

/**
 * Collects telemetry metrics about type inference quality.
 * Must be called after type inference has run on each file.
 */
public class TypeInferenceTelemetryCollector {

  private final AtomicLong totalNames = new AtomicLong(0);
  private final AtomicLong unknownTypeNames = new AtomicLong(0);
  private final AtomicLong unresolvedImportTypeNames = new AtomicLong(0);
  private final AtomicLong totalImports = new AtomicLong(0);
  private final AtomicLong importsWithUnknownType = new AtomicLong(0);
  private final AtomicLong uniqueSymbols = new AtomicLong(0);
  private final AtomicLong unknownSymbols = new AtomicLong(0);

  public void collect(FileInput rootTree) {
    var visitor = new CollectorVisitor();
    rootTree.accept(visitor);
    aggregateMetrics(visitor);
  }

  private synchronized void aggregateMetrics(CollectorVisitor visitor) {
    totalNames.addAndGet(visitor.totalNames);
    unknownTypeNames.addAndGet(visitor.unknownTypeNames);
    unresolvedImportTypeNames.addAndGet(visitor.unresolvedImportTypeNames);
    totalImports.addAndGet(visitor.totalImports);
    importsWithUnknownType.addAndGet(visitor.importsWithUnknownType);
    uniqueSymbols.addAndGet(visitor.uniqueSymbols.size());
    unknownSymbols.addAndGet(visitor.unknownSymbols.size());
  }

  public TypeInferenceTelemetry getTelemetry() {
    return new TypeInferenceTelemetry(
      totalNames.get(),
      unknownTypeNames.get(),
      unresolvedImportTypeNames.get(),
      totalImports.get(),
        importsWithUnknownType.get(),
      uniqueSymbols.get(),
      unknownSymbols.get()
    );
  }

  private static class CollectorVisitor extends BaseTreeVisitor {
    long totalNames = 0;
    long unknownTypeNames = 0;
    long unresolvedImportTypeNames = 0;
    long totalImports = 0;
    long importsWithUnknownType = 0;
    Set<SymbolV2> uniqueSymbols = new HashSet<>();
    Set<SymbolV2> unknownSymbols = new HashSet<>();

    @Override
    public void visitName(Name name) {
      totalNames++;

      PythonType type = name.typeV2();
      if (type == PythonType.UNKNOWN) {
        unknownTypeNames++;
      } else if (type instanceof UnknownType.UnresolvedImportType) {
        unresolvedImportTypeNames++;
      }

      SymbolV2 symbol = name.symbolV2();
      if (symbol != null) {
        uniqueSymbols.add(symbol);
        if (isUnknownSymbol(symbol)) {
          unknownSymbols.add(symbol);
        }
      }

      super.visitName(name);
    }

    @Override
    public void visitImportName(ImportName importName) {
      for (AliasedName aliasedName : importName.modules()) {
        totalImports++;
        if (hasUnknownImportType(aliasedName)) {
          importsWithUnknownType++;
        }
      }
      super.visitImportName(importName);
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      for (AliasedName aliasedName : importFrom.importedNames()) {
        totalImports++;
        if (hasUnknownImportType(aliasedName)) {
          importsWithUnknownType++;
        }
      }
      super.visitImportFrom(importFrom);
    }

    private static boolean hasUnknownImportType(AliasedName aliasedName) {
      Name nameToCheck = aliasedName.alias();
      if (nameToCheck == null) {
        var names = aliasedName.dottedName().names();
        if (!names.isEmpty()) {
          nameToCheck = names.get(names.size() - 1);
        }
      }
      if (nameToCheck == null) {
        return false;
      }
      return isUnknownType(nameToCheck.typeV2());
    }

    private static boolean isUnknownType(PythonType type) {
      return type == PythonType.UNKNOWN || type instanceof UnknownType.UnresolvedImportType;
    }

    private static boolean isUnknownSymbol(SymbolV2 symbol) {
      // A symbol is considered unknown if all its binding usages have unknown types
      return symbol.usages().stream()
        .filter(UsageV2::isBindingUsage)
        .allMatch(usage -> {
          if (usage.tree() instanceof Name name) {
            return isUnknownType(name.typeV2());
          }
          return true;
        });
    }
  }
}

