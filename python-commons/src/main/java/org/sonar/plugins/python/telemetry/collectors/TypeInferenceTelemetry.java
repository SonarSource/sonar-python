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

/**
 * Telemetry data for type inference quality metrics.
 *
 * @param totalNames Total number of Name nodes in all analyzed trees
 * @param unknownTypeNames Number of Names with the UNKNOWN type
 * @param unresolvedImportTypeNames Number of Names with an UnresolvedImportType
 * @param totalImports Total number of imported items (from both import and from...import statements)
 * @param importsWithUnknownType Number of imports that resolve to UNKNOWN or UnresolvedImportType
 * @param uniqueSymbols Number of unique symbols (using symbolsV2)
 * @param unknownSymbols Number of unique symbols which are unknown
 */
public record TypeInferenceTelemetry(
  long totalNames,
  long unknownTypeNames,
  long unresolvedImportTypeNames,
  long totalImports,
  long importsWithUnknownType,
  long uniqueSymbols,
  long unknownSymbols) {

  public static TypeInferenceTelemetry empty() {
    return new TypeInferenceTelemetry(0, 0, 0, 0, 0, 0, 0);
  }

  public TypeInferenceTelemetry add(TypeInferenceTelemetry other) {
    return new TypeInferenceTelemetry(
      this.totalNames + other.totalNames,
      this.unknownTypeNames + other.unknownTypeNames,
      this.unresolvedImportTypeNames + other.unresolvedImportTypeNames,
      this.totalImports + other.totalImports,
      this.importsWithUnknownType + other.importsWithUnknownType,
      this.uniqueSymbols + other.uniqueSymbols,
      this.unknownSymbols + other.unknownSymbols
    );
  }
}

