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
package org.sonar.python.semantic.v2;

import java.util.List;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnknownType;

public class BasicTypeTable implements TypeTable {

  private final ProjectLevelTypeTable projectLevelTypeTable;

  public BasicTypeTable(ProjectLevelTypeTable fakeTypeTable) {
    // used to trigger typeshed resolutions
    this.projectLevelTypeTable = fakeTypeTable;
  }

  @Override
  public PythonType getBuiltinsModule() {
    this.projectLevelTypeTable.getBuiltinsModule();
    return new UnknownType.UnresolvedImportType("");
  }

  @Override
  public PythonType getType(String typeFqn) {
    this.projectLevelTypeTable.getType(typeFqn);
    return new UnknownType.UnresolvedImportType(typeFqn);
  }

  @Override
  public PythonType getType(String... typeFqnParts) {
    this.projectLevelTypeTable.getType(typeFqnParts);
    return new UnknownType.UnresolvedImportType(String.join(".", typeFqnParts));
  }

  @Override
  public PythonType getType(List<String> typeFqnParts) {
    this.projectLevelTypeTable.getType(typeFqnParts);
    return new UnknownType.UnresolvedImportType(String.join(".", typeFqnParts));
  }

  @Override
  public PythonType getModuleType(List<String> typeFqnParts) {
    this.projectLevelTypeTable.getModuleType(typeFqnParts);
    return new UnknownType.UnresolvedImportType(String.join(".", typeFqnParts));
  }
}
