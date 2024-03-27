/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.types.v2;

import java.util.HashMap;

public class TypesTable {

  private final HashMap<String, PythonType> declaredTypesTable;
  private final HashMap<String, PythonType> declaredClassesTable;

  public TypesTable() {
    declaredTypesTable = new HashMap<>();
    declaredClassesTable = new HashMap<>();
  }

  public HashMap<String, PythonType> declaredTypesTable() {
    return declaredTypesTable;
  }

  public PythonType addType(PythonType pythonType) {
    if (pythonType instanceof ClassType classType) {
      return addClassType(classType);
    } else {
      return addAnotherType(pythonType);
    }
  }

  public PythonType addClassType(ClassType classType) {
    var key = classType.getName();
    if (declaredClassesTable.containsKey(key)) {
      return declaredClassesTable.get(key);
    } else {
      declaredClassesTable.put(key, classType);
      return classType;
    }
  }

  public PythonType addAnotherType(PythonType pythonType) {
    var key = pythonType.getName();
    if (declaredTypesTable.containsKey(key)) {
      return declaredTypesTable.get(key);
    } else {
      declaredTypesTable.put(key, pythonType);
      return pythonType;
    }
  }

}
