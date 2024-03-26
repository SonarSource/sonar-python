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
import java.util.List;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.types.pytype.PyTypeTable;
import org.sonar.python.types.v2.converter.PyTypeToPythonTypeConverter;

public class TypesTable {

  private final HashMap<String, PythonType> declaredTypesTable;
  private final PyTypeTable pyTypeTable;

  public TypesTable(PyTypeTable pyTypeTable) {
    this.pyTypeTable = pyTypeTable;
    declaredTypesTable = new HashMap<>();
  }

  public PythonType getTypeForName(String fileName, Name name) {
    return pyTypeTable.getTypeFor(fileName, name)
      .map(pyTypeInfo -> {
        var baseType = pyTypeInfo.baseType();
        var pythonType = PyTypeToPythonTypeConverter.convert(baseType);
        var typeKey = pythonType.toString();
        if (declaredTypesTable.containsKey(typeKey)) {
          pythonType = declaredTypesTable.get(typeKey);
        } else {
          declaredTypesTable.put(typeKey, pythonType);
        }
        return (PythonType) new ObjectType(pythonType, List.of());
      }).orElse(PythonType.UNKNOWN);
  }

}
