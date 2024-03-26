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

import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.tree.NameImpl;

public class TypesTableBuilder extends BaseTreeVisitor {

  private final TypesTable typesTable;
  private final String filePath;

  public TypesTableBuilder(TypesTable typesTable, PythonFile pythonFile) {
    this.typesTable = typesTable;
    this.filePath = getFilePath(pythonFile);
  }

  private String getFilePath(PythonFile pythonFile) {
    if (pythonFile.key().indexOf(':') != -1) {
      return pythonFile.key().substring(pythonFile.key().indexOf(':') + 1);
    } else {
      return pythonFile.key();
    }
  }

  public void annotate(FileInput fileInput) {
    fileInput.accept(this);
  }

  @Override
  public void visitName(Name name) {
    var type = typesTable.getTypeForName(filePath, name);
    if (name instanceof NameImpl ni) {
      ni.pythonType(type);
    }
    super.visitName(name);
  }
}
