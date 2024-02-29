/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.types;

import java.util.Optional;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.NameImpl;

public class PyTypeAnnotation extends BaseTreeVisitor {

  private final TypeContext typeContext;
  private final PythonFile pythonFile;
  private final String filePath;

  public PyTypeAnnotation(TypeContext typeContext, PythonFile pythonFile) {
    this.typeContext = typeContext;
    this.pythonFile = pythonFile;
    if (pythonFile.key().startsWith("project:")) {
      this.filePath = pythonFile.key().substring("project:".length());
    } else {
      this.filePath = pythonFile.key();
    }
  }

  public void annotate(FileInput fileInput) {
    fileInput.accept(new NameVisitor(typeContext, filePath));
  }

  private static class NameVisitor extends BaseTreeVisitor {

    private final TypeContext typeContext;
    private final String filePath;

    public NameVisitor(TypeContext typeContext, String filePath) {
      this.typeContext = typeContext;
      this.filePath = filePath;
    }

    @Override
    public void visitName(Name name) {
      Optional<InferredType> typeForName = typeContext.getTypeFor(this.filePath, name);
      typeForName.ifPresent(type -> ((NameImpl) name).setInferredType(type));
      super.visitName(name);
    }
  }

}
