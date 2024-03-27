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
package org.sonar.python.types;

import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.tree.NameImpl;

public class PyTypeAnnotation extends BaseTreeVisitor {

  private final TypeContext typeContext;
  private final String filePath;

  public PyTypeAnnotation(TypeContext typeContext, PythonFile pythonFile) {
    this.typeContext = typeContext;
    if (pythonFile.key().indexOf(':') != -1) {
      this.filePath = pythonFile.key().substring(pythonFile.key().indexOf(':') + 1);
    } else {
      this.filePath = pythonFile.key();
    }
  }

  public void annotate(FileInput fileInput) {
    fileInput.accept(this);
  }

  @Override
  public void visitName(Name name) {
    var inferredType = typeContext.getTypeFor(this.filePath, name)
      .flatMap(type -> Optional.of(type)
        .filter(Predicate.not(RuntimeType.class::isInstance))
        .or(() -> Optional.of(type)
          .map(RuntimeType.class::cast)
          .filter(runtimeType -> Optional.of(runtimeType)
            .map(RuntimeType::runtimeTypeSymbol)
            .map(Symbol::fullyQualifiedName)
            .isPresent()
          )
        )
      )
      .orElseGet(InferredTypes::anyType);
    ((NameImpl) name).setInferredType(inferredType);
    super.visitName(name);
  }

}
