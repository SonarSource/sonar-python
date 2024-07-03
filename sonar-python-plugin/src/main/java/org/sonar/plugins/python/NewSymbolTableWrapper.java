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
package org.sonar.plugins.python;

import java.util.Map;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.symbol.NewSymbol;
import org.sonar.api.batch.sensor.symbol.NewSymbolTable;

public class NewSymbolTableWrapper {

  private final NewSymbolTable newSymbolTable;
  private final InputFile inputFile;

  public NewSymbolTableWrapper(SensorContext context, InputFile inputFile) {
    this.inputFile = inputFile;
    if (inputFile instanceof GeneratedIPythonFile generatedIPythonFile) {
      this.newSymbolTable = context.newSymbolTable().onFile(generatedIPythonFile.originalFile);
    } else {
      this.newSymbolTable = context.newSymbolTable().onFile(inputFile);
    }
  }

  public void save() {
    newSymbolTable.save();
  }

  public NewSymbolWrapper newSymbol(int var1, int var2, int var3, int var4) {
    if (inputFile instanceof GeneratedIPythonFile generatedIPythonFile) {
      Map<Integer, GeneratedIPythonFile.Offset> offsetMap = generatedIPythonFile.offsetMap;
      var fromOffset = offsetMap.get(var1);
      var toOffset = offsetMap.get(var3);
      return new NewSymbolWrapper(newSymbolTable.newSymbol(fromOffset.line(), var2 + fromOffset.column(),
        toOffset.line(), var4 + toOffset.column()), inputFile);
    }
    return new NewSymbolWrapper(newSymbolTable.newSymbol(var1, var2, var3, var4), inputFile);
  }

  static class NewSymbolWrapper {

      private final NewSymbol newSymbol;
      private final InputFile inputFile;

      public NewSymbolWrapper(NewSymbol newSymbol, InputFile inputFile) {
        this.newSymbol = newSymbol;
        this.inputFile = inputFile;
      }

      public void newReference(int var1, int var2, int var3, int var4) {
        if (this.inputFile instanceof GeneratedIPythonFile generatedIPythonFile) {
          Map<Integer, GeneratedIPythonFile.Offset> offsetMap = generatedIPythonFile.offsetMap;
          var fromOffset = offsetMap.get(var1);
          var toOffset = offsetMap.get(var3);
          newSymbol.newReference(fromOffset.line(), var2 + fromOffset.column(),
            toOffset.line(), var4 + toOffset.column());
        } else {
          newSymbol.newReference(var1, var2, var3, var4);
        }
      }
  }
}
