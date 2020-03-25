/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import java.io.InputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.PythonTreeMaker;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

public class TypeShed {

  private static Map<String, Symbol> builtins;

  private TypeShed() {
  }

  public static Map<String, Symbol> builtinSymbols() {
    if (TypeShed.builtins == null) {
      Map<String, Symbol> builtins = new HashMap<>();
      builtins.put(NONE_TYPE, new ClassSymbolImpl(NONE_TYPE, NONE_TYPE));
      InputStream resource = TypeShed.class.getResourceAsStream("builtins.pyi");
      PythonFile file = new TypeShedPythonFile(resource);
      AstNode astNode = PythonParser.create().parse(file.content());
      FileInput fileInput = new PythonTreeMaker().fileInput(astNode);
      Map<String, Set<Symbol>> globalSymbols = Collections.emptyMap();
      new SymbolTableBuilder("", file, globalSymbols).visitFileInput(fileInput);
      for (Symbol globalVariable : fileInput.globalVariables()) {
        builtins.put(globalVariable.fullyQualifiedName(), globalVariable);
      }
      BaseTreeVisitor visitor = new BaseTreeVisitor() {
        @Override
        public void visitFunctionDef(FunctionDef functionDef) {
          TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();
          Optional.ofNullable(functionDef.name().symbol()).ifPresent(symbol -> {
            if (symbol.kind() == Symbol.Kind.FUNCTION && returnTypeAnnotation != null) {
              FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
              functionSymbol.setDeclaredReturnType(InferredTypes.declaredType(returnTypeAnnotation, builtins));
            }
          });
          super.visitFunctionDef(functionDef);
        }
      };
      fileInput.accept(visitor);
      TypeShed.builtins = Collections.unmodifiableMap(builtins);
    }
    return builtins;
  }

  public static ClassSymbol typeShedClass(String fullyQualifiedName) {
    Symbol symbol = builtinSymbols().get(fullyQualifiedName);
    if (symbol == null) {
      throw new IllegalArgumentException("No TypeShed symbol found for name: " + fullyQualifiedName);
    }
    if (symbol.kind() != Symbol.Kind.CLASS) {
      throw new IllegalArgumentException("TypeShed symbol " + fullyQualifiedName + " is not a class");
    }
    return (ClassSymbol) symbol;
  }

}
