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
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.tree.PythonTreeMaker;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.OTHER;

public class TypeShedVisitor extends BaseTreeVisitor {

  private Map<String, Symbol> symbolsByName = new HashMap<>();
  private static final Logger LOG = Loggers.get(TypeShedVisitor.class);
  private static Map<String, Symbol> typeShedSymbols;

  private TypeShedVisitor() {
  }

  public static Map<String, Symbol> typeShedSymbols() {
    if (TypeShedVisitor.typeShedSymbols == null) {
      Map<String, Symbol> typeShedSymbols = Collections.emptyMap();
      InputStream resource = TypeShedVisitor.class.getResourceAsStream("builtins.pyi");
      try (BufferedReader reader = new BufferedReader(new InputStreamReader(resource, StandardCharsets.UTF_8))) {
        String builtins = reader.lines().collect(Collectors.joining("\n"));
        AstNode astNode = PythonParser.create().parse(builtins);
        FileInput fileInput = new PythonTreeMaker().fileInput(astNode);
        TypeShedVisitor typeShedVisitor = new TypeShedVisitor();
        fileInput.accept(typeShedVisitor);
        typeShedSymbols = typeShedVisitor.symbolsByName;
      } catch (IOException | UncheckedIOException e) {
        LOG.debug("Unable to read builtin types.");
      }
      TypeShedVisitor.typeShedSymbols = Collections.unmodifiableMap(typeShedSymbols);
    }
    return typeShedSymbols;
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    String functionName = functionDef.name().name();
    SymbolImpl symbol = (SymbolImpl) symbolsByName.get(functionName);
    if (symbol != null) {
      symbol.setKind(OTHER);
    } else {
      FunctionSymbolImpl functionSymbol = new FunctionSymbolImpl(functionName, functionName, false, false, false, Collections.emptyList(), null);
      TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();
      if (returnTypeAnnotation != null) {
        functionSymbol.setInferredReturnType(InferredTypes.declaredType(returnTypeAnnotation));
      }
      symbolsByName.put(functionName, functionSymbol);
    }
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    // don't go inside classes
  }
}
