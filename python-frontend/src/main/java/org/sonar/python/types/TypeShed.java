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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.PythonTreeMaker;

import javax.annotation.Nullable;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

public class TypeShed {

  private static final String TYPING = "typing";
  private static final String TYPING_EXTENSIONS = "typing_extensions";
  private static Map<String, Symbol> builtins;
  private static final Map<String, Set<Symbol>> typeShedSymbols = new HashMap<>();
  private static final Map<String, Set<Symbol>> builtinGlobalSymbols = new HashMap<>();

  private static final String STDLIB_2AND3 = "typeshed/stdlib/2and3/";
  private static final String STDLIB_2 = "typeshed/stdlib/2/";
  private static final String STDLIB_3 = "typeshed/stdlib/3/";
  private static final String THIRD_PARTY_2AND3 = "typeshed/third_party/2and3/";
  private static final String THIRD_PARTY_2 = "typeshed/third_party/2/";
  private static final String THIRD_PARTY_3 = "typeshed/third_party/3/";

  private TypeShed() {
  }

  public static Map<String, Symbol> builtinSymbols() {
    if (TypeShed.builtins == null) {
      Map<String, Symbol> builtins = new HashMap<>();
      builtins.put(NONE_TYPE, new ClassSymbolImpl(NONE_TYPE, NONE_TYPE));
      InputStream resource = TypeShed.class.getResourceAsStream("typeshed/stdlib/2and3/builtins.pyi");
      PythonFile file = new TypeShedPythonFile(resource, "");
      AstNode astNode = PythonParser.create().parse(file.content());
      FileInput fileInput = new PythonTreeMaker().fileInput(astNode);
      Map<String, Set<Symbol>> globalSymbols = new HashMap<>();
      Set<Symbol> typingModuleSymbols = typingModuleSymbols();
      globalSymbols.put(TYPING, typingModuleSymbols);
      Set<Symbol> typingExtensionsSymbols = typingExtensionsSymbols(Collections.singletonMap(TYPING, typingModuleSymbols));
      globalSymbols.put(TYPING_EXTENSIONS, typingExtensionsSymbols);
      new SymbolTableBuilder("", file, globalSymbols).visitFileInput(fileInput);
      for (Symbol globalVariable : fileInput.globalVariables()) {
        ((SymbolImpl) globalVariable).removeUsages();
        builtins.put(globalVariable.fullyQualifiedName(), globalVariable);
      }
      TypeShed.builtins = Collections.unmodifiableMap(builtins);
      InferredTypes.setBuiltinSymbols(builtins);
      fileInput.accept(new ReturnTypeVisitor());
      TypeShed.builtinGlobalSymbols.put("", new HashSet<>(builtins.values()));
    }
    return builtins;
  }

  private static void setDeclaredReturnType(Symbol symbol, FunctionDef functionDef) {
    TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();
    if (returnTypeAnnotation == null) {
      return;
    }
    if (symbol.is(Symbol.Kind.FUNCTION)) {
      FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
      functionSymbol.setDeclaredReturnType(InferredTypes.declaredType(returnTypeAnnotation));
    } else if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
      Optional.ofNullable(((FunctionDefImpl) functionDef).functionSymbol()).ifPresent(functionSymbol -> setDeclaredReturnType(functionSymbol, functionDef));
    }
  }

  // visible for testing
  static Set<Symbol> typingModuleSymbols() {
    Map<String, Symbol> typingPython3 = getModuleSymbols(TYPING, STDLIB_3, Collections.emptyMap());
    Map<String, Symbol> typingPython2 = getModuleSymbols(TYPING, STDLIB_2, Collections.emptyMap());
    return commonSymbols(typingPython2, typingPython3);
  }

  private static Set<Symbol> commonSymbols(Map<String, Symbol> symbolsPython2, Map<String, Symbol> symbolsPython3) {
    Set<Symbol> commonSymbols = new HashSet<>();
    symbolsPython3.forEach((fqn, python3Symbol) -> {
      Symbol python2Symbol = symbolsPython2.get(fqn);
      if (python2Symbol == null) {
        commonSymbols.add(python3Symbol);
      } else {
        Set<Symbol> symbols = new HashSet<>();
        symbols.add(python2Symbol);
        symbols.add(python3Symbol);
        commonSymbols.add(AmbiguousSymbolImpl.create(symbols));
      }
    });

    symbolsPython2.forEach((fqn, python2Symbol) -> {
      if (symbolsPython3.get(fqn) == null) {
        commonSymbols.add(python2Symbol);
      }
    });

    return commonSymbols;
  }

  static Set<Symbol> typingExtensionsSymbols(Map<String, Set<Symbol>> typingSymbols) {
    Map<String, Symbol> typingExtensionSymbols = getModuleSymbols(TYPING_EXTENSIONS, THIRD_PARTY_2AND3,
      typingSymbols);
    return new HashSet<>(typingExtensionSymbols.values());
  }

  public static Set<Symbol> symbolsForModule(String moduleName) {
    if (!TypeShed.typeShedSymbols.containsKey(moduleName)) {
      Set<Symbol> symbols = searchTypeShedForModule(moduleName);
      typeShedSymbols.put(moduleName, symbols);
      return symbols;
    }
    return TypeShed.typeShedSymbols.get(moduleName);
  }

  public static Symbol symbolWithFQN(String stdLibModuleName, String fullyQualifiedName) {
    Set<Symbol> symbols = symbolsForModule(stdLibModuleName);
    return symbols.stream().filter(s -> fullyQualifiedName.equals(s.fullyQualifiedName())).findFirst().orElse(null);
  }

  private static Set<Symbol> searchTypeShedForModule(String moduleName) {
    Set<Symbol> standardLibrarySymbols = new HashSet<>(getModuleSymbols(moduleName, STDLIB_2AND3, builtinGlobalSymbols).values());
    if (standardLibrarySymbols.isEmpty()) {
      standardLibrarySymbols = commonSymbols(getModuleSymbols(moduleName, STDLIB_2, builtinGlobalSymbols),
        getModuleSymbols(moduleName, STDLIB_3, builtinGlobalSymbols));
    }
    if (!standardLibrarySymbols.isEmpty()) {
      return standardLibrarySymbols;
    }
    Set<Symbol> thirdPartySymbols = new HashSet<>(getModuleSymbols(moduleName, THIRD_PARTY_2AND3, builtinGlobalSymbols).values());
    if (!thirdPartySymbols.isEmpty()) {
      return thirdPartySymbols;
    }
    return commonSymbols(getModuleSymbols(moduleName, THIRD_PARTY_2, builtinGlobalSymbols),
      getModuleSymbols(moduleName, THIRD_PARTY_3, builtinGlobalSymbols));
  }

  @Nullable
  private static ModuleDescription getResourceForModule(String moduleName, String categoryPath) {
    String[] splittedName = moduleName.split("\\.");
    String pathToModule = String.join("/", moduleName.split("\\."));
    InputStream resource = TypeShed.class.getResourceAsStream(categoryPath + pathToModule + ".pyi");
    if (resource == null) {
      resource = TypeShed.class.getResourceAsStream(categoryPath + moduleName + "/__init__.pyi");
      if (resource == null) {
        return null;
      }
    }
    String packageName = "";
    String fileName = moduleName;
    if (splittedName.length != 1) {
      packageName = String.join(".", Arrays.copyOf(splittedName, splittedName.length - 1));
      fileName = splittedName[splittedName.length - 1];
    }
    return new ModuleDescription(resource, fileName, packageName);
  }

  private static Map<String, Symbol> getModuleSymbols(String moduleName, String categoryPath, Map<String, Set<Symbol>> initialSymbols) {
    ModuleDescription moduleDescription = getResourceForModule(moduleName, categoryPath);
    if (moduleDescription == null) {
      return Collections.emptyMap();
    }
    PythonFile file = new TypeShedPythonFile(moduleDescription.resource, moduleDescription.fileName);
    AstNode astNode = PythonParser.create().parse(file.content());
    FileInput fileInput = new PythonTreeMaker().fileInput(astNode);
    new SymbolTableBuilder(moduleDescription.packageName, file, initialSymbols).visitFileInput(fileInput);
    fileInput.accept(new ReturnTypeVisitor());
    return fileInput.globalVariables().stream()
      .map(symbol -> {
        ((SymbolImpl) symbol).removeUsages();
        return symbol;
      })
      .filter(s -> s.fullyQualifiedName() != null && s.fullyQualifiedName().startsWith(moduleName))
      .collect(Collectors.toMap(Symbol::fullyQualifiedName, Function.identity()));
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

  static class ReturnTypeVisitor extends BaseTreeVisitor {

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      Optional.ofNullable(functionDef.name().symbol()).ifPresent(symbol -> {
        setDeclaredReturnType(symbol, functionDef);
        setParameterTypes(symbol, functionDef);
      });
      super.visitFunctionDef(functionDef);
    }

    private static void setParameterTypes(Symbol symbol, FunctionDef functionDef) {
      if (symbol.is(Symbol.Kind.FUNCTION)) {
        FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
        ParameterList parameters = functionDef.parameters();
        if (parameters != null) {
          // For builtin functions, we don't have type information from typings.pyi for the parameters when constructing the initial symbol table
          // We need to recreate those with that information
          functionSymbol.setParametersWithType(parameters);
        }
      } else if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
        FunctionSymbol funcDefSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
        if (funcDefSymbol != null) {
          setParameterTypes(funcDefSymbol, functionDef);
        }
      }
    }
  }

  static class ModuleDescription {
    InputStream resource;
    String fileName;
    String packageName;

    ModuleDescription(InputStream resource, String fileName, String packageName) {
      this.resource = resource;
      this.fileName = fileName;
      this.packageName = packageName;
    }
  }

}
