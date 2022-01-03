/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.BuiltinSymbols;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos.ModuleSymbol;
import org.sonar.python.types.protobuf.SymbolsProtos.OverloadedFunctionSymbol;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BOOL;
import static org.sonar.plugins.python.api.types.BuiltinTypes.COMPLEX;
import static org.sonar.plugins.python.api.types.BuiltinTypes.DICT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.FLOAT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.INT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.LIST;
import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;
import static org.sonar.plugins.python.api.types.BuiltinTypes.STR;
import static org.sonar.plugins.python.api.types.BuiltinTypes.TUPLE;

public class TypeShed {

  private static Map<String, Symbol> builtins;
  private static final Map<String, Map<String, Symbol>> typeShedSymbols = new HashMap<>();
  private static final Map<String, Set<Symbol>> builtinGlobalSymbols = new HashMap<>();
  private static final Set<String> modulesInProgress = new HashSet<>();

  private static final String STDLIB_2AND3 = "typeshed/stdlib/2and3/";
  private static final String STDLIB_2 = "typeshed/stdlib/2/";
  private static final String STDLIB_3 = "typeshed/stdlib/3/";
  private static final String THIRD_PARTY_2AND3 = "typeshed/third_party/2and3/";
  private static final String THIRD_PARTY_2 = "typeshed/third_party/2/";
  private static final String THIRD_PARTY_3 = "typeshed/third_party/3/";
  private static final String CUSTOM_THIRD_PARTY = "custom/";
  private static final String PROTOBUF = "protobuf/";
  private static final String BUILTINS_FQN = "builtins";
  private static final String BUILTINS_PREFIX = BUILTINS_FQN + ".";
  // Those fundamentals builtins symbols need not to be ambiguous for the frontend to work properly
  private static final Set<String> BUILTINS_TO_DISAMBIGUATE = new HashSet<>(
    Arrays.asList(INT, FLOAT, COMPLEX, STR, BuiltinTypes.SET, DICT, LIST, TUPLE, NONE_TYPE, BOOL, "type", "super", "frozenset", "memoryview"));

  static {
    BUILTINS_TO_DISAMBIGUATE.addAll(BuiltinSymbols.EXCEPTIONS);
  }

  private static final Logger LOG = Loggers.get(TypeShed.class);
  private static Set<String> supportedPythonVersions;

  private TypeShed() {
  }

  public static Map<String, Symbol> builtinSymbols() {
    if ((TypeShed.builtins == null)) {
      supportedPythonVersions = ProjectPythonVersion.currentVersions().stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
      Map<String, Symbol> builtins = getSymbolsFromProtobufModule(BUILTINS_FQN);
      builtins.put(NONE_TYPE, new ClassSymbolImpl(NONE_TYPE, NONE_TYPE));
      TypeShed.builtins = Collections.unmodifiableMap(builtins);
      TypeShed.builtinGlobalSymbols.put("", new HashSet<>(builtins.values()));
    }
    return builtins;
  }

  // used by tests whenever 'sonar.python.version' changes
  static void resetBuiltinSymbols() {
    builtins = null;
    typeShedSymbols.clear();
    builtinSymbols();
  }

  private static void setDeclaredReturnType(Symbol symbol, FunctionDef functionDef) {
    TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();
    if (returnTypeAnnotation == null) {
      return;
    }
    if (symbol.is(Symbol.Kind.FUNCTION)) {
      FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
      functionSymbol.setDeclaredReturnType(InferredTypes.fromTypeshedTypeAnnotation(returnTypeAnnotation));
    } else if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
      Optional.ofNullable(((FunctionDefImpl) functionDef).functionSymbol()).ifPresent(functionSymbol -> setDeclaredReturnType(functionSymbol, functionDef));
    }
  }

  private static Set<Symbol> commonSymbols(Map<String, Symbol> symbolsPython2, Map<String, Symbol> symbolsPython3, String packageName) {
    Set<Symbol> commonSymbols = new HashSet<>();
    symbolsPython3.forEach((localName, python3Symbol) -> {
      Symbol python2Symbol = symbolsPython2.get(localName);
      if (python2Symbol == null || python2Symbol == python3Symbol) {
        commonSymbols.add(python3Symbol);
      } else {
        Set<Symbol> symbols = new HashSet<>();
        symbols.add(python2Symbol);
        symbols.add(python3Symbol);
        commonSymbols.add(new AmbiguousSymbolImpl(localName, packageName + "." + localName, symbols));
      }
    });

    symbolsPython2.forEach((localName, python2Symbol) -> {
      if (symbolsPython3.get(localName) == null) {
        commonSymbols.add(python2Symbol);
      }
    });

    return commonSymbols;
  }

  public static Set<Symbol> symbolsForModule(String moduleName) {
    if (!TypeShed.typeShedSymbols.containsKey(moduleName)) {
      Set<Symbol> symbols = searchTypeShedForModule(moduleName);
      Map<String, Symbol> symbolsByFqn = symbols.stream().collect(Collectors.toMap(Symbol::fullyQualifiedName, s -> s));
      typeShedSymbols.put(moduleName, symbolsByFqn);
      return symbols;
    }
    return new HashSet<>(TypeShed.typeShedSymbols.get(moduleName).values());
  }

  @CheckForNull
  public static Symbol symbolWithFQN(String stdLibModuleName, String fullyQualifiedName) {
    Set<Symbol> symbols = symbolsForModule(stdLibModuleName);
    Symbol symbolByFqn = symbols.stream().filter(s -> fullyQualifiedName.equals(s.fullyQualifiedName())).findFirst().orElse(null);
    if (symbolByFqn != null || !fullyQualifiedName.contains(".")) {
      return symbolByFqn;
    }

    // If FQN of the member does not match the pattern of "package_name.file_name.symbol_name"
    // (e.g. it could be declared in package_name.file_name using import) or in case when
    // we have import with an alias (from module import method as alias_method), we retrieve symbol_name out of
    // FQN and try to look up by local symbol name, rather than FQN
    String[] fqnSplittedByDot = fullyQualifiedName.split("\\.");
    String symbolLocalNameFromFqn = fqnSplittedByDot[fqnSplittedByDot.length - 1];

    // TODO: improve performance - see SONARPY-955
    Set<Symbol> matchByName = symbols.stream().filter(s -> symbolLocalNameFromFqn.equals(s.name())).collect(Collectors.toSet());
    if (matchByName.size() == 1) {
      return matchByName.iterator().next();
    }

    return null;
  }

  private static Set<Symbol> searchTypeShedForModule(String moduleName) {
    if (modulesInProgress.contains(moduleName)) {
      return new HashSet<>();
    }
    modulesInProgress.add(moduleName);
    Collection<Symbol> symbolsFromProtobuf = getSymbolsFromProtobufModule(moduleName).values();
    if (!symbolsFromProtobuf.isEmpty()) {
      modulesInProgress.remove(moduleName);
      return new HashSet<>(symbolsFromProtobuf);
    }
    Set<Symbol> customSymbols = new HashSet<>(getModuleSymbols(moduleName, CUSTOM_THIRD_PARTY, builtinGlobalSymbols).values());
    if (!customSymbols.isEmpty()) {
      modulesInProgress.remove(moduleName);
      return customSymbols;
    }
    Set<Symbol> standardLibrarySymbols = new HashSet<>(getModuleSymbols(moduleName, STDLIB_2AND3, builtinGlobalSymbols).values());
    if (standardLibrarySymbols.isEmpty()) {
      standardLibrarySymbols = commonSymbols(getModuleSymbols(moduleName, STDLIB_2, builtinGlobalSymbols),
        getModuleSymbols(moduleName, STDLIB_3, builtinGlobalSymbols), moduleName);
    }
    if (!standardLibrarySymbols.isEmpty()) {
      modulesInProgress.remove(moduleName);
      return standardLibrarySymbols;
    }
    Set<Symbol> thirdPartySymbols = new HashSet<>(getModuleSymbols(moduleName, THIRD_PARTY_2AND3, builtinGlobalSymbols).values());
    if (thirdPartySymbols.isEmpty()) {
      thirdPartySymbols = commonSymbols(getModuleSymbols(moduleName, THIRD_PARTY_2, builtinGlobalSymbols),
        getModuleSymbols(moduleName, THIRD_PARTY_3, builtinGlobalSymbols), moduleName);
    }
    modulesInProgress.remove(moduleName);
    return thirdPartySymbols;
  }

  @Nullable
  private static ModuleDescription getResourceForModule(String moduleName, String categoryPath) {
    String[] moduleNameHierarchy = moduleName.split("\\.");
    String pathToModule = String.join("/", moduleNameHierarchy);
    String moduleFileName = moduleNameHierarchy[moduleNameHierarchy.length - 1];
    String packageName = String.join(".", Arrays.copyOfRange(moduleNameHierarchy, 0, moduleNameHierarchy.length - 1));
    InputStream resource = TypeShed.class.getResourceAsStream(categoryPath + pathToModule + ".pyi");
    if (resource == null) {
      resource = TypeShed.class.getResourceAsStream(categoryPath + pathToModule + "/__init__.pyi");
      if (resource == null) {
        return null;
      }
      moduleFileName = "__init__";
      packageName = moduleName;
    }
    return new ModuleDescription(resource, moduleFileName, packageName);
  }

  private static Map<String, Symbol> getModuleSymbols(String moduleName, String categoryPath, Map<String, Set<Symbol>> initialSymbols) {
    ModuleDescription moduleDescription = getResourceForModule(moduleName, categoryPath);
    if (moduleDescription == null) {
      return Collections.emptyMap();
    }
    PythonFile file = new TypeShedPythonFile(moduleDescription.resource, moduleDescription.fileName);
    AstNode astNode = PythonParser.create().parse(file.content());
    FileInput fileInput = new PythonTreeMaker().fileInput(astNode);
    new SymbolTableBuilder(moduleDescription.packageName, file, ProjectLevelSymbolTable.from(initialSymbols)).visitFileInput(fileInput);
    fileInput.accept(new ReturnTypeVisitor());
    return fileInput.globalVariables().stream()
      .map(symbol -> {
        ((SymbolImpl) symbol).removeUsages();
        return symbol;
      })
      .filter(s -> s.fullyQualifiedName() != null)
      .collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
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

  /**
   * Some special symbols need NOT to be ambiguous for the frontend to work properly.
   * This method sort ambiguous symbol by python version and returns the one which is valid for
   * the most recent Python version.
   */
  private static Symbol disambiguateWithLatestPythonSymbol(Set<Symbol> alternatives) {
    int max = Integer.MIN_VALUE;
    Symbol latestPythonSymbol = null;
    for (Symbol alternative : alternatives) {
      int maxPythonVersionForSymbol = ((SymbolImpl) alternative).validForPythonVersions().stream().mapToInt(Integer::parseInt).max().orElse(max);
      if (maxPythonVersionForSymbol > max) {
        max = maxPythonVersionForSymbol;
        latestPythonSymbol = alternative;
      }
    }
    return latestPythonSymbol;
  }

  public static Collection<Symbol> stubFilesSymbols() {
    Set<Symbol> symbols = new HashSet<>(TypeShed.builtinSymbols().values());
    typeShedSymbols.values().forEach(symbolsByFqn -> symbols.addAll(symbolsByFqn.values()));
    return symbols;
  }

  static class ReturnTypeVisitor extends BaseTreeVisitor {

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      Optional.ofNullable(functionDef.name().symbol()).ifPresent(symbol -> {
        setDeclaredReturnType(symbol, functionDef);
        setParameterTypes(symbol, functionDef);
        setAnnotatedReturnType(symbol, functionDef);
      });
      super.visitFunctionDef(functionDef);
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
      if (annotatedAssignment.variable().is(Tree.Kind.NAME)) {
        Name variable = (Name) annotatedAssignment.variable();
        Optional.ofNullable(variable.symbol()).ifPresent(symbol -> setAnnotatedType(symbol, annotatedAssignment));
      }
      super.visitAnnotatedAssignment(annotatedAssignment);
    }

    private static void setAnnotatedType(Symbol symbol, AnnotatedAssignment annotatedAssignment) {
      TypeAnnotation typeAnnotation = annotatedAssignment.annotation();
      if (symbol.is(Symbol.Kind.OTHER)) {
        SymbolImpl other = (SymbolImpl) symbol;
        other.setAnnotatedTypeName(typeAnnotation);
      }
    }

    private static void setAnnotatedReturnType(Symbol symbol, FunctionDef functionDef) {
      TypeAnnotation typeAnnotation = functionDef.returnTypeAnnotation();
      if (symbol.is(Symbol.Kind.FUNCTION)) {
        ((FunctionSymbolImpl) symbol).setAnnotatedReturnTypeName(typeAnnotation);
      } else if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
        Optional.ofNullable(((FunctionDefImpl) functionDef).functionSymbol()).ifPresent(functionSymbol -> setAnnotatedReturnType(functionSymbol, functionDef));
      }
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

  private static class ModuleDescription {
    InputStream resource;
    String fileName;
    String packageName;

    ModuleDescription(InputStream resource, String fileName, String packageName) {
      this.resource = resource;
      this.fileName = fileName;
      this.packageName = packageName;
    }
  }

  private static Map<String, Symbol> getSymbolsFromProtobufModule(String moduleName) {
    InputStream resource = TypeShed.class.getResourceAsStream(PROTOBUF + moduleName + ".protobuf");
    if (resource == null) {
      return Collections.emptyMap();
    }
    return getSymbolsFromProtobufModule(deserializedModule(moduleName, resource));
  }

  @CheckForNull
  static ModuleSymbol deserializedModule(String moduleName, InputStream resource) {
    try {
      return ModuleSymbol.parseFrom(resource);
    } catch (IOException e) {
      LOG.debug("Error while deserializing protobuf for module " + moduleName, e);
      return null;
    }
  }

  static Map<String, Symbol> getSymbolsFromProtobufModule(@Nullable ModuleSymbol moduleSymbol) {
    if (moduleSymbol == null) {
      return Collections.emptyMap();
    }

    // TODO: Use a common proxy interface Descriptor instead of using Object
    Map<String, Set<Object>> descriptorsByName = new HashMap<>();
    moduleSymbol.getClassesList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));
    moduleSymbol.getFunctionsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));
    moduleSymbol.getOverloadedFunctionsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));
    moduleSymbol.getVarsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));

    Map<String, Symbol> deserializedSymbols = new HashMap<>();

    for (Map.Entry<String, Set<Object>> entry : descriptorsByName.entrySet()) {
      String name = entry.getKey();
      Set<Symbol> symbols = symbolsFromDescriptor(entry.getValue(), false);
      Symbol disambiguatedSymbol = disambiguateSymbolsWithSameName(name, symbols, moduleSymbol.getFullyQualifiedName());
      deserializedSymbols.put(name, disambiguatedSymbol);
    }
    return deserializedSymbols;
  }

  private static Symbol disambiguateSymbolsWithSameName(String name, Set<Symbol> symbols, String moduleFqn) {
    if (symbols.size() > 1) {
      if (haveAllTheSameFqn(symbols) && !isBuiltinToDisambiguate(moduleFqn, name)) {
        return AmbiguousSymbolImpl.create(symbols);
      }
      if (!moduleFqn.equals(BUILTINS_FQN)) {
        String fqns = symbols.stream()
          .map(Symbol::fullyQualifiedName)
          .map(fqn -> fqn == null ? "N/A" : fqn)
          .collect(Collectors.joining(","));
        LOG.debug("Symbol " + name + " has conflicting fully qualified names:" + fqns);
        LOG.debug("It has been disambiguated with its latest Python version available symbol.");
      }
      return disambiguateWithLatestPythonSymbol(symbols);
    }
    return symbols.iterator().next();
  }

  private static boolean isBuiltinToDisambiguate(String moduleFqn, String name) {
    return moduleFqn.equals(BUILTINS_FQN) && BUILTINS_TO_DISAMBIGUATE.contains(name);
  }

  private static boolean haveAllTheSameFqn(Set<Symbol> symbols) {
    String firstFqn = symbols.iterator().next().fullyQualifiedName();
    return firstFqn != null && symbols.stream().map(Symbol::fullyQualifiedName).allMatch(firstFqn::equals);
  }

  public static boolean isValidForProjectPythonVersion(List<String> validForPythonVersions) {
    if (validForPythonVersions.isEmpty()) {
      return true;
    }
    HashSet<String> intersection = new HashSet<>(validForPythonVersions);
    intersection.retainAll(supportedPythonVersions);
    return !intersection.isEmpty();
  }

  public static Set<Symbol> symbolsFromDescriptor(Set<Object> descriptors, boolean isInsideClass) {
    Set<Symbol> symbols = new HashSet<>();
    for (Object descriptor : descriptors) {
      if (descriptor instanceof SymbolsProtos.ClassSymbol) {
        symbols.add(new ClassSymbolImpl(((SymbolsProtos.ClassSymbol) descriptor)));
      }
      if (descriptor instanceof SymbolsProtos.FunctionSymbol) {
        symbols.add(new FunctionSymbolImpl(((SymbolsProtos.FunctionSymbol) descriptor), isInsideClass));
      }
      if (descriptor instanceof OverloadedFunctionSymbol) {
        if (((OverloadedFunctionSymbol) descriptor).getDefinitionsList().size() < 2) {
          throw new IllegalStateException("Overloaded function symbols should have at least two definitions.");
        }
        symbols.add(fromOverloadedFunction(((OverloadedFunctionSymbol) descriptor), isInsideClass));
      }
      if (descriptor instanceof SymbolsProtos.VarSymbol) {
        symbols.add(new SymbolImpl((SymbolsProtos.VarSymbol) descriptor));
      }
    }
    return symbols;
  }

  private static AmbiguousSymbol fromOverloadedFunction(OverloadedFunctionSymbol overloadedFunctionSymbol, boolean isInsideClass) {
    Set<Symbol> overloadedSymbols = overloadedFunctionSymbol.getDefinitionsList().stream()
      .map(def -> new FunctionSymbolImpl(def, isInsideClass, overloadedFunctionSymbol.getValidForList()))
      .collect(Collectors.toSet());
    return AmbiguousSymbolImpl.create(overloadedSymbols);
  }

  @CheckForNull
  public static Symbol symbolWithFQN(String fullyQualifiedName) {
    Map<String, Symbol> builtinSymbols = builtinSymbols();
    Symbol builtinSymbol = builtinSymbols.get(normalizedFqn(fullyQualifiedName));
    if (builtinSymbol != null) {
      return builtinSymbol;
    }
    String[] fqnSplittedByDot = fullyQualifiedName.split("\\.");
    String moduleName = Arrays.stream(fqnSplittedByDot, 0, fqnSplittedByDot.length - 1).collect(Collectors.joining("."));
    return symbolWithFQN(moduleName, fullyQualifiedName);
  }

  public static String normalizedFqn(String fqn) {
    if (fqn.startsWith(BUILTINS_PREFIX)) {
      return fqn.substring(BUILTINS_PREFIX.length());
    }
    return fqn;
  }

}
