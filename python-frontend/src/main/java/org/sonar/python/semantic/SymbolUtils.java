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
package org.sonar.python.semantic;

import java.io.File;
import java.net.URI;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.TypeShedPythonFile;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.python.tree.TreeUtils.getSymbolFromTree;

public class SymbolUtils {

  private SymbolUtils() {
  }

  public static String getModuleFileName(String fileName) {
    int extensionIndex = fileName.lastIndexOf('.');
    return extensionIndex > 0
      ? fileName.substring(0, extensionIndex)
      : fileName;
  }

  public static String fullyQualifiedModuleName(String packageName, String fileName) {
    String moduleName = getModuleFileName(fileName);
    if (moduleName.equals("__init__")) {
      return packageName;
    }
    return packageName.isEmpty()
      ? moduleName
      : (packageName + "." + moduleName);
  }

  static void resolveTypeHierarchy(ClassDef classDef, @Nullable Symbol symbol, PythonFile pythonFile, Map<String, Symbol> symbolsByName) {
    if (symbol == null || !CLASS.equals(symbol.kind())) {
      return;
    }
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
    if (isBuiltinTypeshedFile(pythonFile) && "str".equals(classSymbol.fullyQualifiedName())) {
      classSymbol.addSuperClass(symbolsByName.get("object"));
      classSymbol.addSuperClass(symbolsByName.get("Sequence"));
      return;
    }
    ArgList argList = classDef.args();
    if (argList == null) {
      return;
    }
    for (Argument argument : argList.arguments()) {
      if (!argument.is(Kind.REGULAR_ARGUMENT)) {
        classSymbol.setHasSuperClassWithoutSymbol();
      } else {
        addParentClass(pythonFile, symbolsByName, classSymbol, (RegularArgument) argument);
      }
    }
  }

  private static void addParentClass(PythonFile pythonFile, Map<String, Symbol> symbolsByName, ClassSymbolImpl classSymbol, RegularArgument regularArgument) {
    Name keyword = regularArgument.keywordArgument();
    if (keyword != null) {
      if (keyword.name().equals("metaclass")) {
        classSymbol.setHasMetaClass();
        getSymbolFromTree(regularArgument.expression())
          .map(Symbol::fullyQualifiedName)
          .ifPresent(classSymbol::setMetaclassFQN);
      }
      return;
    }
    Symbol argumentSymbol = getSymbolFromArgument(regularArgument);
    if (argumentSymbol == null) {
      classSymbol.setHasSuperClassWithoutSymbol();
    } else {
      if ("typing.Generic".equals(argumentSymbol.fullyQualifiedName())) {
        classSymbol.setSupportsGenerics(true);
      }
      Symbol normalizedArgumentSymbol = normalizeSymbol(argumentSymbol, pythonFile, symbolsByName);
      if (normalizedArgumentSymbol != null) {
        classSymbol.addSuperClass(normalizedArgumentSymbol);
      }
    }
  }

  /**
   * Hardcoding some 'typing' module symbols to avoid incomplete type hierarchy for type 'str'
   */
  @CheckForNull
  private static Symbol normalizeSymbol(Symbol symbol, PythonFile pythonFile, Map<String, Symbol> symbolsByName) {
    if (isTypeShedFile(pythonFile) && (symbol.name().equals("Protocol") || symbol.name().equals("Generic"))) {
      // ignore Protocol and Generic to avoid having incomplete type hierarchies
      return null;
    }
    if (isTypingFile(pythonFile) && symbol.name().equals("_Collection")) {
      return symbolsByName.get("Collection");
    }
    return symbol;
  }

  private static boolean isBuiltinTypeshedFile(PythonFile pythonFile) {
    return isTypeShedFile(pythonFile) && pythonFile.fileName().isEmpty();
  }

  private static boolean isTypingFile(PythonFile pythonFile) {
    return isTypeShedFile(pythonFile) && pythonFile.fileName().equals("typing");
  }

  @CheckForNull
  private static Symbol getSymbolFromArgument(RegularArgument regularArgument) {
    Expression expression = regularArgument.expression();
    while (expression.is(Kind.SUBSCRIPTION)) {
      // to support using 'typing' symbols like 'List[str]'
      expression = ((SubscriptionExpression) expression).object();
    }
    if (expression instanceof HasSymbol) {
      return ((HasSymbol) expression).symbol();
    }
    return null;
  }

  public static List<Expression> assignmentsLhs(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .flatMap(TreeUtils::flattenTuples)
      .collect(Collectors.toList());
  }

  static List<Name> boundNamesFromExpression(@CheckForNull Tree tree) {
    List<Name> names = new ArrayList<>();
    if (tree == null) {
      return names;
    }
    if (tree.is(Tree.Kind.NAME)) {
      names.add(((Name) tree));
    } else if (tree.is(Tree.Kind.TUPLE)) {
      ((Tuple) tree).elements().forEach(t -> names.addAll(boundNamesFromExpression(t)));
    } else if (tree.is(Kind.LIST_LITERAL)) {
      ((ListLiteral) tree).elements().expressions().forEach(t -> names.addAll(boundNamesFromExpression(t)));
    } else if (tree.is(Kind.PARENTHESIZED)) {
      names.addAll(boundNamesFromExpression(((ParenthesizedExpression) tree).expression()));
    } else if (tree.is(Kind.UNPACKING_EXPR)) {
      names.addAll(boundNamesFromExpression(((UnpackingExpression) tree).expression()));
    }
    return names;
  }

  public static String pythonPackageName(File file, String projectBaseDirAbsolutePath) {
    File currentDirectory = file.getParentFile();
    Deque<String> packages = new ArrayDeque<>();
    while (!currentDirectory.getAbsolutePath().equals(projectBaseDirAbsolutePath)) {
      File initFile = new File(currentDirectory, "__init__.py");
      if (!initFile.exists()) {
        break;
      }
      packages.push(currentDirectory.getName());
      currentDirectory = currentDirectory.getParentFile();
    }
    return String.join(".", packages);
  }

  @CheckForNull
  public static Path pathOf(PythonFile pythonFile) {
    try {
      URI uri = pythonFile.uri();
      if ("file".equalsIgnoreCase(uri.getScheme())) {
        return Paths.get(uri);
      }
      return null;
    } catch (InvalidPathException e) {
      return null;
    }
  }

  public static boolean isTypeShedFile(PythonFile pythonFile) {
    return pythonFile instanceof TypeShedPythonFile;
  }

  /**
   * @return the offset between parameter position and argument position:
   *   0 if there is no implicit first parameter (self, cls, etc...)
   *   1 if there is an implicit first parameter
   *  -1 if unknown (intent is not clear from context)
   */
  public static int firstParameterOffset(FunctionSymbol functionSymbol, boolean isStaticCall) {
    List<FunctionSymbol.Parameter> parameters = functionSymbol.parameters();
    if (parameters.isEmpty()) {
      return 0;
    }
    String firstParamName = parameters.get(0).name();
    if (firstParamName == null) {
      // First parameter is defined as a tuple
      return -1;
    }
    List<String> decoratorNames = functionSymbol.decorators();
    if (decoratorNames.size() > 1) {
      // We want to avoid FP if there are many decorators
      return -1;
    }
    if (!decoratorNames.isEmpty() && !decoratorNames.get(0).endsWith("classmethod") && !decoratorNames.get(0).endsWith("staticmethod")) {
      // Unknown decorator which might alter the behaviour of the method
      return -1;
    }
    if (functionSymbol.isInstanceMethod() && !isStaticCall) {
      // regular instance call, takes self as first implicit parameter
      return 1;
    }
    if (decoratorNames.size() == 1 && decoratorNames.get(0).endsWith("classmethod")) {
      // class method call, takes cls as first implicit parameter
      return 1;
    }
    // regular static call (function or method), no first implicit parameter
    return 0;
  }

  public static Optional<FunctionSymbol> getOverriddenMethod(FunctionSymbol functionSymbol) {
    Symbol owner = ((FunctionSymbolImpl) functionSymbol).owner();
    if (owner == null || owner.kind() != CLASS) {
      return Optional.empty();
    }
    ClassSymbol classSymbol = (ClassSymbol) owner;
    if (classSymbol.superClasses().isEmpty()) {
      return Optional.empty();
    }
    for (Symbol superClass : classSymbol.superClasses()) {
      if (superClass.kind() == CLASS) {
        Optional<FunctionSymbol> overriddenSymbol = ((ClassSymbol) superClass).resolveMember(functionSymbol.name())
          .filter(symbol -> symbol.kind() == FUNCTION)
          .map(FunctionSymbol.class::cast);
        if (overriddenSymbol.isPresent()) {
          return overriddenSymbol;
        }
      }
    }
    return Optional.empty();
  }

  public static Symbol typeshedSymbolWithFQN(String fullyQualifiedName) {
    String[] fqnSplitByDot = fullyQualifiedName.split("\\.");
    String localName = fqnSplitByDot[fqnSplitByDot.length - 1];
    Symbol symbol = TypeShed.symbolWithFQN(fullyQualifiedName);
    return symbol == null ? new SymbolImpl(localName, fullyQualifiedName) : symbol;
  }

  public static Set<Symbol> flattenAmbiguousSymbols(Set<Symbol> symbols) {
    Set<Symbol> alternatives = new HashSet<>();
    for (Symbol symbol : symbols) {
      if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
        Set<Symbol> flattenedAlternatives = flattenAmbiguousSymbols(((AmbiguousSymbol) symbol).alternatives());
        alternatives.addAll(flattenedAlternatives);
      } else {
        alternatives.add(symbol);
      }
    }
    return alternatives;
  }
}
