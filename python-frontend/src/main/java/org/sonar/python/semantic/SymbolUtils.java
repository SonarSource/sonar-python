/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.python.tree.TreeUtils.getSymbolFromTree;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
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
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.v2.TypeUtils;

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
    if ("__init__".equals(moduleName)) {
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
      if ("metaclass".equals(keyword.name())) {
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
      classSymbol.addSuperClass(argumentSymbol);
    }
  }

  @CheckForNull
  private static Symbol getSymbolFromArgument(RegularArgument regularArgument) {
    Expression expression = regularArgument.expression();
    while (expression.is(Kind.SUBSCRIPTION)) {
      // to support using 'typing' symbols like 'List[str]'
      expression = ((SubscriptionExpression) expression).object();
    }
    if (expression instanceof HasSymbol hasSymbol) {
      return hasSymbol.symbol();
    }
    return null;
  }

  public static List<Expression> assignmentsLhs(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .flatMap(TreeUtils::flattenTuples)
      .toList();
  }

  public static List<Name> boundNamesFromExpression(@CheckForNull Tree tree) {
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

  /**
   * Computes the Python package name for a file using package roots.
   *
   * <p>Finds the first root whose path is a prefix of the file's path and computes the
   * package name by converting the relative path from that root to the file's parent directory
   * into a dotted package name. This supports PEP 420 namespace packages (no __init__.py required).
   *
   * <p>Returns an empty string when the file does not fall under any of the provided roots.
   *
   * @param file the Python source file
   * @param packageRoots list of absolute paths to package root directories
   * @return the dotted package name, or empty string if not under any root
   */
  public static String pythonPackageName(File file, List<String> packageRoots) {
    List<Path> resolvedPackageRoots = packageRoots.stream()
      .map(root -> resolvedPath(new File(root)))
      .toList();
    return pythonPackageName(file, resolvedPackageRoots);
  }

  /**
   * Computes the Python package name using package roots that have already been resolved across
   * filesystem aliases. This overload avoids resolving the same roots for every file in a project.
   */
  public static String pythonPackageName(File file, Iterable<Path> resolvedPackageRoots) {
    Path filePath = resolvedPath(file);
    for (Path rootPath : resolvedPackageRoots) {
      if (filePath.startsWith(rootPath)) {
        return computePackageNameFromRoot(filePath, rootPath);
      }
    }
    return "";
  }

  /**
   * Computes the Python package name for a file relative to a package root.
   * All directories between the root and the file become part of the package name
   * (PEP 420 namespace package support).
   */
  private static String computePackageNameFromRoot(Path file, Path packageRoot) {
    Path parentDir = file.getParent();
    if (parentDir == null || !parentDir.startsWith(packageRoot)) {
      return "";
    }
    Path relativePath = packageRoot.relativize(parentDir);
    if (relativePath.getNameCount() == 0) {
      return "";
    }
    return StreamSupport.stream(relativePath.spliterator(), false)
      .map(Path::toString)
      .collect(Collectors.joining("."));
  }

  /**
   * Resolves filesystem aliases, including Windows 8.3 short paths. If the complete path does
   * not exist, the nearest existing ancestor is resolved and the remaining path is appended.
   */
  public static Path resolvedPath(File file) {
    Path absolutePath = file.toPath().toAbsolutePath().normalize();
    Path existingAncestor = absolutePath;
    while (existingAncestor != null) {
      try {
        return existingAncestor.toRealPath().resolve(existingAncestor.relativize(absolutePath));
      } catch (IOException e) {
        existingAncestor = existingAncestor.getParent();
      }
    }
    return absolutePath;
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

    var hasUnknownFirstParameter = parameters.stream()
      .findFirst()
      .filter(p -> Optional.ofNullable(p.declaredType())
        .filter(Predicate.not(InferredTypes.anyType()::equals))
        .isEmpty()
      )
      .filter(p -> p.name() == null)
      .isPresent();

    if (hasUnknownFirstParameter) {
      // First parameter is defined as a tuple
      return -1;
    }
    List<String> decoratorNames = functionSymbol.decorators();
    if (decoratorNames.size() > 1) {
      // We want to avoid FP if there are many decorators
      return -1;
    }
    if (!decoratorNames.isEmpty() && 
        !decoratorNames.get(0).endsWith("classmethod") && 
        !decoratorNames.get(0).endsWith("staticmethod") &&
        !decoratorNames.get(0).endsWith("abstractmethod")) {
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
    var symbols = getOverriddenMethods(functionSymbol);
    if(symbols.size() == 1){
      return symbols.stream().findFirst();
    }
    return Optional.empty();
  }

  public static List<FunctionSymbol> getOverriddenMethods(FunctionSymbol functionSymbol) {
    Symbol owner = ((FunctionSymbolImpl) functionSymbol).owner();
    if (owner == null || owner.kind() != CLASS) {
      return List.of();
    }
    ClassSymbol classSymbol = (ClassSymbol) owner;

    return classSymbol.superClasses()
      .stream()
      .filter(ClassSymbol.class::isInstance)
      .map(ClassSymbol.class::cast)
      .map(c -> c.resolveMember(functionSymbol.name())
        .map(SymbolUtils::getFunctionSymbols)
        .orElseGet(List::of))
      .filter(Predicate.not(List::isEmpty))
      .findFirst()
      .orElseGet(List::of);
  }

  public static List<FunctionSymbol> getFunctionSymbols(@Nullable Symbol symbol) {
    if (symbol == null) {
      return List.of();
    }

    if (symbol.is(Symbol.Kind.FUNCTION)) {
      return List.of((FunctionSymbol) symbol);
    }

    if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
      var ambiguousSymbol = (AmbiguousSymbol) symbol;

      var functionSymbols = ambiguousSymbol
        .alternatives()
        .stream()
        .filter(alternative -> alternative.is(Symbol.Kind.FUNCTION))
        .map(FunctionSymbol.class::cast)
        .toList();

      if (functionSymbols.size() != ambiguousSymbol.alternatives().size()) {
        return List.of();
      }

      return functionSymbols;
    }
    return List.of();
  }

  public static Optional<FunctionSymbol> getFirstAlternativeIfEqualArgumentNames(List<FunctionSymbol> alternatives) {
    return Optional.of(alternatives)
      .filter(SymbolUtils::isEqualParameterCountAndNames)
      .map(Collection::stream)
      .flatMap(Stream::findFirst);
  }

  public static boolean isEqualParameterCountAndNames(List<FunctionSymbol> alternatives) {
    return alternatives.stream()
      .map(FunctionSymbol::parameters)
      .filter(Objects::nonNull)
      .map(parameters -> parameters.stream()
        .map(parameter -> List.of(Objects.requireNonNullElse(parameter.name(), ""), parameter.isKeywordOnly(), parameter.isPositionalOnly()))
        .toList()
      ).distinct()
      .count() == 1;
  }


  public static boolean canBeAnOverridingMethod(@Nullable FunctionSymbol functionSymbol) {
    if (functionSymbol == null) return true;
    Symbol owner = ((FunctionSymbolImpl) functionSymbol).owner();
    if (owner == null || owner.kind() != CLASS) return false;
    ClassSymbol classSymbol = (ClassSymbol) owner;
    if (classSymbol.hasUnresolvedTypeHierarchy()) return true;
    for (Symbol superClass : classSymbol.superClasses()) {
      if (superClass.is(CLASS)) {
        boolean canHaveMember = ((ClassSymbol) superClass).canHaveMember(functionSymbol.name());
        if (canHaveMember) return true;
      }
    }
    return false;
  }

  public static Symbol typeshedSymbolWithFQN(String fullyQualifiedName) {
    String[] fqnSplitByDot = fullyQualifiedName.split("\\.");
    String localName = fqnSplitByDot[fqnSplitByDot.length - 1];
    Symbol symbol = TypeShed.symbolWithFQN(fullyQualifiedName);
    return symbol == null ? new SymbolImpl(localName, fullyQualifiedName) : ((SymbolImpl) symbol).copyWithoutUsages();
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

  public static boolean isPrivateName(String name) {
    return name.startsWith("_") && !name.startsWith("__");
  }

  public static String qualifiedNameOrEmpty(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && symbol.fullyQualifiedName() != null ? symbol.fullyQualifiedName() : "";
  }

  public static PythonType getPythonType(SymbolV2 symbol) {
    return symbol.usages().stream()
      .filter(UsageV2::isBindingUsage)
      .map(UsageV2::tree)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(Expression.class))
      .map(Expression::typeV2)
      .collect(TypeUtils.toUnionType());
  }

  public static Optional<Symbol> symbolV2ToSymbolV1(SymbolV2 symbolV2) {
    return symbolV2.usages().stream()
      .filter(UsageV2::isBindingUsage)
      .map(UsageV2::tree)
      .filter(HasSymbol.class::isInstance)
      .map(tree -> ((HasSymbol) tree).symbol())
      .filter(Objects::nonNull)
      .findFirst();
  }
}
