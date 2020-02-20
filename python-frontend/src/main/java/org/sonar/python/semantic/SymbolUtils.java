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
package org.sonar.python.semantic;

import java.io.File;
import java.net.URI;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;

public class SymbolUtils {

  private SymbolUtils() {
  }

  public static String fullyQualifiedModuleName(String packageName, String fileName) {
    int extensionIndex = fileName.lastIndexOf('.');
    String moduleName = extensionIndex > 0
      ? fileName.substring(0, extensionIndex)
      : fileName;
    if (moduleName.equals("__init__")) {
      return packageName;
    }
    return packageName.isEmpty()
      ? moduleName
      : (packageName + "." + moduleName);
  }

  public static Set<Symbol> globalSymbols(FileInput fileInput, String packageName, PythonFile pythonFile) {
    SymbolTableBuilder symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile);
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, pythonFile.fileName());
    fileInput.accept(symbolTableBuilder);
    Set<Symbol> globalSymbols = new HashSet<>();
    for (Symbol globalVariable : fileInput.globalVariables()) {
      if (globalVariable.kind() == Symbol.Kind.CLASS) {
        globalSymbols.add(new ClassSymbolImpl((ClassSymbol) globalVariable));
      } else if (globalVariable.kind() == Symbol.Kind.FUNCTION) {
        globalSymbols.add(new FunctionSymbolImpl(globalVariable.name(), ((FunctionSymbol) globalVariable)));
      } else {
        globalSymbols.add(new SymbolImpl(globalVariable.name(), fullyQualifiedModuleName + "." + globalVariable.name()));
      }
    }
    return globalSymbols;
  }

  @CheckForNull
  public static String getTypeName(@Nullable Symbol objectSymbol) {
    return Optional.ofNullable(getTypeSymbol(objectSymbol))
      .map(Symbol::fullyQualifiedName)
      .orElse(null);
  }

  @CheckForNull
  public static Symbol getTypeSymbol(@Nullable Symbol objectSymbol) {
    if (objectSymbol == null) {
      return null;
    }
    Type type = ((SymbolImpl) objectSymbol).type();
    if (type != null) {
      return type.symbol();
    }
    return null;
  }

  static void resolveTypeHierarchy(ClassDef classDef, @Nullable Symbol symbol) {
    if (symbol == null || !Symbol.Kind.CLASS.equals(symbol.kind())) {
      return;
    }
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
    ArgList argList = classDef.args();
    classSymbol.setHasUnresolvedTypeHierarchy(false);
    if (argList == null) {
      return;
    }
    for (Argument argument : argList.arguments()) {
      if (!argument.is(Kind.REGULAR_ARGUMENT) || !(((RegularArgument) argument).expression() instanceof HasSymbol)) {
        classSymbol.setHasUnresolvedTypeHierarchy(true);
        continue;
      }
      Expression expression = ((RegularArgument) argument).expression();
      Symbol argumentSymbol = ((HasSymbol) expression).symbol();
      if (argumentSymbol == null) {
        classSymbol.setHasUnresolvedTypeHierarchy(true);
      } else {
        if (!Symbol.Kind.CLASS.equals(argumentSymbol.kind()) && !BuiltinSymbols.all().contains(argumentSymbol.fullyQualifiedName())) {
          classSymbol.setHasUnresolvedTypeHierarchy(true);
        }
        classSymbol.addSuperClass(argumentSymbol);
      }
    }
  }

  static List<Expression> assignmentsLhs(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .flatMap(SymbolUtils::flattenTuples)
      .collect(Collectors.toList());
  }

  private static Stream<Expression> flattenTuples(Expression expression) {
    if (expression.is(Kind.TUPLE)) {
      Tuple tuple = (Tuple) expression;
      return tuple.elements().stream().flatMap(SymbolUtils::flattenTuples);
    } else {
      return Stream.of(expression);
    }
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

  public static String pythonPackageName(File file, File projectBaseDir) {
    File currentDirectory = file.getParentFile();
    Deque<String> packages = new ArrayDeque<>();
    while (!currentDirectory.getAbsolutePath().equals(projectBaseDir.getAbsolutePath())) {
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
  static Path pathOf(PythonFile pythonFile) {
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

  public static Map<String, Set<Symbol>> externalModulesSymbols() {
    Map<String, Set<Symbol>> globalSymbols = new HashMap<>();
    globalSymbols.put("flask_mail", new HashSet<>(Arrays.asList(
      classSymbol("Mail", "flask_mail.Mail"),
      classSymbol("Connection", "flask_mail.Connection")
      )));
    globalSymbols.put("smtplib", new HashSet<>(Arrays.asList(
      classSymbol("SMTP", "smtplib.SMTP"),
      classSymbol("SMTP_SSL", "smtplib.SMTP_SSL")
    )));
    globalSymbols.put("zipfile", new HashSet<>(Collections.singleton(classSymbol("ZipFile", "zipfile.ZipFile"))));
    globalSymbols.put("http.cookies", new HashSet<>(Collections.singletonList(classSymbol("SimpleCookie", "http.cookies.SimpleCookie"))));

    globalSymbols.put("django.http", new HashSet<>(Arrays.asList(
      classSymbol("HttpResponse", "django.http.HttpResponse"),
      classSymbol("HttpResponseRedirect", "django.http.HttpResponseRedirect"),
      classSymbol("HttpResponsePermanentRedirect", "django.http.HttpResponsePermanentRedirect"),
      classSymbol("HttpResponseNotModified", "django.http.HttpResponseNotModified"),
      classSymbol("HttpResponseNotFound", "django.http.HttpResponseNotFound"),
      classSymbol("HttpResponseForbidden", "django.http.HttpResponseForbidden"),
      classSymbol("HttpResponseNotAllowed", "django.http.HttpResponseNotAllowed"),
      classSymbol("HttpResponseGone", "django.http.HttpResponseGone"),
      classSymbol("HttpResponseServerError", "django.http.HttpResponseServerError"),
      classSymbol("HttpResponseBadRequest", "django.http.HttpResponseBadRequest")
    )));

    globalSymbols.put("django.http.response", new HashSet<>(Collections.singleton(
      classSymbol("HttpResponse", "django.http.response.HttpResponse")
    )));

    ClassSymbolImpl flaskResponse = classSymbol("Response", "flask.Response");
    globalSymbols.put("flask", new HashSet<>(Arrays.asList(
      flaskResponse,
      new FunctionSymbolImpl("make_response", "flask.make_response", false, false, false, Collections.emptyList(), new Type(flaskResponse)),
      new FunctionSymbolImpl("redirect", "flask.redirect", false, false, false, Collections.emptyList(), new Type(flaskResponse))
    )));

    globalSymbols.put("werkzeug.datastructures", new HashSet<>(Collections.singleton(
      classSymbol("Headers", "werkzeug.datastructures.Headers")
    )));

    return globalSymbols;
  }

  private static ClassSymbolImpl classSymbol(String name, String fullyQualifiedName) {
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(name, fullyQualifiedName);
    classSymbol.setHasUnresolvedTypeHierarchy(false);
    return classSymbol;
  }
}
