/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
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

  public static Set<Symbol> globalSymbols(FileInput fileInput, String fullyQualifiedModuleName) {
    GlobalSymbolsVisitor globalSymbolsVisitor = new GlobalSymbolsVisitor(fullyQualifiedModuleName);
    fileInput.accept(globalSymbolsVisitor);
    return new HashSet<>(globalSymbolsVisitor.symbolsByName.values());
  }

  private static class GlobalSymbolsVisitor extends BaseTreeVisitor {
    private Map<String, Symbol> symbolsByName = new HashMap<>();
    private String fullyQualifiedModuleName;

    GlobalSymbolsVisitor(String fullyQualifiedModuleName) {
      this.fullyQualifiedModuleName = fullyQualifiedModuleName;
    }

    private Symbol symbol(Tree tree) {
      if (tree.is(Kind.FUNCDEF)) {
        FunctionDef functionDef = (FunctionDef) tree;
        return new FunctionSymbolImpl(functionDef, fullyQualifiedModuleName + "." + functionDef.name().name());
      }
      Name name = (Name) tree;
      return new SymbolImpl(name.name(), fullyQualifiedModuleName + "." + name.name());
    }

    private void addSymbol(Tree tree, String name) {
      SymbolImpl symbol = (SymbolImpl) symbolsByName.get(name);
      if (symbol != null) {
        symbol.setKind(Symbol.Kind.OTHER);
      } else {
        symbolsByName.put(name, symbol(tree));
      }
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      addSymbol(functionDef, functionDef.name().name());
    }

    @Override
    public void visitClassDef(ClassDef classDef) {
      addSymbol(classDef.name(), classDef.name().name());
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
      assignmentsLhs((assignmentStatement)).stream()
        .map(SymbolUtils::boundNamesFromExpression)
        .flatMap(Collection::stream)
        .forEach(name -> addSymbol(name, name.name()));
      super.visitAssignmentStatement(assignmentStatement);
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
      if (annotatedAssignment.variable().is(Kind.NAME)) {
        Name variable = (Name) annotatedAssignment.variable();
        addSymbol(variable, variable.name());
      }
      super.visitAnnotatedAssignment(annotatedAssignment);
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
}
