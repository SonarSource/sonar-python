/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5918")
public class ImplicitlySkippedTestCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Skip this test explicitly.";

  private static final List<TestFramework> testFrameworks = new ArrayList<>();

  static {
    // Unit Test method source : https://docs.python.org/2/library/unittest.html#assert-methods
    testFrameworks.add(new TestFramework("UnitTest", Arrays.asList("unittest", "TestCase"), new HashSet<>(Arrays.asList("assertEqual",
      "assertNotEqual", "assertTrue", "assertFalse", "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn",
      "assertNotIn", "assertIsInstance", "assertNotIsInstance", "assertRaises", "assertRaisesRegexp", "assertAlmostEqual",
      "assertNotAlmostEqual", "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegexpMatches",
      "assertNotRegexpMatches", "assertItemsEqual", "assertDictContainsSubset", "assertMultiLineEqual", "assertSequenceEqual",
      "assertListEqual", "assertTupleEqual", "assertSetEqual", "assertDictEqual"))));
  }

  private static final Tree.Kind[] literalsKind = {Tree.Kind.STRING_LITERAL, Tree.Kind.NUMERIC_LITERAL, Tree.Kind.LIST_LITERAL,
    Tree.Kind.BOOLEAN_LITERAL_PATTERN, Tree.Kind.NUMERIC_LITERAL_PATTERN, Tree.Kind.NONE_LITERAL_PATTERN, Tree.Kind.STRING_LITERAL_PATTERN};

  static class TestFramework {
    String name;
    List<String> keywords;
    Set<String> supportedAssertMethods;

    public TestFramework(String name, List<String> keywords, Set<String> supportedAssertMethods) {
      this.name = name;
      this.keywords = keywords;
      this.supportedAssertMethods = supportedAssertMethods;
    }

    public boolean matchAnyProvidedClasses(List<String> classes) {
      return classes.stream().anyMatch(parentClass -> keywords.stream().allMatch(parentClass::contains));
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

      if (!functionDef.name().name().startsWith("test")) {
        return;
      }

      ReturnStatement returnStatement = getReturnStatementFromFirstIfStatement(functionDef);
      if (returnStatement == null) {
        return;
      }

      // We add an issue only if there is an assert statement somewhere in the function definition
      if (containsAssertion(functionDef)) {
        ctx.addIssue(returnStatement, MESSAGE);
      }
    });
  }

  private static ReturnStatement getReturnStatementFromFirstIfStatement(FunctionDef functionDef) {
    // check first non assignment statement is an if with a return
    for (Statement statement : functionDef.body().statements()) {
      // skip literal assignment expression
      if (statement.is(Tree.Kind.ASSIGNMENT_STMT) && ((AssignmentStatement) statement).assignedValue().is(literalsKind)) {
        continue;
      }

      if (statement.is(Tree.Kind.IF_STMT)) {
        IfStatement ifStatement = (IfStatement) statement;
        List<Statement> statements = ifStatement.body().statements();
        if (statements.get(0).is(Tree.Kind.RETURN_STMT)) {
          return (ReturnStatement) statements.get(0);
        }
      }

      return null;
    }
    return null;
  }

  private static boolean containsAssertion(FunctionDef functionDef) {
    List<String> parentClasses = getParentClassFromFunctionDef(functionDef);
    AssertionVisitor assertVisitor = new AssertionVisitor(parentClasses);
    functionDef.accept(assertVisitor);
    return assertVisitor.hasAnAssert;
  }

  private static List<String> getParentClassFromFunctionDef(FunctionDef functionDef) {
    List<String> libraries = new ArrayList<>();
    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(functionDef, Tree.Kind.CLASSDEF);
    if (classDef != null) {
      libraries.addAll(getInheritedClassesFQN(classDef));
    }

    return libraries;
  }

  private static List<String> getInheritedClassesFQN(ClassDef classDefinition) {
    return getParentClasses(TreeUtils.getClassSymbolFromDef(classDefinition)).stream()
      .map(Symbol::fullyQualifiedName)
      .collect(Collectors.toList());
  }

  private static List<Symbol> getParentClasses(ClassSymbol classSymbol) {
    List<Symbol> superclasses = new ArrayList<>();
    if (classSymbol != null) {
      for (Symbol symbol : classSymbol.superClasses()) {
        superclasses.add(symbol);
        if (symbol instanceof ClassSymbol) {
          superclasses.addAll(getParentClasses((ClassSymbol) symbol));
        }
      }
    }
    return superclasses;
  }

  static class AssertionVisitor extends BaseTreeVisitor {
    boolean hasAnAssert = false;
    List<String> parentClasses;

    public AssertionVisitor(List<String> parentClasses) {
      this.parentClasses = parentClasses;
    }

    @Override
    public void visitAssertStatement(AssertStatement assertStatement) {
      hasAnAssert = true;
      super.visitAssertStatement(assertStatement);
    }

    @Override
    public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
      if (qualifiedExpression.qualifier().is(Tree.Kind.NAME) && "self".equals(((Name) qualifiedExpression.qualifier()).name())
        && isMethodFromTestFramework(qualifiedExpression.name().name())) {
        hasAnAssert = true;
      }
      super.visitQualifiedExpression(qualifiedExpression);
    }

    private boolean isMethodFromTestFramework(String methodName) {
      return testFrameworks.stream()
        .anyMatch(testFramework -> testFramework.matchAnyProvidedClasses(parentClasses) && testFramework.supportedAssertMethods.contains(methodName));
    }
  }
}
