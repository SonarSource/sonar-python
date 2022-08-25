package org.sonar.python.checks.tests;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class TestFrameworkUtils {
  private static final List<TestFramework> testFrameworks = new ArrayList<>();
  static {
    // Unit Test method source : https://docs.python.org/2/library/unittest.html#assert-methods
    testFrameworks.add(new TestFramework("unittest", List.of("unittest", "TestCase"), Set.of("assertEqual", "assertNotEqual",
      "assertTrue", "assertFalse", "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn",
      "assertIsInstance", "assertNotIsInstance", "assertRaises", "assertRaisesRegexp", "assertAlmostEqual", "assertNotAlmostEqual",
      "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegexpMatches", "assertNotRegexpMatches",
      "assertItemsEqual", "assertDictContainsSubset", "assertMultiLineEqual", "assertSequenceEqual", "assertListEqual", "assertTupleEqual",
      "assertSetEqual", "assertDictEqual"), Set.of("assertRaises")));
    testFrameworks.add(new TestFramework("pytest", List.of("pytest"), Set.of(), Set.of("raises")));
  }

  static class TestFramework {
    String name;
    List<String> keywords;
    Set<String> supportedAssertMethods;
    Set<String> otherMethods;

    public TestFramework(String name, List<String> keywords, Set<String> supportedAssertMethods, Set<String> otherMethods) {
      this.name = name;
      this.keywords = keywords;
      this.supportedAssertMethods = supportedAssertMethods;
      this.otherMethods = otherMethods;
    }

    public boolean matchAnyProvidedClasses(List<String> classes) {
      return classes.stream().anyMatch(parentClass -> keywords.stream().allMatch(parentClass::contains));
    }
  }

  private TestFrameworkUtils() {}

  public static boolean isAnAssertOfAnySupportedTestFramework(CallExpression callExpression) {
    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.CLASSDEF);
    if (classDef == null) {
      return false;
    }

    Set<String> methods = getParentClassTestFrameworkFromFunctionDef(classDef);

    return Optional.of(callExpression).stream()
      .filter(callExpr -> callExpr.callee().is(Tree.Kind.QUALIFIED_EXPR))
      .map(callExpr -> (QualifiedExpression) callExpr.callee())
      .filter(callee -> callee.qualifier().is(Tree.Kind.NAME) && ((Name) callee.qualifier()).name().equals("self"))
      .anyMatch(callee -> methods.contains(callee.name().name()));
  }

  private static Set<String> getParentClassTestFrameworkFromFunctionDef(ClassDef classDef) {
    return testFrameworks.stream()
      .filter(testFramework -> testFramework.matchAnyProvidedClasses(getInheritedClassesFQN(classDef)))
      .findFirst()
      .map(t -> t.supportedAssertMethods)
      .orElseGet(Collections::emptySet);
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
}
