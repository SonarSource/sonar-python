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
package org.sonar.python.checks.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.caching.CacheContextImpl.dummyCache;

class UnittestUtilsTest {

  @Test
  void test_isWithinUnittestTestCase() {
    String code = "import unittest\nclass A(unittest.TestCase):  ...";
    FileInput fileInput = parse("mod1.py", code);
    Tree tree = lastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isTrue();

    code = "import unittest\nclass A(unittest.case.TestCase):  ...";
    fileInput = parse("mod1.py", code);
    tree = lastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isTrue();

    code = "import random_wrapper\nclass A(random_wrapper.unittest.TestCase):  ...";
    fileInput = parse("mod1.py", code);
    tree = lastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isTrue();

    code = "import random\nclass A(random.TestCase):  ...";
    fileInput = parse("mod1.py", code);
    tree = lastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isFalse();

    code = "import unittest\nclass A(unittest.other):  ...";
    fileInput = parse("mod1.py", code);
    tree = lastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isFalse();

    code = "...";
    fileInput = parse("mod1.py", code);
    tree = lastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isFalse();
  }

  @Test
  void all_methods() {
    assertThat(UnittestUtils.allMethods())
      .hasSize(61)
      .contains("skipTest")
      .doesNotContain("skiptTest");
  }

  @Test
  void all_assert_methods() {
    assertThat(UnittestUtils.allAssertMethods()).hasSize(38);
  }

  @Test
  void test_name_helpers() {
    assertThat(UnittestUtils.isTestMethodName("test_something")).isTrue();
    assertThat(UnittestUtils.isTestMethodName("helper")).isFalse();
    assertThat(UnittestUtils.isTestClassName("TestSomething")).isTrue();
    assertThat(UnittestUtils.isTestClassName("Helper")).isFalse();
    assertThat(UnittestUtils.isPytestFileName("test_module.py")).isTrue();
    assertThat(UnittestUtils.isPytestFileName("module_test.py")).isTrue();
    assertThat(UnittestUtils.isPytestFileName("module.py")).isFalse();
  }

  @Test
  void test_constructor_detection() {
    FileInput fileInput = parse("test_module.py", """
      class TestWithInit:
        def __init__(self):
          pass

      class TestWithNew:
        def __new__(cls):
          return super().__new__(cls)

      class TestWithoutConstructor:
        def helper(self):
          pass
      """);

    var classes = fileInput.statements().statements().stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .toList();

    assertThat(UnittestUtils.hasConstructor(classes.get(0))).isTrue();
    assertThat(UnittestUtils.hasConstructor(classes.get(1))).isTrue();
    assertThat(UnittestUtils.hasConstructor(classes.get(2))).isFalse();
  }

  @Test
  void test_pytest_style_test_class() {
    FileInput fileInput = parse("test_module.py", """
      import pytest

      class TestCollected:
        @pytest.fixture()
        def setup_data(self):
          return 42

        def test_ok(self):
          pass

      class TestWithInit:
        def __init__(self):
          pass

      class TestWithSetUp:
        def setUp(self):
          pass

      class Helper:
        def test_ok(self):
          pass
      """);

    var classes = fileInput.statements().statements().stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .toList();

    assertThat(UnittestUtils.isPytestStyleTestClass(classes.get(0), "test_module.py")).isTrue();
    assertThat(UnittestUtils.isPytestStyleTestClass(classes.get(1), "test_module.py")).isFalse();
    assertThat(UnittestUtils.isPytestStyleTestClass(classes.get(2), "test_module.py")).isTrue();
    assertThat(UnittestUtils.isPytestStyleTestClass(classes.get(3), "test_module.py")).isFalse();
  }

  @Test
  void test_pytest_style_test_function() {
    FileInput fileInput = parse("test_module.py", """
      def test_module_level():
        pass

      class TestContainer:
        def test_in_class(self):
          pass

      class Helper:
        def test_helper(self):
          pass

        def wrapper(self):
          def test_nested():
            pass
          return test_nested

      def helper():
        pass
      """);

    var functions = allDescendants(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).stream()
      .map(FunctionDef.class::cast)
      .toList();

    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(0), "test_module.py")).isTrue();
    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(1), "test_module.py")).isTrue();
    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(2), "test_module.py")).isFalse();
    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(3), "test_module.py")).isFalse();
    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(4), "test_module.py")).isTrue();
    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(5), "test_module.py")).isFalse();
    assertThat(UnittestUtils.isPytestStyleTestFunction(functions.get(0), "module.py")).isFalse();
  }

  @Test
  void test_pytest_lifecycle_methods() {
    FileInput fileInput = parse("test_module.py", """
      import pytest

      class TestFixtureBased:
        @pytest.fixture(scope="module")
        def setup_data(self):
          return 42

      class TestTearDownBased:
        def tearDown(self):
          pass

      class TestXunitBased:
        def teardown_method(self):
          pass

      class TestWithoutLifecycle:
        def helper(self):
          pass
      """);

    var classes = fileInput.statements().statements().stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .toList();

    assertThat(UnittestUtils.hasPytestLifecycleMethods(classes.get(0))).isTrue();
    assertThat(UnittestUtils.hasPytestLifecycleMethods(classes.get(1))).isTrue();
    assertThat(UnittestUtils.hasPytestLifecycleMethods(classes.get(2))).isTrue();
    assertThat(UnittestUtils.hasPytestLifecycleMethods(classes.get(3))).isFalse();
  }

  @Test
  void test_isUnittestTestCaseClass() {
    String code = "import unittest\nclass A(unittest.TestCase):\n  ...";
    FileInput fileInput = parse("mod1.py", code);
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    assertThat(UnittestUtils.isUnittestTestCaseClass(classDef)).isTrue();

    code = "import random\nclass A(random.TestCase):\n  ...";
    fileInput = parse("mod1.py", code);
    classDef = (ClassDef) fileInput.statements().statements().get(1);
    assertThat(UnittestUtils.isUnittestTestCaseClass(classDef)).isFalse();
  }

  @Test
  void test_pytest_raises_helpers() {
    List<Boolean> isPytestRaises = new ArrayList<>();
    List<Boolean> hasMatchArgument = new ArrayList<>();
    List<String> exceptionArguments = new ArrayList<>();

    analyzeCallExpressions("""
      import pytest
      from pytest import raises
      pytest.raises(Exception, match="bad")
      raises(ValueError)
      """, (ctx, callExpression) -> {
      isPytestRaises.add(UnittestUtils.isPytestRaises(callExpression, ctx));
      hasMatchArgument.add(UnittestUtils.hasPytestRaisesMatchArgument(callExpression));
      exceptionArguments.add(UnittestUtils.pytestExpectedExceptionArgument(callExpression).expression().firstToken().value());
    });

    assertThat(isPytestRaises).containsExactly(true, true);
    assertThat(hasMatchArgument).containsExactly(true, false);
    assertThat(exceptionArguments).containsExactly("Exception", "ValueError");
  }

  @Test
  void test_unittest_assert_raises_helpers() {
    List<Boolean> isAssertRaises = new ArrayList<>();
    List<Boolean> hasMessageCheck = new ArrayList<>();
    List<String> exceptionArguments = new ArrayList<>();

    analyzeCallExpressions("""
      import unittest

      class MyTest(unittest.TestCase):
        def test_ok(self):
          self.assertRaises(Exception, explode)
          self.assertRaisesRegex(Exception, "bad", explode)
      """, (ctx, callExpression) -> {
      isAssertRaises.add(UnittestUtils.isUnittestAssertRaises(callExpression, ctx));
      hasMessageCheck.add(UnittestUtils.hasUnittestAssertRaisesMessageCheck(callExpression, ctx));
      exceptionArguments.add(UnittestUtils.unittestExceptionArgument(callExpression).expression().firstToken().value());
    });

    assertThat(isAssertRaises).containsExactly(true, true);
    assertThat(hasMessageCheck).containsExactly(false, true);
    assertThat(exceptionArguments).containsExactly("Exception", "Exception");
  }

  private static FileInput parse(String fileName, String code) {
    FileInput fileInput = new PythonTreeMaker().fileInput(PythonParser.create().parse(code));
    new SymbolTableBuilder("", new TestPythonVisitorRunner.MockPythonFile("", fileName, code)).visitFileInput(fileInput);
    return fileInput;
  }

  private static void analyzeCallExpressions(String code, BiConsumer<org.sonar.plugins.python.api.SubscriptionContext, CallExpression> consumer) {
    var file = new TestPythonVisitorRunner.MockPythonFile("", "mod1.py", code);
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file, null, "", ProjectLevelSymbolTable.empty(), dummyCache());
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> consumer.accept(ctx, (CallExpression) ctx.syntaxNode()));
      }
    };
    SubscriptionVisitor.analyze(List.of(check), context);
  }

  private static <T extends Tree> List<T> allDescendants(Tree tree, java.util.function.Predicate<Tree> predicate) {
    List<T> result = new ArrayList<>();
    for (Tree child : tree.children()) {
      if (predicate.test(child)) {
        result.add((T) child);
      }
      result.addAll(allDescendants(child, predicate));
    }
    return result;
  }

  private static <T extends Tree> T lastDescendant(Tree tree, java.util.function.Predicate<Tree> predicate) {
    List<T> descendants = allDescendants(tree, predicate);
    return descendants.get(descendants.size() - 1);
  }
}
