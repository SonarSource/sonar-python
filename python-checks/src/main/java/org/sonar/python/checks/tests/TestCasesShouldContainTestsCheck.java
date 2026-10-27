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
package org.sonar.python.checks.tests;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.UnittestUtils;

import static java.util.function.Predicate.not;

@Rule(key = "S2187")
public class TestCasesShouldContainTestsCheck extends PythonSubscriptionCheck {
  private static final String CLASS_MESSAGE = "Add some tests to this class.";
  private static final String FILE_MESSAGE = "Add some tests to this file.";
  private static final TypeMatcher ABSTRACT_BASE_CLASS_MATCHER = TypeMatchers.isOrExtendsType("abc.ABC");
  private static final TypeMatcher NOT_IMPLEMENTED_ERROR_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("builtins.NotImplementedError"),
    TypeMatchers.isObjectInstanceOf("builtins.NotImplementedError"));

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, TestCasesShouldContainTestsCheck::checkFileInput);
  }

  private static void checkFileInput(SubscriptionContext ctx) {
    FileInput fileInput = (FileInput) ctx.syntaxNode();
    StatementList statements = fileInput.statements();
    if (statements == null) {
      return;
    }

    boolean pytestStyleFile = UnittestUtils.isPytestFileName(ctx.pythonFile().fileName());
    List<ClassInfo> classes = collectClasses(ctx, statements, pytestStyleFile);
    List<ClassInfo> candidateClasses = classes.stream()
      .filter(ClassInfo::isCandidate)
      .toList();
    boolean hasCollectedTests = hasModuleLevelPytestTests(statements) || hasClassLevelTests(statements);

    if (pytestStyleFile && !hasCollectedTests) {
      ctx.addFileIssue(FILE_MESSAGE);
      return;
    }

    candidateClasses
            .stream()
            .filter(not(ClassInfo::hasCollectedTests))
            .filter(candidateClass -> !hasDescendantWithTests(candidateClass, classes))
            .filter(candidateClass -> !isLikelySharedBaseClass(candidateClass, ctx))
            .forEach(candidateClass -> ctx.addIssue(candidateClass.classDef().name(), CLASS_MESSAGE));
  }

  private static List<ClassInfo> collectClasses(SubscriptionContext ctx, StatementList statements, boolean pytestStyleFile) {
    return statements.statements()
      .stream().filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .map(classDef -> {
        ClassType classType = classType(classDef);
        return new ClassInfo(classDef, classType, hasCollectedTests(classDef), isCandidateTestClass(ctx, classDef, pytestStyleFile));
      })
      .toList();
  }

  private static boolean hasModuleLevelPytestTests(StatementList statements) {
    return statements.statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .anyMatch(TestCasesShouldContainTestsCheck::isCollectedTestMethod);
  }

  private static boolean hasClassLevelTests(StatementList statements) {
    return statements.statements().stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .anyMatch(TestCasesShouldContainTestsCheck::hasCollectedTests);
  }

  private static boolean isCandidateTestClass(SubscriptionContext ctx, ClassDef classDef, boolean pytestStyleFile) {
    return (pytestStyleFile && UnittestUtils.isPytestStyleTestClass(classDef, ctx.pythonFile().fileName()))
      || UnittestUtils.isUnittestTestCaseClass(classDef);
  }

  private static boolean hasCollectedTests(ClassDef classDef) {
    return hasLocalTestMethod(classDef) || hasInheritedTestMethod(classDef);
  }

  private static boolean hasLocalTestMethod(ClassDef classDef) {
    return classDef.body().statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .anyMatch(TestCasesShouldContainTestsCheck::isCollectedTestMethod);
  }

  private static boolean hasInheritedTestMethod(ClassDef classDef) {
    ClassType classType = classType(classDef);
    return classType != null && superClassesDeclareTests(classType, new LinkedHashSet<>());
  }

  private static boolean superClassesDeclareTests(ClassType classType, Set<ClassType> visitedTypes) {
    if (!visitedTypes.add(classType)) {
      return false;
    }
    return classType.superClasses().stream()
      .map(TypeWrapper::type)
      .filter(ClassType.class::isInstance)
      .map(ClassType.class::cast)
      .anyMatch(superClassType ->
        superClassesDeclareTests(superClassType, visitedTypes)
          || superClassType.members().stream()
            .filter(member -> member.type() instanceof FunctionType)
            .map(Member::name)
            .anyMatch(TestCasesShouldContainTestsCheck::isTestMethodName));
  }

  private static boolean hasDescendantWithTests(ClassInfo candidateClass, List<ClassInfo> classes) {
    ClassType classType = candidateClass.classType();
    if (classType == null) {
      return false;
    }
    return classes.stream()
      .filter(other -> other != candidateClass)
      .filter(ClassInfo::hasCollectedTests)
      .map(ClassInfo::classType)
      .filter(Objects::nonNull)
      .anyMatch(otherClassType -> isDescendantOf(otherClassType, classType, new LinkedHashSet<>()));
  }

  private static boolean isDescendantOf(ClassType classType, ClassType ancestor, Set<ClassType> visitedTypes) {
    if (!visitedTypes.add(classType)) {
      return false;
    }
    return classType.superClasses().stream()
      .map(TypeWrapper::type)
      .filter(ClassType.class::isInstance)
      .map(ClassType.class::cast)
      .anyMatch(superClass -> superClass.fullyQualifiedName().equals(ancestor.fullyQualifiedName())
        || isDescendantOf(superClass, ancestor, visitedTypes));
  }

  private static boolean isLikelySharedBaseClass(ClassInfo candidateClass, SubscriptionContext ctx) {
    ClassDef classDef = candidateClass.classDef();
    String className = classDef.name().name();
    return className.startsWith("Base")
      || className.endsWith("Mixin")
      || isAbstractBaseClass(classDef, candidateClass.classType(), ctx)
      || onlyContainsPlaceholderMethods(classDef, ctx);
  }

  private static boolean isAbstractBaseClass(ClassDef classDef, @Nullable ClassType classType, SubscriptionContext ctx) {
    return classType != null && (!classType.metaClasses().isEmpty()
      || classType.members().stream().anyMatch(member -> "__metaclass__".equals(member.name()))
      || hasAncestorMatching(classDef, ABSTRACT_BASE_CLASS_MATCHER, ctx));
  }

  @Nullable
  private static ClassType classType(ClassDef classDef) {
    return classDef.name().typeV2() instanceof ClassType classType ? classType : null;
  }

  private static boolean hasAncestorMatching(ClassDef classDef, TypeMatcher matcher, SubscriptionContext ctx) {
    var args = classDef.args();
    if (args == null) {
      return false;
    }
    for (var argument : args.arguments()) {
      if (argument instanceof RegularArgument regularArgument && matcher.isTrueFor(regularArgument.expression(), ctx)) {
        return true;
      }
    }
    return false;
  }

  private static boolean onlyContainsPlaceholderMethods(ClassDef classDef, SubscriptionContext ctx) {
    return classDef.body().statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .findAny()
      .filter(functionDef -> classDef.body().statements().stream()
        .filter(FunctionDef.class::isInstance)
        .map(FunctionDef.class::cast)
        .allMatch(method -> isPlaceholderMethod(method, ctx)))
      .isPresent();
  }

  private static boolean isPlaceholderMethod(FunctionDef functionDef, SubscriptionContext ctx) {
    List<Statement> statements = functionDef.body().statements();
    return statements.size() == 1 && isPlaceholderStatement(statements.get(0), ctx);
  }

  private static boolean isPlaceholderStatement(Statement statement, SubscriptionContext ctx) {
    return statement.is(Tree.Kind.PASS_STMT) || isNotImplementedErrorRaise(statement, ctx);
  }

  private static boolean isNotImplementedErrorRaise(Statement statement, SubscriptionContext ctx) {
    if (!(statement instanceof RaiseStatement raiseStatement) || raiseStatement.expressions().size() != 1) {
      return false;
    }
    return isNotImplementedError(raiseStatement.expressions().get(0), ctx);
  }

  private static boolean isNotImplementedError(Expression expression, SubscriptionContext ctx) {
    return NOT_IMPLEMENTED_ERROR_MATCHER.isTrueFor(expression, ctx);
  }

  private static boolean isCollectedTestMethod(FunctionDef functionDef) {
    return UnittestUtils.isTestMethodName(functionDef.name().name());
  }

  private static boolean isTestMethodName(String name) {
    return UnittestUtils.isTestMethodName(name);
  }

  private record ClassInfo(ClassDef classDef, @Nullable ClassType classType, boolean hasCollectedTests, boolean isCandidate) {
  }
}
