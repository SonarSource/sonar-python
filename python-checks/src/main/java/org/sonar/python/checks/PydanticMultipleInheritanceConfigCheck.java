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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S8963")
public class PydanticMultipleInheritanceConfigCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Refactor this Pydantic model to avoid multiple inheritance with conflicting configurations.";

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, PydanticMultipleInheritanceConfigCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (!IS_PYDANTIC_MODEL.isTrueFor(classDef.name(), ctx)) {
      return;
    }

    ArgList argList = classDef.args();
    if (argList == null) {
      return;
    }

    List<Expression> superClasses = collectSuperClasses(argList);
    if (superClasses.size() < 2) {
      return;
    }

    // Exception: the model class itself explicitly defines model_config, making the
    // configuration explicit and predictable (as stated in the rule's Exceptions section).
    if (classDefinesModelConfigLocally(classDef)) {
      return;
    }

    Set<ClassType> visitedClassesDefiningModelConfig = new HashSet<>();
    List<Expression> superClassesWithNewModelConfig = new ArrayList<>();

    for (Expression superClass : superClasses) {
      Set<ClassType> classesDefiningModelConfig = findModelConfigDefiners(superClass);
      boolean introducesNewModelConfigClass = false;
      for (ClassType classDefiningModelConfig : classesDefiningModelConfig) {
        if (visitedClassesDefiningModelConfig.add(classDefiningModelConfig)) {
          introducesNewModelConfigClass = true;
        }
      }
      if (introducesNewModelConfigClass) {
        superClassesWithNewModelConfig.add(superClass);
      }
    }

    reportIfConflictingConfigs(ctx, classDef, visitedClassesDefiningModelConfig, superClassesWithNewModelConfig);
  }

  private static boolean classDefinesModelConfigLocally(ClassDef classDef) {
    PythonType type = classDef.name().typeV2();
    if (type instanceof ClassType classType) {
      return definesModelConfigLocally(classType);
    }
    return false;
  }

  private static List<Expression> collectSuperClasses(ArgList argList) {
    List<Expression> superClasses = new ArrayList<>();
    for (Argument argument : argList.arguments()) {
      if (argument instanceof RegularArgument regularArgument && regularArgument.keywordArgument() == null) {
        superClasses.add(regularArgument.expression());
      }
    }
    return superClasses;
  }

  private static void reportIfConflictingConfigs(
      SubscriptionContext ctx,
      ClassDef classDef,
      Set<ClassType> visitedClassesDefiningModelConfig,
      List<Expression> superClassesWithNewModelConfig) {
    if (superClassesWithNewModelConfig.size() >= 2) {
      PreciseIssue issue = ctx.addIssue(classDef.name(), MESSAGE);
      for (Expression superClass : superClassesWithNewModelConfig) {
        issue.secondary(superClass, "This base class defines \"model_config\".");
      }
    }
  }

  /**
   * Returns the set of ClassType objects in the MRO of {@code superClass} that locally define
   * {@code model_config}.
   */
  private static Set<ClassType> findModelConfigDefiners(Expression superClass) {
    PythonType type = superClass.typeV2();
    if (!(type instanceof ClassType classType)) {
      return Set.of();
    }
    Set<ClassType> classesDefiningModelConfig = new HashSet<>();
    for (ClassType ancestor : classType.mro().orElse(List.of())) {
      if (definesModelConfigLocally(ancestor)) {
        classesDefiningModelConfig.add(ancestor);
      }
    }
    return classesDefiningModelConfig;
  }

  private static boolean definesModelConfigLocally(ClassType classType) {
    return classType.members().stream().map(Member::name).anyMatch("model_config"::equals);
  }
}
