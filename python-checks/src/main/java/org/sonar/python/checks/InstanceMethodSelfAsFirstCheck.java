/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5720")
public class InstanceMethodSelfAsFirstCheck extends PythonSubscriptionCheck {

  // We allow "_" as sometimes it is conventionally used to signal that self won't be used.
  private static final List<String> ALLOWED_NAMES = Arrays.asList("self", "_");
  private static final List<String> ALLOWED_NAMES_IN_METACLASSES = Arrays.asList("cls", "mcs");
  private static final List<String> EXCEPTIONS = Arrays.asList("__init_subclass__", "__class_getitem__", "__new__");

  private static final String DEFAULT_IGNORED_DECORATORS = "abstractmethod";
  private List<String> decoratorsToExclude;

  @RuleProperty(
    key = "ignoredDecorators",
    description = "Comma-separated list of decorators which will disable this rule.",
    defaultValue = DEFAULT_IGNORED_DECORATORS)
  public String ignoredDecorators = DEFAULT_IGNORED_DECORATORS;

  private List<String> getExcludedDecorators() {
    if (decoratorsToExclude == null) {
      decoratorsToExclude = new ArrayList<>();
      decoratorsToExclude.add("staticmethod");
      decoratorsToExclude.add("classmethod");
      decoratorsToExclude.addAll(Arrays.asList(this.ignoredDecorators.split(",")));
    }

    return decoratorsToExclude;
  }

  private boolean isNonInstanceMethodDecorator(Decorator decorator) {
    String fqn = TreeUtils.decoratorNameFromExpression(decorator.expression());
    return fqn != null && this.getExcludedDecorators().stream().anyMatch(fqn::contains);
  }

  private static boolean isExceptionalUsageInClassBody(Usage usage, ClassDef parentClass) {
    if (usage.kind() != Usage.Kind.FUNC_DECLARATION) {
      Tree ancestor = TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF);
      return isUsedAsDecorator(ancestor, usage.tree()) || parentClass.equals(ancestor);
    }
    return false;
  }

  private static boolean isUsedAsDecorator(@Nullable Tree tree, Tree usageTree) {
    if (tree instanceof FunctionDef functionDef) {
      return functionDef.decorators().stream()
        .map(Decorator::expression)
        .map(expression -> expression.is(Tree.Kind.CALL_EXPR) ? ((CallExpression) expression).callee() : expression)
        .anyMatch(expression -> expression.equals(usageTree));
    }
    return false;
  }

  private boolean isRelevantMethod(ClassDef classDef, ClassSymbol classSymbol, FunctionDef functionDef) {
    // Skip some known special methods
    if (EXCEPTIONS.contains(functionDef.name().name())) {
      return false;
    }

    // Skip if the class has a ignored decorator
    if (functionDef.decorators().stream().anyMatch(this::isNonInstanceMethodDecorator)) {
      return false;
    }

    FunctionSymbol functionSymbol = TreeUtils.getFunctionSymbolFromDef(functionDef);
    if (functionSymbol == null || functionSymbol.usages().stream().anyMatch(usage -> isExceptionalUsageInClassBody(usage, classDef))) {
      return false;
    }

    return !classSymbol.isOrExtends("zope.interface.Interface");
  }

  private static boolean isValidExceptionForCls(FunctionDef functionDef, String name, boolean mightBeMetaclass) {
    return ALLOWED_NAMES_IN_METACLASSES.contains(name) && (mightBeMetaclass || !functionDef.decorators().isEmpty());
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      ClassSymbol classSymbol = TreeUtils.getClassSymbolFromDef(classDef);

      // Do not raise on nested classes - they might use a different name for "self" in order to avoid confusion.
      if (classSymbol == null || TreeUtils.firstAncestorOfKind(classDef, Tree.Kind.CLASSDEF) != null) {
        return;
      }

      // We consider that a class MIGHT be a metaclass, either if the class is decorated, inherits from "type",
      // "typing.Protocol" or has an unresolved type hiearchy. In this case we shall also accept "cls" or "mcs"
      // as the name of the first parameter.
      boolean mightBeMetaclass = !classDef.decorators().isEmpty()
        || classSymbol.isOrExtends("type")
        || classSymbol.isOrExtends("typing.Protocol")
        || classSymbol.hasUnresolvedTypeHierarchy();

      TreeUtils.topLevelFunctionDefs(classDef).forEach(functionDef -> {
        List<Parameter> parameters = TreeUtils.positionalParameters(functionDef);
        if (parameters.isEmpty()) {
          return;
        }

        Parameter first = parameters.get(0);
        if (first.starToken() != null) {
          return;
        }

        Optional.ofNullable(first.name())
          .map(Name::name)
          .ifPresent(name -> {
            if (!ALLOWED_NAMES.contains(name)
              && isRelevantMethod(classDef, classSymbol, functionDef)
              && !isValidExceptionForCls(functionDef, name, mightBeMetaclass)) {
              ctx.addIssue(first, String.format("Rename \"%s\" to \"self\" or add the missing \"self\" parameter.", name));
            }
          });
      });
    });
  }
}
