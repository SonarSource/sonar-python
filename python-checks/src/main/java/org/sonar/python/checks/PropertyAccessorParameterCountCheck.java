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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5724")
public class PropertyAccessorParameterCountCheck extends PythonSubscriptionCheck {

  private static class PropertyAccessorTriple {
    private Optional<FunctionDef> getter = Optional.empty();
    private Optional<FunctionDef> setter = Optional.empty();
    private Optional<FunctionDef> deleter = Optional.empty();
  }

  private static class CollectPropertiesVisitor extends BaseTreeVisitor {
    private Map<String, PropertyAccessorTriple> decoratorStyleProperties = new HashMap<>();
    private List<PropertyAccessorTriple> propertyCallStyleProperties = new ArrayList<>();

    private static Optional<FunctionDef> findFunctionDefFromArgument(List<RegularArgument> arguments, int position) {
      if (arguments.size() <= position) {
        return Optional.empty();
      }

      RegularArgument argument = arguments.get(position);
      Expression argumentExpr = argument.expression();
      if (!(argumentExpr instanceof HasSymbol)) {
        return Optional.empty();
      }

      Symbol symbol = ((HasSymbol) argumentExpr).symbol();
      if (symbol == null) {
        return Optional.empty();
      }

      return symbol.usages().stream()
        .filter(usage -> usage.kind() == Usage.Kind.FUNC_DECLARATION)
        .map(usage -> usage.tree().parent())
        .filter(tree -> tree.is(Tree.Kind.FUNCDEF))
        .map(FunctionDef.class::cast)
        .findFirst();
    }

    @Override
    public void visitCallExpression(CallExpression pyCallExpressionTree) {
      Symbol callee = pyCallExpressionTree.calleeSymbol();
      if (callee == null || !"property".equals(callee.name())) {
        return;
      }

      List<Argument> argumentList = pyCallExpressionTree.arguments();
      List<RegularArgument> regularArguments = argumentList.stream()
        .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .toList();

      // Do not bother with tuple arguments and keyword arguments
      if (regularArguments.size() != argumentList.size() || regularArguments.stream().anyMatch(arg -> arg.keywordArgument() != null)) {
        return;
      }

      PropertyAccessorTriple triple = new PropertyAccessorTriple();
      triple.getter = findFunctionDefFromArgument(regularArguments, 0);
      triple.setter = findFunctionDefFromArgument(regularArguments, 1);
      triple.deleter = findFunctionDefFromArgument(regularArguments, 2);

      propertyCallStyleProperties.add(triple);
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // First check if the function definition has a @property decorator
      boolean hasPropertyDecorator = pyFunctionDefTree.decorators().stream()
        .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
        .anyMatch("property"::equals);

      if (hasPropertyDecorator) {
        decoratorStyleProperties.compute(pyFunctionDefTree.name().name(), (key, value) -> {
          if (value == null) {
            value = new PropertyAccessorTriple();
          }
          value.getter = Optional.of(pyFunctionDefTree);
          return value;
        });
        return;
      }

      Optional<String[]> setterOrDeleterDecoratorNames = pyFunctionDefTree.decorators().stream()
        .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
        .filter(Objects::nonNull)
        .map(decoratorName -> decoratorName.split("\\."))
        .filter(names -> names.length == 2 && ("setter".equals(names[1]) || "deleter".equals(names[1])))
        .findFirst();

      setterOrDeleterDecoratorNames.ifPresent(names -> {
        String propertyName = names[0];
        String accessor = names[1];
        decoratorStyleProperties.compute(propertyName, (key, value) -> {
          if (value == null) {
            // This should not happen in a valid python code (e.g. @foo.setter cannot be used before declaring foo), but be defensive.
            value = new PropertyAccessorTriple();
          }

          if ("setter".equals(accessor)) {
            value.setter = Optional.of(pyFunctionDefTree);
          } else if ("deleter".equals(accessor)) {
            value.deleter = Optional.of(pyFunctionDefTree);
          }

          return value;
        });
      });
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      // Do not descend into nested classes
    }

    public List<PropertyAccessorTriple> propertyAccessors() {
      return Stream.concat(this.propertyCallStyleProperties.stream(), this.decoratorStyleProperties.values().stream())
        .toList();
    }
  }

  private static long countRequiredParameters(FunctionDef functionDef) {
    ParameterList parameterList = functionDef.parameters();
    if (parameterList == null) {
      return 0;
    }

    return parameterList.all().stream()
      .filter(p -> p.is(Tree.Kind.TUPLE_PARAMETER)
        || (p.is(Tree.Kind.PARAMETER) && ((Parameter) p).defaultValue() == null))
      .count();
  }

  private static void checkOnlySelfParameter(SubscriptionContext ctx, FunctionDef functionDef, String messageTemplate) {
    long actualParams = countRequiredParameters(functionDef);
    if (actualParams > 1) {
      ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), String.format(messageTemplate, actualParams - 1));
    }
  }

  private static void checkSetterParameters(SubscriptionContext ctx, FunctionDef functionDef) {
    long requiredParameters = countRequiredParameters(functionDef);

    if (requiredParameters > 2) {
      ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), String.format(
        "Remove %d parameters; property setter methods receive \"self\" and a value.", requiredParameters - 2));
    } else if (requiredParameters < 2 && TreeUtils.positionalParameters(functionDef).size() < 2) {
      ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), "Add the value parameter; property setter methods receive \"self\" and a value.");
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();

      CollectPropertiesVisitor visitor = new CollectPropertiesVisitor();
      classDef.body().accept(visitor);

      List<PropertyAccessorTriple> propertyAccessors = visitor.propertyAccessors();
      for (PropertyAccessorTriple triple : propertyAccessors) {
        triple.getter.ifPresent(functionDef -> checkOnlySelfParameter(ctx, functionDef, "Remove %d parameters; property getter methods receive only \"self\"."));
        triple.setter.ifPresent(functionDef -> checkSetterParameters(ctx, functionDef));
        triple.deleter.ifPresent(functionDef -> checkOnlySelfParameter(ctx, functionDef, "Remove %d parameters; property deleter methods receive only \"self\"."));
      }
    });
  }
}
