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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key="S5724")
public class PropertyAccessorParameterCountCheck extends PythonSubscriptionCheck {

  private static class PropertyAccessorTriple {
    private Optional<FunctionDef> getter = Optional.empty();
    private Optional<FunctionDef> setter = Optional.empty();
    private Optional<FunctionDef> deleter = Optional.empty();
  }

  private static class CollectPropertiesVisitor extends BaseTreeVisitor {
    private Map<String, PropertyAccessorTriple> decoratorStyleProperties = new HashMap<>();
    private List<PropertyAccessorTriple> propertyCallStyleProperties = new ArrayList<>();

    private static Optional<FunctionDef> findFunctionDefFromArgument(Expression argument) {
      if (!(argument instanceof HasSymbol)) {
        return Optional.empty();
      }

      Symbol symbol = ((HasSymbol) argument).symbol();
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

      PropertyAccessorTriple triple = new PropertyAccessorTriple();
      List<Argument> arguments = pyCallExpressionTree.arguments();
      for (int i = 0; i < arguments.size(); ++i) {
        Argument currentArg = arguments.get(i);
        if (!currentArg.is(Tree.Kind.REGULAR_ARGUMENT)) {
          continue;
        }

        Expression argExpr = ((RegularArgument) currentArg).expression();
        switch (i) {
          case 0: triple.getter = findFunctionDefFromArgument(argExpr); break;
          case 1: triple.setter = findFunctionDefFromArgument(argExpr); break;
          case 2: triple.deleter = findFunctionDefFromArgument(argExpr); break;
          default:
            break;
        }
      }

      propertyCallStyleProperties.add(triple);
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // First check if the function definition has a @property decorator
      boolean hasPropertyDecorator = pyFunctionDefTree.decorators().stream()
        .map(decorator -> decorator.name().names())
        .anyMatch(names -> names.size() == 1 && "property".equals(names.get(0).name()));

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

      Optional<List<Name>> setterOrDeleterDecoratorNames = pyFunctionDefTree.decorators().stream()
        .map(decorator -> decorator.name().names())
        .filter(names -> names.size() == 2 && ("setter".equals(names.get(1).name()) || "deleter".equals(names.get(1).name())))
        .findFirst();

      setterOrDeleterDecoratorNames.ifPresent(names -> {
        String propertyName = names.get(0).name();
        String accessor = names.get(1).name();
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
        .collect(Collectors.toList());
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
        triple.getter.ifPresent(functionDef -> {
          int actualParams = TreeUtils.positionalParameters(functionDef).size();
          if (actualParams > 1) {
            ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), String.format(
              "Remove %d parameters; property getter methods receive only \"self\".", actualParams - 1));
          }
        });
        triple.setter.ifPresent(functionDef -> {
          int actualParams = TreeUtils.positionalParameters(functionDef).size();
          if (actualParams > 2) {
            ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), String.format(
              "Remove %d parameters; property setter methods receive \"self\" and a value.", actualParams - 2));
          } else if (actualParams < 2) {
            ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), "Add the value parameter; property setter methods receive \"self\" and a value.");
          }
        });
        triple.deleter.ifPresent(functionDef -> {
          int actualParams = TreeUtils.positionalParameters(functionDef).size();
          if (actualParams > 1) {
            ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), String.format(
              "Remove %d parameters; property deleter methods receive only \"self\".", actualParams - 1));
          }
        });
      }
    });
  }
}
