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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FUNCDEF;

@Rule(key = "S2638")
public class ChangeMethodContractCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      String functionName = functionDef.name().name();
      if (functionName.startsWith("__") && functionName.endsWith("__")) {
        // ignore special methods
        return;
      }
      Symbol symbol = functionDef.name().symbol();
      if (symbol == null || symbol.kind() != FUNCTION) {
        return;
      }
      FunctionSymbol functionSymbol = (FunctionSymbol) symbol;
      if (functionSymbol.hasVariadicParameter() || functionSymbol.hasDecorators()) {
        // ignore function declarations with packed params
        return;
      }
      checkMethodContract(ctx, functionDef, functionSymbol);
    });
  }

  private static void checkMethodContract(SubscriptionContext ctx, FunctionDef functionDef, FunctionSymbol functionSymbol) {
    getOverriddenMethod(functionSymbol).ifPresent(overriddenMethod -> {
      if (overriddenMethod.hasVariadicParameter() || overriddenMethod.hasDecorators()) {
        // ignore function declarations with packed params
        return;
      }
      int paramsDiff = functionSymbol.parameters().size() - overriddenMethod.parameters().size();
      if (paramsDiff > 0) {
        reportOnExtraParameters(ctx, overriddenMethod.parameters().size(), functionDef, overriddenMethod.definitionLocation());
      } else if (paramsDiff < 0) {
        reportOnMissingParameters(ctx, overriddenMethod, functionDef);
      } else {
        checkDefaultValuesAndParamNames(ctx, overriddenMethod, functionSymbol, functionDef);
      }
    });
  }

  private static Optional<FunctionSymbol> getOverriddenMethod(FunctionSymbol functionSymbol) {
    Symbol owner = ((FunctionSymbolImpl) functionSymbol).owner();
    if (owner == null || owner.kind() != CLASS) {
      return Optional.empty();
    }
    ClassSymbol classSymbol = (ClassSymbol) owner;
    if (classSymbol.superClasses().isEmpty()) {
      return Optional.empty();
    }
    for (Symbol superClass : classSymbol.superClasses()) {
      if (superClass.kind() == CLASS) {
        Optional<FunctionSymbol> overriddenSymbol = ((ClassSymbol) superClass).resolveMember(functionSymbol.name())
                .filter(symbol -> symbol.kind() == FUNCTION)
                .map(FunctionSymbol.class::cast);
        if (overriddenSymbol.isPresent()) {
          return overriddenSymbol;
        }
      }
    }
    return Optional.empty();
  }


  private static void reportOnMissingParameters(SubscriptionContext ctx, FunctionSymbol overriddenMethod, FunctionDef functionDef) {
    int indexFirstMissingParam = TreeUtils.nonTupleParameters(functionDef).size();
    List<FunctionSymbol.Parameter> overridenParams = overriddenMethod.parameters();
    String missingParameters = overridenParams.subList(indexFirstMissingParam, overridenParams.size())
      .stream()
      .filter(parameter -> !parameter.hasDefaultValue())
      .map(FunctionSymbol.Parameter::name)
      .collect(Collectors.joining(" "));
    if (!missingParameters.isEmpty()) {
      PreciseIssue preciseIssue = ctx.addIssue(functionDef.name(), "Add missing parameters " + missingParameters.trim() + ".");
      addDefinitionSecondaryLocation(preciseIssue, overriddenMethod.definitionLocation());
    }
  }

  private static void addDefinitionSecondaryLocation(PreciseIssue preciseIssue, @Nullable LocationInFile overriddenMethodLocation) {
    if (overriddenMethodLocation != null) {
      preciseIssue.secondary(overriddenMethodLocation, null);
    }
  }

  private static void reportOnExtraParameters(SubscriptionContext ctx, int indexFirstExtraParams, FunctionDef functionDef, @Nullable LocationInFile definitionLocation) {
    List<Parameter> parameters = TreeUtils.nonTupleParameters(functionDef);
    for (int i = indexFirstExtraParams; i < parameters.size(); i++) {
      Parameter parameter = parameters.get(i);
      Name name = parameter.name();
      if (parameter.defaultValue() == null && name != null) {
        PreciseIssue preciseIssue = ctx.addIssue(parameter, "Remove parameter " + name.name() + " or provide default value.");
        addDefinitionSecondaryLocation(preciseIssue, definitionLocation);
      }
    }
  }


  private static void checkDefaultValuesAndParamNames(SubscriptionContext ctx, FunctionSymbol overriddenMethod, FunctionSymbol functionSymbol, FunctionDef functionDef) {
    Map<String, Integer> mismatchedOverriddenParamPosition = new HashMap<>();
    Map<String, Integer> mismatchedParamPosition = new HashMap<>();

    List<Parameter> parameters = TreeUtils.nonTupleParameters(functionDef);
    for (int i = 0; i < overriddenMethod.parameters().size(); i++) {
      FunctionSymbol.Parameter overriddenParam = overriddenMethod.parameters().get(i);
      FunctionSymbol.Parameter parameter = functionSymbol.parameters().get(i);
      if (!Objects.equals(overriddenParam.name(), parameter.name())) {
        mismatchedOverriddenParamPosition.put(overriddenParam.name(), i);
        mismatchedParamPosition.put(parameter.name(), i);
      }
      if (overriddenParam.hasDefaultValue() && !parameter.hasDefaultValue()) {
        PreciseIssue preciseIssue = ctx.addIssue(parameters.get(i), "Add a default value to parameter " + parameter.name() + ".");
        addDefinitionSecondaryLocation(preciseIssue, overriddenMethod.definitionLocation());
      }
    }

    mismatchedParamPosition.forEach((name, index) -> {
      Integer overriddenParamIndex = mismatchedOverriddenParamPosition.get(name);
      AnyParameter parameter = parameters.get(index);
      PreciseIssue preciseIssue;
      if (overriddenParamIndex != null) {
        preciseIssue = ctx.addIssue(parameter, "Move parameter " + name + " to position " + overriddenParamIndex + ".");
      } else {
        preciseIssue = ctx.addIssue(parameter, "Rename this parameter as \"" + overriddenMethod.parameters().get(index).name() +  "\".");
      }
      addDefinitionSecondaryLocation(preciseIssue, overriddenMethod.definitionLocation());
    });

  }
}
