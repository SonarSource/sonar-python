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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SuperProxyType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.types.v2.matchers.AnyTypePredicate;
import org.sonar.python.types.v2.matchers.HasOnlyFunctionDecoratorsSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.IsTypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.sonar.plugins.python.api.tree.Tree.Kind.FUNCDEF;

@Rule(key = "S2638")
public class ChangeMethodContractCheck extends PythonSubscriptionCheck {

  private static final TypePredicate SUPPORTED_DECORATORS = new HasOnlyFunctionDecoratorsSatisfyingPredicate(
    new AnyTypePredicate(List.of(new IsTypePredicate("abc.abstractmethod"), new IsTypePredicate("typing.overload"))));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      String functionName = functionDef.name().name();
      if (functionName.startsWith("__") && functionName.endsWith("__")) {
        // ignore special methods
        return;
      }
      if (!(functionDef.name().typeV2() instanceof FunctionType functionSymbol)) {
        return;
      }
      if (functionSymbol.hasVariadicParameter() || functionSymbol.hasDecorators()) {
        // ignore function declarations with packed params
        return;
      }
      checkMethodContract(ctx, functionSymbol);
    });
  }

  private static void checkMethodContract(SubscriptionContext ctx, FunctionType method) {
    overriddenMethod(method)
      .ifPresent(overriddenMethod -> {
        if (overriddenMethod.hasVariadicParameter() || hasUnsupportedDecorators(ctx, overriddenMethod)) {
          // ignore function declarations with packed params
          return;
        }

        // Optional keyword-only additions cannot displace an inherited positional
        // argument. Ignore them while comparing the existing contract.
        List<ParameterV2> parameters = method.parameters().stream()
          .filter(parameter -> !parameter.isKeywordOnly() || !parameter.hasDefaultValue()
            || overriddenMethod.parameters().stream().anyMatch(parent -> Objects.equals(parent.name(), parameter.name())))
          .toList();
        int paramsDiff = parameters.size() - overriddenMethod.parameters().size();

        if (paramsDiff != 0 && overriddenMethod.parameters().stream().anyMatch(ParameterV2::isKeywordOnly)) {
          reportIssue(ctx, "Change this method signature to accept the same arguments as the method it overrides.", method.definitionLocation().orElse(null), overriddenMethod);
          return;
        }

        if (paramsDiff > 0) {
          reportOnExtraParameters(ctx, method, overriddenMethod);
        } else if (paramsDiff < 0) {
          reportOnMissingParameters(ctx, method, overriddenMethod, parameters.size());
        } else {
          checkDefaultValuesAndParamNames(ctx, parameters, overriddenMethod);
        }
      });
  }

  private static Optional<FunctionType> overriddenMethod(FunctionType method) {
    if (!(method.owner() instanceof ClassType owner)) {
      return Optional.empty();
    }
    return new SuperProxyType(owner).resolveMember(method.name())
      .flatMap(ChangeMethodContractCheck::unambiguousContract);
  }

  static Optional<FunctionType> unambiguousContract(PythonType type) {
    if (type instanceof FunctionType function) {
      return Optional.of(function);
    }
    if (!(type instanceof UnionType union) || union.candidates().isEmpty()
      || !union.candidates().stream().allMatch(FunctionType.class::isInstance)) {
      return Optional.empty();
    }
    List<FunctionType> alternatives = union.candidates().stream().map(FunctionType.class::cast).toList();
    FunctionType first = alternatives.get(0);
    // Defaults and binding modes matter as much as names when selecting a contract.
    return alternatives.stream().allMatch(function -> sameContract(first, function)) ? Optional.of(first) : Optional.empty();
  }

  private static boolean sameContract(FunctionType first, FunctionType other) {
    if (first.hasVariadicParameter() != other.hasVariadicParameter() || first.parameters().size() != other.parameters().size()
      || first.isInstanceMethod() != other.isInstanceMethod() || first.isClassMethod() != other.isClassMethod()
      || first.hasDecorators() != other.hasDecorators()
      || !first.decorators().stream().map(TypeWrapper::type).toList().equals(other.decorators().stream().map(TypeWrapper::type).toList())) {
      return false;
    }
    for (int i = 0; i < first.parameters().size(); i++) {
      ParameterV2 left = first.parameters().get(i);
      ParameterV2 right = other.parameters().get(i);
      if (!Objects.equals(left.name(), right.name()) || left.hasDefaultValue() != right.hasDefaultValue()
        || left.isKeywordOnly() != right.isKeywordOnly() || left.isPositionalOnly() != right.isPositionalOnly()) {
        return false;
      }
    }
    return true;
  }

  private static boolean hasUnsupportedDecorators(SubscriptionContext ctx, FunctionType function) {
    return function.hasDecorators() && !SUPPORTED_DECORATORS.check(function, TypePredicateContext.of(ctx.typeTable())).isTrue();
  }

  private static void reportOnMissingParameters(SubscriptionContext ctx, FunctionType method, FunctionType overriddenMethod, int indexFirstMissingParam) {
    List<ParameterV2> overriddenParams = overriddenMethod.parameters();
    var missingParameters = overriddenParams.subList(indexFirstMissingParam, overriddenParams.size())
      .stream()
      .map(ParameterV2::name)
      .toList();
    if (!missingParameters.isEmpty()) {
//      If at least one parameter missing name - set message as "Add missing N parameters" where N is amount of missing parameters
      reportIssue(ctx, getMissingParametersMessage(missingParameters), method.definitionLocation().orElse(null), overriddenMethod);
    }
  }

  private static String getMissingParametersMessage(List<String> missingParameters) {
    if (missingParameters.stream().anyMatch(name -> name == null || name.isEmpty())) {
      return missingParameters.size() == 1 ? "Add 1 missing parameter." : ("Add " + missingParameters.size() + " missing parameters.");
    }
    return "Add missing parameters " + String.join(" ", missingParameters).trim() + ".";
  }

  private static void reportIssue(SubscriptionContext ctx, String message, @Nullable LocationInFile location, FunctionType overriddenMethod) {
    Optional.ofNullable(location).ifPresent(issueLocation -> {
      LocationInFile secondaryLocation = overriddenMethod.definitionLocation().orElse(null);
      if (secondaryLocation != null) {
        PreciseIssue preciseIssue = ctx.addIssue(issueLocation, message);
        preciseIssue.secondary(secondaryLocation, "Overridden method's definition");
      } else {
        ctx.addIssue(issueLocation, message + " This method overrides " + overriddenMethod.fullyQualifiedName().replaceFirst("^builtins\\.", "") + ".");
      }
    });
  }

  private static void reportOnExtraParameters(SubscriptionContext ctx, FunctionType method, FunctionType overriddenMethod) {
    List<String> parameterNames = method.parameters().stream().map(ParameterV2::name).toList();
    List<String> overriddenParameterNames = overriddenMethod.parameters().stream().map(ParameterV2::name).toList();
    long renamedPositionalParameters = overriddenMethod.parameters().stream()
      .filter(parameter -> !parameter.isKeywordOnly() && !parameterNames.contains(parameter.name()))
      .count();
    for (ParameterV2 parameter : method.parameters()) {
      if (overriddenParameterNames.contains(parameter.name())) {
        continue;
      }
      // Reserve parameters whose names match first, then pair the remaining
      // inherited positional parameters with positional renames in source order.
      if (!parameter.isKeywordOnly() && renamedPositionalParameters > 0) {
        renamedPositionalParameters--;
      } else if (!parameter.hasDefaultValue() && parameter.name() != null && !parameter.name().isEmpty()) {
        reportIssue(ctx, "Remove parameter " + parameter.name() + " or provide default value.", parameter.location(), overriddenMethod);
      }
    }
  }


  private static void checkDefaultValuesAndParamNames(SubscriptionContext ctx, List<ParameterV2> parameters, FunctionType overriddenMethod) {
    Map<String, Integer> mismatchedOverriddenParamPosition = new HashMap<>();
    Map<String, Integer> mismatchedParamPosition = new HashMap<>();

    for (int i = 0; i < overriddenMethod.parameters().size(); i++) {
      ParameterV2 overriddenParam = overriddenMethod.parameters().get(i);
      ParameterV2 parameter = parameters.get(i);
      if (!Objects.equals(overriddenParam.name(), parameter.name())) {
        mismatchedOverriddenParamPosition.put(overriddenParam.name(), i);
        mismatchedParamPosition.put(parameter.name(), i);
      } else {
        checkDefaultValueAndKeywordOnly(ctx, overriddenMethod, overriddenParam, parameter);
      }
    }

    mismatchedParamPosition.forEach((name, index) -> {
      Integer overriddenParamIndex = mismatchedOverriddenParamPosition.get(name);
      ParameterV2 parameter = parameters.get(index);
      if (overriddenParamIndex != null && !parameter.isKeywordOnly()) {
        reportIssue(ctx, "Move parameter " + name + " to position " + overriddenParamIndex + ".", parameter.location(), overriddenMethod);
      }
    });

  }

  private static void checkDefaultValueAndKeywordOnly(SubscriptionContext ctx, FunctionType overriddenMethod, ParameterV2 overriddenParam,
    ParameterV2 parameter) {
    String prefix = "Make parameter " + parameter.name();
    if (overriddenParam.hasDefaultValue() && !parameter.hasDefaultValue()) {
      reportIssue(ctx, "Add a default value to parameter " + parameter.name() + ".", parameter.location(), overriddenMethod);
    }
    if ((!overriddenParam.isKeywordOnly() && !overriddenParam.isPositionalOnly()) && (parameter.isKeywordOnly() || parameter.isPositionalOnly())) {
      reportIssue(ctx, prefix + " keyword-or-positional.", parameter.location(), overriddenMethod);
    }
    if (overriddenParam.isKeywordOnly() && parameter.isPositionalOnly()) {
      reportIssue(ctx, prefix + " keyword only.", parameter.location(), overriddenMethod);
    }
  }
}
