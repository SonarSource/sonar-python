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
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S930")
public class ArgumentNumberCheck extends PythonSubscriptionCheck {

  private static final String FUNCTION_DEFINITION = "Function definition.";
  private static final String NEW_METHOD_NAME = "__new__";
  private static final TypeMatcher SUPER_INSTANCE_MATCHER = TypeMatchers.isObjectOfType("super");
  private static final TypeMatcher BUILTIN_NEW_MATCHER = TypeMatchers.any(TypeMatchers.isType("type.__new__"), TypeMatchers.isType("object.__new__"));
  private static final TypeMatcher ZOPE_INTERFACE_METHOD_MATCHER = TypeMatchers.isFunctionOwnerSatisfying(
    TypeMatchers.isOrExtendsType("zope.interface.Interface"));
  private static final TypeMatcher TYPE_VAR_MATCHER = TypeMatchers.isObjectSatisfying(TypeMatchers.isOrExtendsType("typing.TypeVar"));
  private static final TypeMatcher TYPE_INSTANCE_MATCHER = TypeMatchers.isObjectSatisfying(TypeMatchers.isOrExtendsType("type"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ArgumentNumberCheck::checkCall);
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    var calleeType = callExpression.callee().typeV2();
    var alternatives = calleeType instanceof UnionType union ? union.candidates() : Set.of(calleeType);
    // Library unions also encode overloads and version-specific descriptors. Keep this extension
    // limited to source-defined alternatives until those signatures can be checked reliably.
    List<FunctionType> functions = new ArrayList<>();
    for (var alternative : alternatives) {
      if (!(alternative instanceof FunctionType function) || isException(callExpression, function, calleeType instanceof UnionType)) {
        return;
      }
      functions.add(function);
    }
    if (isCallException(ctx, callExpression, functions)) {
      return;
    }

    ArgumentIssues commonIssues = null;
    for (FunctionType function : functions) {
      var issues = new ArgumentIssues(checkPositionalParameters(callExpression, function));
      checkKeywordArguments(issues, callExpression, function);
      if (commonIssues == null) {
        commonIssues = issues;
      } else {
        commonIssues.retainAll(issues);
      }
      if (commonIssues.isEmpty()) {
        return;
      }
    }
    // A valid alternative, an unknown signature, or conflicting corrections must not produce a finding.
    if (commonIssues != null) {
      reportIssues(ctx, callExpression, functions, commonIssues);
    }
  }

  private record PositionalIssue(long missingArguments, long expectedArguments, boolean hasDefaults) {
    String message(String subject) {
      boolean missing = missingArguments > 0;
      String action = missing ? ("Add " + missingArguments + " missing") : ("Remove " + -missingArguments + " unexpected");
      String bound = "";
      if (hasDefaults) {
        bound = missing ? "at least " : "at most ";
      }
      return action + " arguments; " + subject + " expects " + bound + expectedArguments + " positional arguments.";
    }
  }

  private static class ArgumentIssues {
    @Nullable
    private PositionalIssue positional;
    private final Set<String> missingKeywords = new LinkedHashSet<>();
    private final Set<RegularArgument> unexpectedKeywords = new LinkedHashSet<>();

    ArgumentIssues(@Nullable PositionalIssue positional) {
      this.positional = positional;
    }

    void retainAll(ArgumentIssues other) {
      if (positional != null && other.positional != null
        && positional.missingArguments == other.positional.missingArguments
        && positional.expectedArguments == other.positional.expectedArguments) {
        positional = new PositionalIssue(positional.missingArguments, positional.expectedArguments,
          positional.hasDefaults || other.positional.hasDefaults);
      } else {
        positional = null;
      }
      missingKeywords.retainAll(other.missingKeywords);
      unexpectedKeywords.retainAll(other.unexpectedKeywords);
    }

    boolean isEmpty() {
      return positional == null && missingKeywords.isEmpty() && unexpectedKeywords.isEmpty();
    }
  }

  private static void reportIssues(SubscriptionContext ctx, CallExpression call, List<FunctionType> functions, ArgumentIssues issues) {
    if (issues.positional != null) {
      String subject = functions.stream().map(FunctionType::name).distinct().count() == 1
        ? ("'" + functions.getFirst().name() + "'") : "this callable";
      addIssue(ctx, call.callee(), issues.positional.message(subject), functions);
    }
    for (RegularArgument argument : issues.unexpectedKeywords) {
      addIssue(ctx, argument, "Remove this unexpected named argument '" + argument.keywordArgument().name() + "'.", functions);
    }
    if (!issues.missingKeywords.isEmpty()) {
      String names = issues.missingKeywords.stream().sorted().map(name -> "'" + name + "'").collect(Collectors.joining(" "));
      addIssue(ctx, call.callee(), "Add the missing keyword arguments: " + names, functions);
    }
  }

  @CheckForNull
  private static PositionalIssue checkPositionalParameters(CallExpression callExpression, FunctionType functionType) {
    int self = 0;
    if (isCalledAsInstanceMethod(callExpression, functionType) && functionHasSelfParameter(functionType)) {
      self = 1;
    }
    Map<String, ParameterV2> positionalParamsWithoutDefault = positionalParamsWithoutDefault(functionType);
    long nbPositionalParamsWithDefault = functionType.parameters().stream()
      .filter(parameterName -> !parameterName.isKeywordOnly() && parameterName.hasDefaultValue())
      .count();

    List<RegularArgument> arguments = callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .toList();
    long nbPositionalArgs = arguments.stream().filter(a -> a.keywordArgument() == null).count();
    long nbNonKeywordOnlyPassedWithKeyword = arguments.stream()
      .map(RegularArgument::keywordArgument)
      .filter(k -> k != null && positionalParamsWithoutDefault.containsKey(k.name()) && !positionalParamsWithoutDefault.get(k.name()).isPositionalOnly())
      .count();

    long minimumPositionalArgs = positionalParamsWithoutDefault.size();
    long nbMissingArgs = minimumPositionalArgs - nbPositionalArgs - self - nbNonKeywordOnlyPassedWithKeyword;
    if (nbMissingArgs > 0) {
      return new PositionalIssue(nbMissingArgs, minimumPositionalArgs - self, nbPositionalParamsWithDefault > 0);
    } else if (nbMissingArgs + nbPositionalParamsWithDefault + nbNonKeywordOnlyPassedWithKeyword < 0) {
      return new PositionalIssue(nbMissingArgs + nbPositionalParamsWithDefault,
        minimumPositionalArgs - self + nbPositionalParamsWithDefault, nbPositionalParamsWithDefault > 0);
    }
    return null;
  }

  private static boolean isCalledAsInstanceMethod(CallExpression callExpression, FunctionType functionType) {
    return functionType.isInstanceMethod()
      && callExpression.callee() instanceof QualifiedExpression qualifiedExpression
      && qualifiedExpression.qualifier().typeV2() instanceof ObjectType;
  }

  private static boolean functionHasSelfParameter(FunctionType functionType) {
    return !functionType.parameters().isEmpty();
  }

  private static Map<String, ParameterV2> positionalParamsWithoutDefault(FunctionType functionType) {
    int unnamedIndex = 0;
    Map<String, ParameterV2> result = new HashMap<>();
    for (ParameterV2 parameter : functionType.parameters()) {
      if (!parameter.isKeywordOnly() && !parameter.hasDefaultValue()) {
        String name = parameter.name();
        if (name == null || name.isEmpty()) {
          result.put("!unnamed" + unnamedIndex, parameter);
          unnamedIndex++;
        } else {
          result.put(parameter.name(), parameter);
        }
      }
    }
    return result;
  }

  private static boolean isException(CallExpression callExpression, FunctionType functionType, boolean isUnion) {
    // Unannotated local functions can still have STUB origin; their source location identifies them reliably.
    return (isUnion && functionType.definitionLocation().isEmpty())
      || functionType.hasDecorators()
      || functionType.hasVariadicParameter()
      || isCalledAsBoundInstanceMethod(callExpression, functionType);
  }

  private static boolean isCallException(SubscriptionContext ctx, CallExpression callExpression, List<FunctionType> functions) {
    return callExpression.arguments().stream().anyMatch(argument -> argument.is(Tree.Kind.UNPACKING_EXPR))
      || ZOPE_INTERFACE_METHOD_MATCHER.isTrueFor(callExpression.callee(), ctx)
      || isSuperCall(ctx, callExpression)
      || (functions.stream().anyMatch(function -> NEW_METHOD_NAME.equals(function.name())) && isAliasedSuperNewCall(ctx, callExpression))
      || isReceiverTypeVar(ctx, callExpression)
      || isReceiverTypeInstance(ctx, callExpression);
  }

  private static boolean isAliasedSuperNewCall(SubscriptionContext ctx, CallExpression callExpression) {
    if (!(callExpression.callee() instanceof Name name)) {
      return false;
    }
    // __new__ is implicitly static, so an alias of super().__new__ legitimately
    // receives cls explicitly. The alias prevents isSuperCall from recognizing it.
    var values = ctx.valuesAtLocation(name);
    if (values.isEmpty()) {
      // Without reaching definitions (for example in a try scope), a builtin allocator may
      // be a super().__new__ alias whose implicit static binding is not modeled reliably.
      return BUILTIN_NEW_MATCHER.isTrueFor(name, ctx);
    }
    return values.stream()
      .anyMatch(value -> value instanceof QualifiedExpression qualifiedExpression
        && NEW_METHOD_NAME.equals(qualifiedExpression.name().name())
        && SUPER_INSTANCE_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx));
  }

  private static boolean isCalledAsBoundInstanceMethod(CallExpression callExpression, FunctionType functionType) {
    return functionType.isInstanceMethod() && !callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR);
  }

  private static boolean isReceiverTypeVar(SubscriptionContext ctx, CallExpression callExpression) {
    if (callExpression.callee() instanceof QualifiedExpression qualifiedExpression) {
      return TYPE_VAR_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static boolean isReceiverTypeInstance(SubscriptionContext ctx, CallExpression callExpression) {
    // see SONARPY-3591
    if (callExpression.callee() instanceof QualifiedExpression qualifiedExpression) {
      return TYPE_INSTANCE_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static boolean isSuperCall(SubscriptionContext ctx, CallExpression callExpression) {
    if (callExpression.callee() instanceof QualifiedExpression qualifiedExpression) {
      return SUPER_INSTANCE_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static void addIssue(SubscriptionContext ctx, Tree tree, String message, List<FunctionType> functions) {
    PreciseIssue issue = ctx.addIssue(tree, message);
    functions.forEach(function -> function.definitionLocation().ifPresent(location -> issue.secondary(location, FUNCTION_DEFINITION)));
  }

  private static void checkKeywordArguments(ArgumentIssues issues, CallExpression callExpression, FunctionType functionType) {
    List<ParameterV2> parameters = functionType.parameters();
    parameters.stream()
      .filter(parameter -> parameter.isKeywordOnly() && !parameter.hasDefaultValue())
      .map(ParameterV2::name)
      .forEach(issues.missingKeywords::add);

    for (Argument argument : callExpression.arguments()) {
      RegularArgument arg = (RegularArgument) argument;
      Name keyword = arg.keywordArgument();
      if (keyword != null) {
        if (parameters.stream().noneMatch(parameter -> keyword.name().equals(parameter.name()) && !parameter.isPositionalOnly())) {
          issues.unexpectedKeywords.add(arg);
        } else {
          issues.missingKeywords.remove(keyword.name());
        }
      }
    }
  }
}
