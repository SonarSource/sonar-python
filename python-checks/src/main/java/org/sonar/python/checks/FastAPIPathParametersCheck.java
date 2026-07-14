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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8411")
public class FastAPIPathParametersCheck extends PythonSubscriptionCheck {

  private static final String MISSING_PARAM_MESSAGE = "Add path parameter \"%s\" to the function signature.";
  private static final String POSITIONAL_ONLY_MESSAGE = "Path parameter \"%s\" should not be positional-only.";

  private static final List<String> HTTP_METHODS = List.of(
    "get", "post", "put", "delete", "patch", "options", "head", "trace");

  private static final Pattern PATH_PARAM_PATTERN = Pattern.compile("\\{([a-zA-Z_]\\w*)(?::[a-zA-Z_]\\w*)?\\}");

  private static final TypeMatcher FASTAPI_ROUTE_MATCHER = TypeMatchers.any(
    HTTP_METHODS.stream()
      .flatMap(method -> Stream.of(
        TypeMatchers.isType("fastapi.FastAPI." + method),
        TypeMatchers.isType("fastapi.APIRouter." + method)))
  );

  private static final TypeMatcher FASTAPI_DEPENDS_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.param_functions.Depends"),
    TypeMatchers.isType("fastapi.param_functions.Security"));
  private static final TypeMatcher FASTAPI_PATH_MATCHER = TypeMatchers.isType("fastapi.param_functions.Path");
  private static final TypeMatcher TYPING_ANNOTATED_MATCHER = TypeMatchers.isType("typing.Annotated");
  private static final TypeMatcher FASTAPI_APPLICATION_OR_ROUTER_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.FastAPI"),
    TypeMatchers.isType("fastapi.applications.FastAPI"),
    TypeMatchers.isType("fastapi.APIRouter"),
    TypeMatchers.isType("fastapi.routing.APIRouter"));
  private static final Set<String> FASTAPI_APPLICATION_OR_ROUTER_NAMES = Set.of(
    "FastAPI",
    "fastapi.FastAPI",
    "fastapi.applications.FastAPI",
    "APIRouter",
    "fastapi.APIRouter",
    "fastapi.routing.APIRouter");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FastAPIPathParametersCheck::checkFunction);
  }

  private static void checkFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    for (Decorator decorator : functionDef.decorators()) {
      checkDecorator(ctx, decorator, functionDef);
    }
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator, FunctionDef functionDef) {
    Expression expr = decorator.expression();
    if (!(expr instanceof CallExpression callExpr)) {
      return;
    }

    if (!FASTAPI_ROUTE_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }

    Set<String> pathParams = extractPathParameters(callExpr);
    if (pathParams.isEmpty()) {
      return;
    }

    DependencyCollector dependencyCollector = new DependencyCollector(ctx);
    dependencyCollector.collect(functionDef, callExpr);

    reportIssues(ctx, functionDef, pathParams, dependencyCollector);
  }

  private static final class DependencyCollector {
    private final SubscriptionContext ctx;
    private final Set<String> names = new HashSet<>();
    private final Set<String> positionalOnlyParams = new HashSet<>();
    private final Set<FunctionDef> visitedFunctions = new HashSet<>();
    // True when a dynamic, unresolved, or unsupported source could still provide route-visible parameters.
    // In that case missing-parameter issues are not reliable, but positional-only issues are still reported.
    private boolean hasUnresolvedOrUnsupportedParameterSources;

    DependencyCollector(SubscriptionContext ctx) {
      this.ctx = ctx;
    }

    void collect(FunctionDef pathOperationFunction, CallExpression pathOperationDecoratorCall) {
      visitFunction(pathOperationFunction, false, true);
      visitDependenciesArgument(pathOperationDecoratorCall);
      visitApplicationOrRouterDependencies(pathOperationDecoratorCall);
    }

    private void visitApplicationOrRouterDependencies(CallExpression pathOperationDecoratorCall) {
      // FastAPI applies dependencies declared on FastAPI(...) / APIRouter(...) to every route
      // registered on that object. For `router = APIRouter(dependencies=[Depends(dep)])` and
      // `@router.get(...)`, inspect the receiver (`router`) as another dependency entry point.
      if (!(pathOperationDecoratorCall.callee() instanceof QualifiedExpression qualifiedExpression)) {
        return;
      }

      Expression receiver = resolveLocalAliasChain(qualifiedExpression.qualifier());
      if (receiver instanceof CallExpression callExpression && isFastApiApplicationOrRouterCall(callExpression)) {
        visitDependenciesArgument(callExpression);
      }
    }

    private void visitDependenciesArgument(CallExpression callExpression) {
      RegularArgument dependenciesArgument = TreeUtils.argumentByKeyword("dependencies", callExpression.arguments());
      if (dependenciesArgument == null) {
        return;
      }

      Expression dependenciesExpression = resolveLocalAliasChain(dependenciesArgument.expression());
      if (!(dependenciesExpression instanceof ListLiteral || dependenciesExpression instanceof Tuple)) {
        if (dependenciesExpression instanceof Name || dependenciesExpression instanceof QualifiedExpression) {
          markUnresolvedParameterSource();
        } else {
          markUnsupportedParameterSource();
        }
        return;
      }

      Expressions.expressionsFromListOrTuple(dependenciesExpression).forEach(this::visitDependencyListElement);
    }

    private void visitDependencyListElement(Expression element) {
      Expression dependency = resolveLocalAliasChain(element);
      if (dependency instanceof CallExpression callExpression && isDependsCall(callExpression)) {
        visitDependsCall(callExpression, null);
      } else if (dependency instanceof Name || dependency instanceof QualifiedExpression) {
        markUnresolvedParameterSource();
      } else {
        markUnsupportedParameterSource();
      }
    }

    private void visitFunction(FunctionDef functionDef, boolean skipFirstParameter, boolean recordPositionalOnlyParams) {
      if (!visitedFunctions.add(functionDef)) {
        return;
      }

      ParameterList parameterList = functionDef.parameters();
      if (parameterList == null) {
        return;
      }

      visitParameterList(parameterList, skipFirstParameter, recordPositionalOnlyParams);
    }

    private void visitLambda(LambdaExpression lambdaExpression) {
      ParameterList parameterList = lambdaExpression.parameters();
      if (parameterList == null) {
        return;
      }
      visitParameterList(parameterList, false, false);
    }

    private void visitParameterList(ParameterList parameterList, boolean skipFirstParameter, boolean recordPositionalOnlyParams) {
      List<Parameter> parameters = parameterList.nonTuple().stream()
        .skip(skipFirstParameter ? 1 : 0)
        .toList();

      int slashParameterIndex = IntStream.range(0, parameters.size())
        .filter(i -> isSlashParameter(parameters.get(i)))
        .findFirst()
        .orElse(-1);
      for (int i = 0; i < parameters.size(); i++) {
        Parameter parameter = parameters.get(i);
        if (parameter.name() != null) {
          visitParameter(parameter, slashParameterIndex != -1 && i < slashParameterIndex, recordPositionalOnlyParams);
        }
      }
    }

    private void visitParameter(Parameter parameter, boolean isPositionalOnly, boolean recordPositionalOnlyParams) {
      if (parameter.starToken() != null) {
        if ("**".equals(parameter.starToken().value())) {
          markUnresolvedParameterSource();
        }
        return;
      }

      Set<String> pathAliases = new HashSet<>();
      Expression parameterType = visitTypeAnnotation(parameter.typeAnnotation(), pathAliases::add).orElse(null);
      visitDefaultValue(parameter.defaultValue(), parameterType, pathAliases::add);

      Name parameterName = parameter.name();
      if (parameterName != null) {
        if (pathAliases.isEmpty()) {
          names.add(parameterName.name());
          if (isPositionalOnly && recordPositionalOnlyParams) {
            positionalOnlyParams.add(parameterName.name());
          }
        } else {
          names.addAll(pathAliases);
          if (isPositionalOnly && recordPositionalOnlyParams) {
            positionalOnlyParams.addAll(pathAliases);
          }
        }
      }
    }

    private Optional<Expression> visitTypeAnnotation(@Nullable TypeAnnotation typeAnnotation, Consumer<String> pathAliasConsumer) {
      if (typeAnnotation == null) {
        return Optional.empty();
      }
      return visitTypeAnnotationExpression(typeAnnotation.expression(), new HashSet<>(), pathAliasConsumer);
    }

    private Optional<Expression> visitTypeAnnotationExpression(Expression annotationExpression, Set<SymbolV2> visitedAliases, Consumer<String> pathAliasConsumer) {
      // Parenthesized annotations are common in formatted multiline Annotated[...] expressions.
      Expression expression = Expressions.removeParentheses(annotationExpression);

      if (expression instanceof Name name) {
        Optional<Expression> assignedExpression = assignedTypeAliasValue(name, visitedAliases);
        if (assignedExpression.isPresent()) {
          return visitTypeAnnotationExpression(assignedExpression.get(), visitedAliases, pathAliasConsumer);
        }
      }

      if (expression instanceof SubscriptionExpression subscriptionExpression && isAnnotatedObject(subscriptionExpression.object())) {
        // Annotated stores the real type first; FastAPI metadata such as Depends and Path follows.
        List<Expression> subscripts = subscriptionExpression.subscripts().expressions();
        if (subscripts.isEmpty()) {
          return Optional.empty();
        }
        Expression baseType = subscripts.get(0);
        subscripts.stream()
          .skip(1)
          .forEach(metadata -> visitAnnotationMetadata(metadata, baseType, pathAliasConsumer));
        return Optional.of(baseType);
      }

      if (expression instanceof CallExpression) {
        markUnsupportedParameterSource();
      }

      return Optional.of(expression);
    }

    private void visitAnnotationMetadata(Expression metadata, @Nullable Expression parameterType, Consumer<String> pathAliasConsumer) {
      visitDependencyMarkerExpression(metadata, parameterType, pathAliasConsumer);
    }

    private void visitDefaultValue(@Nullable Expression defaultValue, @Nullable Expression parameterType, Consumer<String> pathAliasConsumer) {
      if (defaultValue != null) {
        visitDependencyMarkerExpression(defaultValue, parameterType, pathAliasConsumer);
      }
    }

    // FastAPI reads dependency/path marker objects from both parameter defaults
    // and Annotated[...] metadata, so both contexts share the same handling.
    private void visitDependencyMarkerExpression(Expression expression, @Nullable Expression parameterType, Consumer<String> pathAliasConsumer) {
      Expression resolvedExpression = resolveLocalAliasChain(expression);
      if (resolvedExpression instanceof CallExpression callExpression) {
        if (isDependsCall(callExpression)) {
          visitDependsCall(callExpression, parameterType);
        } else if (isPathCall(callExpression)) {
          visitPathAlias(callExpression, pathAliasConsumer);
        } else {
          markUnsupportedParameterSource();
        }
      } else if (resolvedExpression instanceof Name || resolvedExpression instanceof QualifiedExpression) {
        markUnresolvedParameterSource();
      }
    }

    private void visitDependsCall(CallExpression dependsCall, @Nullable Expression parameterType) {
      Optional<Expression> explicitTarget = TreeUtils.nthArgumentOrKeywordOptional(0, "dependency", dependsCall.arguments())
        .map(RegularArgument::expression);
      Expression target = explicitTarget.orElse(parameterType);
      if (target == null) {
        // Bare Depends() without a parameter type gives FastAPI no callable to inspect.
        return;
      }
      visitDependencyCallable(target);
    }

    private void visitPathAlias(CallExpression pathCall, Consumer<String> pathAliasConsumer) {
      RegularArgument aliasArgument = TreeUtils.argumentByKeyword("alias", pathCall.arguments());
      if (aliasArgument == null) {
        return;
      }
      Optional<String> alias = extractStringValue(aliasArgument.expression());
      if (alias.isPresent()) {
        pathAliasConsumer.accept(alias.get());
      } else {
        markUnresolvedParameterSource();
      }
    }

    private void visitDependencyCallable(Expression expression) {
      // FastAPI accepts any callable dependency. We only inspect callable shapes that can be resolved locally.
      Expression target = resolveLocalAliasChain(expression);

      Optional<FunctionDef> functionDef = getFunctionDef(target);
      if (functionDef.isPresent()) {
        // Depends(get_item) / Depends(SomeClass.get_item): FastAPI inspects the dependency function signature.
        visitFunction(functionDef.get(), false, false);
        return;
      }

      Optional<ClassDef> classDef = getClassDef(target);
      if (classDef.isPresent()) {
        // Depends(ItemQuery) / Depends() with an Annotated class type: FastAPI inspects the constructor signature.
        visitClassConstructor(classDef.get());
        return;
      }

      if (target instanceof CallExpression callExpression) {
        // Depends(ItemChecker()) / checker = ItemChecker(); Depends(checker): FastAPI inspects the instance __call__ signature.
        // FastAPI accepts arbitrary callable results from dependency factory functions, but we cannot inspect them statically.
        Expression callee = resolveLocalAliasChain(callExpression.callee());
        Optional<ClassDef> calleeClassDef = getClassDef(callee);
        if (calleeClassDef.isPresent()) {
          visitCallableInstance(calleeClassDef.get());
          return;
        }
      }

      if (target instanceof LambdaExpression lambdaExpression) {
        // Depends(lambda item_id: ...): FastAPI accepts any callable, including lambdas.
        visitLambda(lambdaExpression);
        return;
      }

      if (target instanceof Name || target instanceof QualifiedExpression) {
        markUnresolvedParameterSource();
      } else {
        markUnsupportedParameterSource();
      }
    }

    private void visitClassConstructor(ClassDef classDef) {
      Optional<FunctionDef> initFunction = topLevelMethod(classDef, "__init__");
      if (initFunction.isPresent()) {
        visitFunction(initFunction.get(), true, false);
      } else if (!classDef.decorators().isEmpty() || classDef.args() != null) {
        // Decorators and base classes can generate or inherit constructor parameters.
        markUnsupportedParameterSource();
      }
    }

    private void visitCallableInstance(ClassDef classDef) {
      Optional<FunctionDef> callFunction = topLevelMethod(classDef, "__call__");
      if (callFunction.isPresent()) {
        visitFunction(callFunction.get(), true, false);
      } else if (!classDef.decorators().isEmpty() || classDef.args() != null) {
        // Decorators and base classes can generate or inherit callable behavior.
        markUnsupportedParameterSource();
      }
    }

    private static Expression resolveLocalAliasChain(Expression expression) {
      // Follow simple local aliases like `route_deps = [Deps(...)]` while preserving the last expression when resolution stops.
      Expression target = Expressions.removeParentheses(expression);
      Set<Name> visitedAliases = new HashSet<>();
      while (target instanceof Name name) {
        Expression assignedValue = Expressions.singleAssignedValue(name, visitedAliases);
        if (assignedValue == null) {
          return target;
        }
        target = Expressions.removeParentheses(assignedValue);
      }
      return target;
    }

    private Optional<Expression> assignedTypeAliasValue(Name name, Set<SymbolV2> visitedAliases) {
      Expression assignedValue = Expressions.singleAssignedValue(name);
      if (assignedValue == null) {
        return Optional.empty();
      }
      SymbolV2 symbol = name.symbolV2();
      if (symbol != null && !visitedAliases.add(symbol)) {
        // Stop chasing type aliases such as A = B; B = A. The unresolved alias chain may hide Annotated metadata.
        markUnresolvedParameterSource();
        return Optional.empty();
      }
      return Optional.of(Expressions.removeParentheses(assignedValue));
    }

    private static Optional<FunctionDef> topLevelMethod(ClassDef classDef, String methodName) {
      return TreeUtils.topLevelFunctionDefs(classDef).stream()
        .filter(functionDef -> methodName.equals(functionDef.name().name()))
        .findFirst();
    }

    private static Optional<FunctionDef> getFunctionDef(Expression expression) {
      return findDeclarationAncestor(expression, UsageV2.Kind.FUNC_DECLARATION, FunctionDef.class, Tree.Kind.FUNCDEF);
    }

    private static Optional<ClassDef> getClassDef(Expression expression) {
      return findDeclarationAncestor(expression, UsageV2.Kind.CLASS_DECLARATION, ClassDef.class, Tree.Kind.CLASSDEF);
    }

    private static <T extends Tree> Optional<T> findDeclarationAncestor(Expression expression, UsageV2.Kind usageKind, Class<T> declarationClass, Tree.Kind declarationKind) {
      // Map a resolved symbol usage back to the local declaration syntax node.
      Name name;
      if (expression instanceof Name n) {
        name = n;
      } else if (expression instanceof QualifiedExpression qe) {
        name = qe.name();
      } else {
        name = null;
      }
      if (name == null) {
        return Optional.empty();
      }
      SymbolV2 symbol = name.symbolV2();
      if (symbol == null) {
        return Optional.empty();
      }
      return symbol.usages().stream()
        .filter(u -> u.kind() == usageKind)
        .map(UsageV2::tree)
        .map(tree -> TreeUtils.firstAncestorOfKind(tree, declarationKind))
        .filter(Objects::nonNull)
        .map(declarationClass::cast)
        .findFirst();
    }

    private boolean isAnnotatedObject(Expression expression) {
      if (TYPING_ANNOTATED_MATCHER.isTrueFor(expression, ctx)) {
        return true;
      }
      return localAliasResolvedName(expression)
        .filter(name -> "Annotated".equals(name) || "typing.Annotated".equals(name) || "typing_extensions.Annotated".equals(name))
        .isPresent();
    }

    private boolean isDependsCall(CallExpression callExpression) {
      return FASTAPI_DEPENDS_MATCHER.isTrueFor(callExpression.callee(), ctx)
        || localAliasResolvedName(callExpression.callee()).filter(name -> "Depends".equals(name) || "Security".equals(name)).isPresent();
    }

    private boolean isPathCall(CallExpression callExpression) {
      return FASTAPI_PATH_MATCHER.isTrueFor(callExpression.callee(), ctx)
        || localAliasResolvedName(callExpression.callee()).filter("Path"::equals).isPresent();
    }

    private boolean isFastApiApplicationOrRouterCall(CallExpression callExpression) {
      return FASTAPI_APPLICATION_OR_ROUTER_MATCHER.isTrueFor(callExpression.callee(), ctx)
        || localAliasResolvedName(callExpression.callee())
          .filter(FASTAPI_APPLICATION_OR_ROUTER_NAMES::contains)
          .isPresent();
    }

    private static Optional<String> localAliasResolvedName(Expression expression) {
      return TreeUtils.stringValueFromNameOrQualifiedExpression(resolveLocalAliasChain(expression));
    }

    private static boolean isSlashParameter(Parameter parameter) {
      return parameter.starToken() != null && "/".equals(parameter.starToken().value());
    }

    private void markUnresolvedParameterSource() {
      // Use when resolving further would likely require dynamic or intermodule data-flow analysis.
      hasUnresolvedOrUnsupportedParameterSources = true;
    }

    private void markUnsupportedParameterSource() {
      // Use when the expression is not one of the FastAPI shapes we explicitly support. Some such
      // expressions may still be valid FastAPI usage that we do not model statically yet, such as
      // unpacking, calls, conditional expressions, or subscriptions producing Depends(...) or
      // Path(...) values;
      // others may be plainly invalid API usage and could theoretically be ignored. Proving that
      // precisely is non-trivial and better left to type checking, so keep the rule conservative
      // and bail out.
      hasUnresolvedOrUnsupportedParameterSources = true;
    }

  }

  private static Set<String> extractPathParameters(CallExpression callExpr) {
    Set<String> pathParams = new HashSet<>();
    String pathString = getPathArgument(callExpr).orElse("");
    Matcher matcher = PATH_PARAM_PATTERN.matcher(pathString);
    while (matcher.find()) {
      pathParams.add(matcher.group(1));
    }
    return pathParams;
  }

  private static Optional<String> getPathArgument(CallExpression callExpr) {
    return TreeUtils.nthArgumentOrKeywordOptional(0, "path", callExpr.arguments())
      .flatMap(arg -> extractStringValue(arg.expression()));
  }

  private static Optional<String> extractStringValue(Expression expression) {
    return Optional.ofNullable(Expressions.extractStringLiteral(expression))
      .map(Expressions::unescape);
  }

  private static void reportIssues(
    SubscriptionContext ctx,
    FunctionDef functionDef,
    Set<String> pathParams,
    DependencyCollector dependencyCollector) {
    pathParams.stream()
      .filter(param -> !dependencyCollector.names.contains(param))
      .filter(param -> !dependencyCollector.hasUnresolvedOrUnsupportedParameterSources)
      .forEach(param -> ctx.addIssue(functionDef.name(), String.format(MISSING_PARAM_MESSAGE, param)));

    pathParams.stream()
      .filter(dependencyCollector.positionalOnlyParams::contains)
      .forEach(param -> ctx.addIssue(functionDef.name(), String.format(POSITIONAL_ONLY_MESSAGE, param)));
  }
}
