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
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8415")
public class FastAPIHTTPExceptionDocumentedCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Document this HTTPException with status code %d in the \"responses\" parameter.";

  private static final String FASTAPI_MODULE = "fastapi.applications.FastAPI";
  private static final String API_ROUTER_MODULE = "fastapi.routing.APIRouter";
  private static final Set<String> ROUTES = Set.of(
    "get", "post", "put", "delete", "patch", "options", "head", "trace");

  private static final TypeMatcher FASTAPI_ROUTE_MATCHER = TypeMatchers.any(
    Stream.concat(
      ROUTES.stream().map(methodName -> TypeMatchers.isType(FASTAPI_MODULE + "." + methodName)),
      ROUTES.stream().map(methodName -> TypeMatchers.isType(API_ROUTER_MODULE + "." + methodName))));

  private static final TypeMatcher FASTAPI_APP_OR_ROUTER_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.FastAPI"),
    TypeMatchers.isType(FASTAPI_MODULE),
    TypeMatchers.isType("fastapi.APIRouter"),
    TypeMatchers.isType(API_ROUTER_MODULE));

  private static final TypeMatcher INCLUDE_ROUTER_MATCHER = TypeMatchers.any(
    TypeMatchers.isType(FASTAPI_MODULE + ".include_router"),
    TypeMatchers.isType(API_ROUTER_MODULE + ".include_router"));

  private static final TypeMatcher IS_HTTP_EXCEPTION = TypeMatchers.any(
    TypeMatchers.isType("fastapi.exceptions.HTTPException"),
    TypeMatchers.isType("fastapi.HTTPException"));

  private static final int MAX_FUNCTION_CALLS = 100;

  private final Set<Expression> reportedHttpExceptionCalls = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::init);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkFunctionDef);
  }

  private void init(SubscriptionContext ctx) {
    reportedHttpExceptionCalls.clear();
  }

  private void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    DecoratorAnalysisResult analysisResult = new DecoratorAnalysis(ctx, functionDef).analyze();

    if (!analysisResult.isFastApiEndpoint() || !analysisResult.canAnalyzeResponses()) {
      return;
    }

    List<RaiseInfo> httpExceptions = new RaiseInfoCollector(ctx, functionDef).collect();

    reportUndocumentedExceptions(ctx, httpExceptions, analysisResult.documentedStatusCodes);
  }

  private static class DecoratorAnalysis {
    private SubscriptionContext ctx;
    private FunctionDef functionDef;

    private boolean canAnalyzeResponses = true;

    public DecoratorAnalysis(SubscriptionContext ctx, FunctionDef functionDef) {
      this.ctx = ctx;
      this.functionDef = functionDef;
    }

    public DecoratorAnalysisResult analyze() {
      List<CallExpression> fastApiRouteDecorators = functionDef.decorators().stream()
        .map(Decorator::expression)
        .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class))
        .filter(callExpr -> isFastApiRouteDecorator(callExpr, ctx))
        .filter(callExpr -> !isExcludedFromSchema(callExpr))
        .toList();

      Set<Integer> documentedStatusCodes = new HashSet<>();
      boolean isFastApiEndpoint = !fastApiRouteDecorators.isEmpty();
      canAnalyzeResponses = true;

      for (CallExpression fastApiRouteDecorator : fastApiRouteDecorators) {
        documentedStatusCodes.addAll(processDecorator(fastApiRouteDecorator));
      }

      return new DecoratorAnalysisResult(isFastApiEndpoint, canAnalyzeResponses, documentedStatusCodes);
    }

    private static boolean isFastApiRouteDecorator(CallExpression callExpr, SubscriptionContext ctx) {
      return FASTAPI_ROUTE_MATCHER.isTrueFor(callExpr.callee(), ctx);
    }

    private boolean isExcludedFromSchema(CallExpression decoratorCall) {
      if (hasFalsyIncludeInSchema(decoratorCall)) {
        return true;
      }
      // FastAPI combines the route flag with the flags of the FastAPI()/APIRouter() the route is registered on
      // with a logical AND, so a falsy flag at any level keeps the route out of the OpenAPI schema.
      if (!(decoratorCall.callee() instanceof QualifiedExpression qualifiedExpression)) {
        return false;
      }
      Expression receiver = Expressions.removeParentheses(qualifiedExpression.qualifier());
      return isReceiverConstructedExcluded(receiver) || isExcludedViaIncludeRouter(receiver);
    }

    private static boolean hasFalsyIncludeInSchema(CallExpression callExpr) {
      RegularArgument includeInSchemaArg = TreeUtils.argumentByKeyword("include_in_schema", callExpr.arguments());
      return includeInSchemaArg != null && Expressions.isFalsy(includeInSchemaArg.expression());
    }

    /**
     * Handles {@code router = APIRouter(include_in_schema=False)} / {@code app = FastAPI(include_in_schema=False)}.
     * A router imported from another module resolves to a bare Name, so it is left alone: its construction
     * arguments are not reachable from a single-file analysis.
     */
    private boolean isReceiverConstructedExcluded(Expression receiver) {
      Expression target = resolveLocalAliasChain(receiver);
      return target instanceof CallExpression constructorCall
        && FASTAPI_APP_OR_ROUTER_MATCHER.isTrueFor(constructorCall.callee(), ctx)
        && hasFalsyIncludeInSchema(constructorCall);
    }

    /**
     * Handles {@code app.include_router(router, include_in_schema=False)}. If the same router is also included
     * elsewhere without the flag, one copy would still reach the schema, but this rule favours staying silent
     * over reporting a route that cannot be proven public.
     */
    private boolean isExcludedViaIncludeRouter(Expression receiver) {
      if (!(receiver instanceof Name routerName)) {
        return false;
      }
      return Optional.ofNullable(routerName.symbolV2())
        .map(SymbolV2::usages)
        .stream()
        .flatMap(List::stream)
        .filter(usage -> usage.kind() == UsageV2.Kind.OTHER)
        .anyMatch(this::isRouterArgumentOfExcludingInclude);
    }

    private boolean isRouterArgumentOfExcludingInclude(UsageV2 usage) {
      RegularArgument argument = TreeUtils.firstAncestorOfClass(usage.tree(), RegularArgument.class);
      if (argument == null) {
        return false;
      }
      return Optional.ofNullable(TreeUtils.firstAncestorOfClass(argument, CallExpression.class))
        .filter(parentCall -> INCLUDE_ROUTER_MATCHER.isTrueFor(parentCall.callee(), ctx))
        .filter(parentCall -> isRouterBeingIncluded(parentCall, argument))
        .filter(DecoratorAnalysis::hasFalsyIncludeInSchema)
        .isPresent();
    }

    private static boolean isRouterBeingIncluded(CallExpression includeRouterCall, RegularArgument argument) {
      return TreeUtils.nthArgumentOrKeywordOptional(0, "router", includeRouterCall.arguments())
        .filter(routerArgument -> routerArgument == argument)
        .isPresent();
    }

    private static Expression resolveLocalAliasChain(Expression expression) {
      // Follow simple local aliases like `internal = APIRouter(include_in_schema=False)` while preserving the
      // last expression when resolution stops.
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

    private Set<Integer> processDecorator(CallExpression callExpr) {
      RegularArgument responsesArg = TreeUtils.argumentByKeyword("responses", callExpr.arguments());
      if (responsesArg != null) {
        Expression responsesExpr = responsesArg.expression();
        if (responsesExpr instanceof DictionaryLiteral) {
          return extractDocumentedStatusCodes(responsesExpr);
        } else {
          canAnalyzeResponses = false;
        }
      }
      return Set.of();
    }

    private static Set<Integer> extractDocumentedStatusCodes(Expression responsesExpr) {
      Set<Integer> statusCodes = new HashSet<>();

      if (responsesExpr instanceof DictionaryLiteral dictLiteral) {
        for (DictionaryLiteralElement element : dictLiteral.elements()) {
          if (element instanceof KeyValuePair keyValuePair) {
            Expression key = keyValuePair.key();
            extractStatusCode(key).ifPresent(statusCodes::add);
          }
        }
      }

      return statusCodes;
    }
  }

  private void reportUndocumentedExceptions(
    SubscriptionContext ctx,
    List<RaiseInfo> httpExceptions,
    Set<Integer> documentedStatusCodes) {
    for (RaiseInfo raiseInfo : httpExceptions) {
      if (!documentedStatusCodes.contains(raiseInfo.statusCode) && !reportedHttpExceptionCalls.contains(raiseInfo.httpExceptionExpression)) {
        ctx.addIssue(raiseInfo.httpExceptionExpression, String.format(MESSAGE, raiseInfo.statusCode));
        reportedHttpExceptionCalls.add(raiseInfo.httpExceptionExpression);
      }
    }
  }

  private record DecoratorAnalysisResult(
    boolean isFastApiEndpoint,
    boolean canAnalyzeResponses,
    Set<Integer> documentedStatusCodes) {
  }

  private static class RaiseInfoCollector {
    private final SubscriptionContext ctx;
    private final FunctionDef functionDef;

    RaiseInfoCollector(SubscriptionContext ctx, FunctionDef functionDef) {
      this.ctx = ctx;
      this.functionDef = functionDef;
    }

    public List<RaiseInfo> collect() {
      List<RaiseInfo> result = new ArrayList<>(HTTPExceptionVisitor.collect(ctx, functionDef));

      String fqn = FullyQualifiedNameHelper.getFullyQualifiedName(functionDef.name().typeV2()).orElse(null);
      if (fqn == null) {
        return result;
      }

      CallGraph callGraph = ctx.callGraph();

      callGraph.forwardStream(fqn)
        .limit(MAX_FUNCTION_CALLS)
        .forEach(node -> node.tree()
          .flatMap(TreeUtils.toOptionalInstanceOfMapper(FunctionDef.class))
          .ifPresent(calledFunction -> result.addAll(HTTPExceptionVisitor.collect(ctx, calledFunction))));

      return result;
    }
  }

  private static class HTTPExceptionVisitor extends BaseTreeVisitor {
    private final SubscriptionContext ctx;
    private final List<RaiseInfo> httpExceptions = new ArrayList<>();

    HTTPExceptionVisitor(SubscriptionContext ctx) {
      this.ctx = ctx;
    }

    @Override
    public void visitRaiseStatement(RaiseStatement raiseStmt) {
      List<RaiseInfo> raiseInfos = raiseStmt.expressions().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class))
        .filter(callExpr -> IS_HTTP_EXCEPTION.isTrueFor(callExpr.callee(), ctx))
        .flatMap(HTTPExceptionVisitor::extractRaiseInfos)
        .toList();

      httpExceptions.addAll(raiseInfos);
      super.visitRaiseStatement(raiseStmt);
    }

    private static Stream<RaiseInfo> extractRaiseInfos(CallExpression callExpr) {
      return extractStatusCodeFromHTTPException(callExpr).map(statusCode -> new RaiseInfo(callExpr.callee(), statusCode));
    }

    private static Stream<Integer> extractStatusCodeFromHTTPException(CallExpression callExpr) {
      RegularArgument statusCodeArg = TreeUtils.nthArgumentOrKeyword(0, "status_code", callExpr.arguments());

      if (statusCodeArg == null) {
        return Stream.empty();
      }

      return extractStatusCode(statusCodeArg.expression()).stream();
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // don't decend into nested functions
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      // don't decend into nested lambdas
    }

    public static List<RaiseInfo> collect(SubscriptionContext ctx, FunctionDef tree) {
      HTTPExceptionVisitor visitor = new HTTPExceptionVisitor(ctx);
      tree.body().accept(visitor);
      return visitor.httpExceptions;
    }
  }

  private static Optional<Integer> extractStatusCode(Expression statusCodeExpr) {
    if (statusCodeExpr instanceof Name name) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(name);
      if (singleAssignedValue != null) {
        return extractStatusCode(singleAssignedValue);
      }
    } else if (statusCodeExpr instanceof NumericLiteral numericLiteral) {
      try {
        return Optional.of((int) numericLiteral.valueAsLong());
      } catch (NumberFormatException e) {
        return Optional.empty();
      }
    } else if (statusCodeExpr instanceof StringLiteral stringLiteral) {
      try {
        return Optional.of(Integer.parseInt(stringLiteral.trimmedQuotesValue()));
      } catch (NumberFormatException e) {
        return Optional.empty();
      }
    }
    return Optional.empty();
  }

  private static class RaiseInfo {
    final Expression httpExceptionExpression;
    final int statusCode;

    RaiseInfo(Expression httpExceptionExpression, int statusCode) {
      this.httpExceptionExpression = httpExceptionExpression;
      this.statusCode = statusCode;
    }
  }
}
