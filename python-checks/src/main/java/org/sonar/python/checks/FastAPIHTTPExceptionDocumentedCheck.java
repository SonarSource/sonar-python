/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
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
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
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

  private static final TypeMatcher IS_HTTP_EXCEPTION = TypeMatchers.any(
    TypeMatchers.isType("fastapi.exceptions.HTTPException"),
    TypeMatchers.isType("fastapi.HTTPException"));

  private static final int MAX_RECURSION_DEPTH = 5;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FastAPIHTTPExceptionDocumentedCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
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

  private static void reportUndocumentedExceptions(
    SubscriptionContext ctx,
    List<RaiseInfo> httpExceptions,
    Set<Integer> documentedStatusCodes) {
    for (RaiseInfo raiseInfo : httpExceptions) {
      if (!documentedStatusCodes.contains(raiseInfo.statusCode)) {
        ctx.addIssue(raiseInfo.httpExceptionExpression, String.format(MESSAGE, raiseInfo.statusCode));
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
    private final Set<FunctionDef> visited = new HashSet<>();

    RaiseInfoCollector(SubscriptionContext ctx, FunctionDef functionDef) {
      this.ctx = ctx;
      this.functionDef = functionDef;
    }

    public List<RaiseInfo> collect() {
      return collect(functionDef, 0);
    }

    private List<RaiseInfo> collect(FunctionDef functionDef, int depth) {
      List<RaiseInfo> result = new ArrayList<>();

      if (visited.contains(functionDef) || depth > MAX_RECURSION_DEPTH) {
        return result;
      }
      visited.add(functionDef);

      StatementList body = functionDef.body();
      if (body == null) {
        return result;
      }

      HTTPExceptionVisitor visitor = new HTTPExceptionVisitor(ctx);
      body.accept(visitor);
      result.addAll(visitor.httpExceptions);

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
  }

  private static Optional<Integer> extractStatusCode(Expression statusCodeExpr) {
    if (statusCodeExpr instanceof Name name) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(name);
      if (singleAssignedValue != null) {
        return extractStatusCode(singleAssignedValue);
      }
    } else if (statusCodeExpr instanceof NumericLiteral numericLiteral) {
      return Optional.of((int) numericLiteral.valueAsLong());
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
