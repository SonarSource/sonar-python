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
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.DeclaredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.types.InferredTypes.containsDeclaredType;

@Rule(key = "S5886")
public class FunctionReturnTypeCheck extends PythonSubscriptionCheck {

  private static final TypeMatcher UNION_ANNOTATION = TypeMatchers.any(
    TypeMatchers.isType("typing.Optional"), TypeMatchers.isType("typing.Union"));
  private static final TypeMatcher SUBCLASS_FACTORY = TypeMatchers.any(
    TypeMatchers.isType("logging.getLogger"), TypeMatchers.isType("logging.Logger.getChild"),
    TypeMatchers.isType("sqlalchemy.and_"), TypeMatchers.isType("sqlalchemy.or_"));

  private static final String MESSAGE = "Return a value of type \"%s\" instead of \"%s\" or update function \"%s\" type hint.";
  private static final Set<String> ITERABLE_TYPES = Set.of("typing.Generator", "typing.Iterator", "typing.Iterable",
    "collections.abc.Generator", "collections.abc.Iterator", "collections.abc.Iterable");
  private static final Set<String> ASYNC_ITERABLE_TYPES = Set.of("typing.AsyncGenerator", "typing.AsyncIterator", "typing.AsyncIterable", "collections.abc.AsyncGenerator",
    "collections.abc.AsyncIterator", "collections.abc.AsyncIterable");
  private static final String CONTEXT_MANAGER_DECORATOR_FQN = "contextlib.contextmanager";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      Symbol symbol = functionDef.name().symbol();
      if (symbol == null || !symbol.is(Symbol.Kind.FUNCTION)) {
        return;
      }

      if (isContextManager(functionDef)) {
        return;
      }

      FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
      InferredType declaredReturnType = functionSymbol.declaredReturnType();
      if (declaredReturnType == InferredTypes.anyType()) {
        return;
      }
      ReturnTypeVisitor returnTypeVisitor = new ReturnTypeVisitor(declaredReturnType, ctx);
      functionDef.body().accept(returnTypeVisitor);
      raiseIssues(ctx, functionDef, declaredReturnType, returnTypeVisitor);
    });
  }

  private static boolean isContextManager(FunctionDef functionDef) {
    return functionDef.decorators()
      .stream()
      .map(Decorator::expression)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::symbol)
      .filter(Objects::nonNull)
      .map(Symbol::fullyQualifiedName)
      .anyMatch(CONTEXT_MANAGER_DECORATOR_FQN::equals);
  }

  private static void raiseIssues(SubscriptionContext ctx, FunctionDef functionDef, InferredType declaredReturnType, ReturnTypeVisitor returnTypeVisitor) {
    String functionName = functionDef.name().name();
    String returnTypeName = InferredTypes.typeName(declaredReturnType);
    if (!returnTypeVisitor.yieldExpressions.isEmpty()) {
      boolean isAsyncFunction = functionDef.asyncKeyword() != null;
      String recommendedSuperType = isAsyncFunction ? "typing.AsyncGenerator" : "typing.Generator";
      // Here we should probably use an equivalent of "canBeOrExtend" (accepting uncertainty) instead of "mustBeOrExtend"
      if (canBeOrExtendIterableType(declaredReturnType)) {
        if (isMixedUpAnnotation(isAsyncFunction, declaredReturnType)) {
          returnTypeVisitor.yieldExpressions
            .forEach(y -> {
              PreciseIssue issue =
                ctx.addIssue(y, String.format("Annotate function \"%s\" with \"%s\" or one of its supertypes.", functionName, recommendedSuperType));
              addSecondaries(issue, functionDef);
            });
        }
        return;
      }
      returnTypeVisitor.yieldExpressions
        .forEach(y -> {
          PreciseIssue issue =
            ctx.addIssue(y, String.format("Remove this yield statement or annotate function \"%s\" with \"%s\" or one of its supertypes.", functionName, recommendedSuperType));
          addSecondaries(issue, functionDef);
        });
    }
    returnTypeVisitor.invalidReturnStatements.forEach(i -> {
      PreciseIssue issue;
      if (i.expressions().size() > 1) {
        issue = ctx.addIssue(i, String.format(MESSAGE, returnTypeName, "tuple", functionName));
      } else if (i.expressions().size() == 1 && InferredTypes.typeName(i.expressions().get(0).type()) != null) {
        issue = ctx.addIssue(i.expressions().get(0), String.format(MESSAGE, returnTypeName, InferredTypes.typeName(i.expressions().get(0).type()), functionName));
      } else {
        issue = ctx.addIssue(i, String.format("Return a value of type \"%s\" or update function \"%s\" type hint.", returnTypeName, functionName));
      }
      addSecondaries(issue, functionDef);
    });
  }

  private static boolean canBeOrExtendIterableType(InferredType inferredType) {
    return Stream.concat(ITERABLE_TYPES.stream(), ASYNC_ITERABLE_TYPES.stream())
      .anyMatch(typeName -> canBeOrExtend(inferredType, typeName));
  }

  private static boolean canBeOrExtend(InferredType inferredType, String typeName) {
    // The reason of the implementation of the method instead of using already having DeclaredType.canBeOrExtend
    // is that we consider the declared type of the type annotation to be as trustworthy as if it were a runtime type.
    return Optional.of(inferredType)
      .filter(DeclaredType.class::isInstance)
      .map(DeclaredType.class::cast)
      .map(DeclaredType::getTypeClass)
      .filter(ClassSymbol.class::isInstance)
      .map(ClassSymbol.class::cast)
      .map(s -> s.canBeOrExtend(typeName))
      .orElseGet(() -> inferredType.canBeOrExtend(typeName));
  }

  private static boolean isMixedUpAnnotation(boolean isAsyncFunction, InferredType declaredReturnType) {
    return isAsyncFunction ? ITERABLE_TYPES.stream().anyMatch(declaredReturnType::mustBeOrExtend) : ASYNC_ITERABLE_TYPES.stream().anyMatch(declaredReturnType::mustBeOrExtend);
  }

  private static void addSecondaries(PreciseIssue issue, FunctionDef functionDef) {
    issue.secondary(functionDef.name(), "Function definition.");
    TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();
    if (returnTypeAnnotation != null) {
      issue.secondary(returnTypeAnnotation.expression(), "Type hint.");
    }
  }

  private static class ReturnTypeVisitor extends BaseTreeVisitor {

    InferredType returnType;
    private final SubscriptionContext context;
    List<YieldExpression> yieldExpressions = new ArrayList<>();
    List<ReturnStatement> invalidReturnStatements = new ArrayList<>();

    ReturnTypeVisitor(InferredType returnType, SubscriptionContext context) {
      this.returnType = returnType;
      this.context = context;
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Don't visit nested functions
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      List<Expression> expressions = returnStatement.expressions();
      if (expressions.isEmpty()) {
        if (!InferredTypes.NONE.isCompatibleWith(returnType)) {
          invalidReturnStatements.add(returnStatement);
        }
      } else if (!returnStatement.commas().isEmpty()) {
        // Hardcoded "tuple" type due to a limitation on extracting type information from tuple literals
        if (!InferredTypes.TUPLE.isCompatibleWith(returnType)) {
          invalidReturnStatements.add(returnStatement);
        }
      } else {
        Expression expression = expressions.get(0);
        InferredType inferredType = expression.type();
        if (returnType.mustBeOrExtend("typing.TypedDict")) {
          // TODO SONARPY-1340: This is a workaround to avoid FPs with TypedDict, however, this produces false-negatives. We should have a more
          //  fine-grained solution to check dictionaries against TypedDict.
          return;
        }

        if (!containsDeclaredType(inferredType) && !inferredType.isCompatibleWith(returnType)
          && !canReturnAnnotatedSubclass(expression, returnStatement)) {
          invalidReturnStatements.add(returnStatement);
        }
      }
      super.visitReturnStatement(returnStatement);
    }

    private boolean canReturnAnnotatedSubclass(Expression expression, ReturnStatement statement) {
      Expression returned = Expressions.removeParentheses(expression);
      if (!(returned instanceof CallExpression call)
        || !hasStubReturnAnnotation(call.callee().typeV2())
        || !SUBCLASS_FACTORY.isTrueFor(call.callee(), context)
        || !(returned.typeV2() instanceof ObjectType object)
        || !(object.unwrappedType() instanceof ClassType returnedClass)) {
        return false;
      }
      // Logger configuration can live elsewhere. SQL boolean factories can produce a
      // BooleanClauseList or preserve a supplied expression, including a custom subclass.
      // Stub origin alone is not evidence of this behavior (e.g. os.urandom returns bytes).
      FunctionDef enclosingFunction = TreeUtils.firstAncestorOfClass(statement, FunctionDef.class);
      TypeAnnotation annotation = enclosingFunction == null ? null : enclosingFunction.returnTypeAnnotation();
      return annotation != null && annotationAllowsSubclass(annotation.expression(), returnedClass);
    }

    private static boolean hasStubReturnAnnotation(PythonType callee) {
      if (callee instanceof UnionType union) {
        return union.candidates().stream().allMatch(ReturnTypeVisitor::hasStubReturnAnnotation);
      }
      return callee instanceof FunctionType function && function.typeOrigin() == TypeOrigin.STUB
        && function.definitionLocation().isEmpty() && function.returnType() instanceof ObjectType;
    }

    private boolean annotationAllowsSubclass(Expression annotation, ClassType returnedClass) {
      Expression unwrapped = Expressions.removeParentheses(annotation);
      if (unwrapped instanceof SubscriptionExpression subscription
        && UNION_ANNOTATION.isTrueFor(subscription.object(), context)) {
        return subscription.subscripts().expressions().stream().anyMatch(candidate -> annotationAllowsSubclass(candidate, returnedClass));
      }
      if (unwrapped instanceof BinaryExpression union && union.is(Tree.Kind.BITWISE_OR)) {
        return annotationAllowsSubclass(union.leftOperand(), returnedClass) || annotationAllowsSubclass(union.rightOperand(), returnedClass);
      }
      // Require an actual inheritance relation, not structural or numeric compatibility.
      // Constructors and generic return classes are deliberately excluded.
      return unwrapped.typeV2() instanceof ClassType annotatedClass && !annotatedClass.isGeneric()
        && annotatedClass.mro().filter(mro -> mro.contains(returnedClass)).isPresent();
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      yieldExpressions.add(yieldExpression);
      super.visitYieldExpression(yieldExpression);
    }
  }
}
