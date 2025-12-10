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
package org.sonar.python.semantic.v2.types;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

public final class CallReturnTypeCalculator {

  private CallReturnTypeCalculator() {
  }

  public static PythonType computeCallExpressionType(CallExpression callExpr, TypePredicateContext typePredicateContext) {
    PythonType calleeType = callExpr.callee().typeV2();
    TypeSource typeSource = computeTypeSource(calleeType, callExpr);
    PythonType returnType = returnTypeOfCall(calleeType);

    if (returnType instanceof ObjectType objectType) {
      returnType = ObjectType.Builder.fromType(objectType)
        .withTypeSource(typeSource)
        .build();
    }

    return collapseSelfTypeIfNeeded(callExpr, returnType, typePredicateContext);
  }

  private static PythonType returnTypeOfCall(PythonType calleeType) {
    if (calleeType instanceof ClassType classType) {
      return ObjectType.fromType(classType);
    }
    if (calleeType instanceof FunctionType functionType) {
      return functionType.returnType();
    }
    if (calleeType instanceof UnionType unionType) {
      Set<PythonType> types = new HashSet<>();
      for (PythonType candidate : unionType.candidates()) {
        PythonType typeOfCandidate = returnTypeOfCall(candidate);
        if (typeOfCandidate instanceof UnknownType) {
          return PythonType.UNKNOWN;
        }
        types.add(typeOfCandidate);
      }
      return UnionType.or(types);
    }
    if (calleeType instanceof ObjectType objectType) {
      Optional<PythonType> pythonType = objectType.resolveMember("__call__");
      return pythonType.map(CallReturnTypeCalculator::returnTypeOfCall).orElse(PythonType.UNKNOWN);
    }
    return PythonType.UNKNOWN;
  }

  private static PythonType collapseSelfTypeIfNeeded(CallExpression callExpr, PythonType returnType, TypePredicateContext typePredicateContext) {
    if (!containsSelfType(returnType, typePredicateContext)) {
      return returnType;
    }

    // Methods not called as instance methods don't have a receiver from which the return type can be inferred
    PythonType receiverType = getReceiverType(callExpr);
    if (!isInstanceMethodCall(callExpr) || receiverType == PythonType.UNKNOWN) {
      return PythonType.UNKNOWN;
    }

    return collapseSelfType(returnType, receiverType);
  }

  private static boolean containsSelfType(PythonType type, TypePredicateContext typePredicateContext) {
    return TypeInferenceMatcher.of(
      TypeInferenceMatchers.any(
        TypeInferenceMatchers.isSelf(),
        TypeInferenceMatchers.isObjectSatisfying(TypeInferenceMatchers.isSelf())))
      .evaluate(type, typePredicateContext).isTrue();
  }

  private static boolean isInstanceMethodCall(CallExpression callExpr) {
    PythonType calleeType = callExpr.callee().typeV2();
    if (calleeType instanceof FunctionType functionType) {
      return functionType.isInstanceMethod();
    }
    return false;
  }

  private static PythonType getReceiverType(CallExpression callExpr) {
    if (callExpr.callee() instanceof QualifiedExpression qualifiedExpr) {
      PythonType qualifierType = qualifiedExpr.qualifier().typeV2();
      if (qualifierType instanceof ObjectType objectType) {
        return objectType.type();
      }
    }
    return PythonType.UNKNOWN;
  }

  private static PythonType collapseSelfType(PythonType returnType, PythonType receiverType) {
    if (returnType instanceof ObjectType objectType && objectType.type() instanceof SelfType) {
      return ObjectType.Builder.fromType(objectType)
        .withType(receiverType)
        .build();
    }
    return PythonType.UNKNOWN;
  }

  private static TypeSource computeTypeSource(PythonType calleeType, CallExpression callExpr) {
    if (isCalleeLocallyDefinedFunction(calleeType)) {
      return TypeSource.TYPE_HINT;
    }
    return calleeTypeSource(callExpr);
  }

  private static boolean isCalleeLocallyDefinedFunction(PythonType pythonType) {
    if (pythonType instanceof FunctionType functionType) {
      return functionType.typeOrigin() == TypeOrigin.LOCAL;
    }
    if (pythonType instanceof UnionType unionType) {
      return unionType.candidates().stream().anyMatch(CallReturnTypeCalculator::isCalleeLocallyDefinedFunction);
    }
    return false;
  }

  private static TypeSource calleeTypeSource(CallExpression callExpr) {
    if (callExpr.callee() instanceof QualifiedExpression qualifiedExpression) {
      return qualifiedExpression.qualifier().typeV2().typeSource();
    }
    return callExpr.callee().typeV2().typeSource();
  }
}

