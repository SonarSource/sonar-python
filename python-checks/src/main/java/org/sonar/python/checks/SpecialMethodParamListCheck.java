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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5722")
public class SpecialMethodParamListCheck extends PythonSubscriptionCheck {

  // Only the "self" parameter
  private static final List<String> EXACTLY_ONE = Arrays.asList(
    "__del__", "__repr__", "__str__", "__bytes__", "__hash__", "__bool__", "__dir__", "__len__", "__length_hint__",
    "__iter__", "__reversed__", "__neg__", "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__float__",
    "__index__", "__trunc__", "__floor__", "__ceil__", "__enter__", "__await__", "__aiter__", "__anext__", "__aenter__",
    "__getnewargs_ex__", "__getnewargs__", "__getstate__", "__reduce__", "__copy__", "__unicode__", "__nonzero__",
    "__fspath__");

  private static final List<String> EXACTLY_TWO = Arrays.asList(
    // "self" + 1 parameter
    "__format__", "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__", "__getattr__", "__getattribute__",
    "__delattr__", "__delete__", "__instancecheck__", "__subclasscheck__", "__getitem__", "__missing__", "__delitem__",
    "__contains__", "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__", "__mod__", "__pow__",
    "__divmod__", "__lshift__", "__rshift__", "__and__", "__xor__", "__or__", "__radd__", "__rsub__", "__rmul__",
    "__rmatmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__", "__rdivmod__", "__rlshift__", "__rrshift__",
    "__rand__", "__rxor__", "__ror__", "__iadd__", "__isub__", "__imul__", "__imatmul__", "__itruediv__", "__ifloordiv__",
    "__imod__", "__ipow__", "__ilshift__", "__irshift__", "__iand__", "__ixor__", "__ior__", "__round__", "__setstate__",
    "__reduce_ex__", "__deepcopy__", "__cmp__", "__div__",
    // Two parameters including the first class argument
    "__class_getitem__");

  // "self" + 2 parameters
  private static final List<String> EXACTLY_THREE = Arrays.asList(
    "__setattr__", "__get__", "__set__", "__setitem__", "__set_name__");

  // "self" + 3 parameters
  private static final List<String> EXACTLY_FOUR = Arrays.asList("__exit__", "__aexit__");

  private static Map<String, Integer> numberOfParams;

  static {
    numberOfParams = new HashMap<>();
    numberOfParams.putAll(EXACTLY_ONE.stream().collect(Collectors.toMap(Function.identity(), v -> 1)));
    numberOfParams.putAll(EXACTLY_TWO.stream().collect(Collectors.toMap(Function.identity(), v -> 2)));
    numberOfParams.putAll(EXACTLY_THREE.stream().collect(Collectors.toMap(Function.identity(), v -> 3)));
    numberOfParams.putAll(EXACTLY_FOUR.stream().collect(Collectors.toMap(Function.identity(), v -> 4)));
  }

  private static String lessThanExpectedMessage(String name, int expected, int actual) {
    return String.format("Add %d parameters. Method %s should have %d parameters.", expected - actual, name, expected);
  }

  private static String moreThanExpectedMessage(String name, int expected, int actual) {
    return String.format("Remove %d parameters. Method %s should have %d parameters.", actual - expected, name, expected);
  }

  private static boolean isRelevantMethodDefinition(FunctionDef def) {
    if (!def.isMethodDefinition()) {
      return false;
    }

    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(def, Tree.Kind.CLASSDEF);
    ClassSymbol classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
    if (classSymbol == null) {
      return false;
    }

    if (classSymbol.isOrExtends("zope.interface.Interface")) {
      return false;
    }

    String name = def.name().name();
    return name.startsWith("__") && name.endsWith("__");
  }

  private static boolean hasPackedOrKeywordParameter(List<AnyParameter> parameters) {
    return parameters.stream().filter(p -> p.is(Tree.Kind.PARAMETER)).map(Parameter.class::cast).anyMatch(p -> p.starToken() != null);
  }

  private static boolean isRequiredParameter(AnyParameter parameter) {
    if (parameter.is(Tree.Kind.TUPLE_PARAMETER)) {
      return true;
    }

    return parameter.is(Tree.Kind.PARAMETER) && ((Parameter) parameter).defaultValue() == null;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (!isRelevantMethodDefinition(functionDef)) {
        return;
      }

      String name = functionDef.name().name();
      ParameterList parameterList = functionDef.parameters();

      List<AnyParameter> parameters = Collections.emptyList();
      if (parameterList != null) {
        parameters = parameterList.all();
      }

      // Do not raise and bail out if the method was declared with packed or keyword arguments.
      if (hasPackedOrKeywordParameter(parameters)) {
        return;
      }

      Integer expectedParams = numberOfParams.get(functionDef.name().name());
      if (expectedParams == null) {
        // The special method was not found in the map.
        return;
      }

      List<AnyParameter> nonOptionalParameters = parameters.stream()
        .filter(SpecialMethodParamListCheck::isRequiredParameter)
        .toList();
      int numRequiredParams = nonOptionalParameters.size();
      int numAllParams = parameters.size();

      if (numRequiredParams > expectedParams) {
        PreciseIssue issue = ctx.addIssue(functionDef.name(), moreThanExpectedMessage(name, expectedParams, numRequiredParams));
        for (int i = expectedParams; i < nonOptionalParameters.size(); ++i) {
          issue.secondary(nonOptionalParameters.get(i), null);
        }
      } else if (numRequiredParams < expectedParams && numAllParams < expectedParams) {
        ctx.addIssue(functionDef.name(), lessThanExpectedMessage(name, expectedParams, numAllParams));
      }
    });
  }
}
