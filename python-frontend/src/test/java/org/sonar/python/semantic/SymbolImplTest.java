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
package org.sonar.python.semantic;

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpressionInFunction;
import static org.sonar.python.PythonTestUtils.parse;


class SymbolImplTest {

  @Test
  void assert_is() {
    Symbol x = symbols("x = 42").get("x");
    assertThat(x.is(Symbol.Kind.OTHER)).isTrue();

    Symbol foo = symbols("def foo(): ...").get("foo");
    assertThat(foo.is(Symbol.Kind.FUNCTION)).isTrue();
    assertThat(foo.is(Symbol.Kind.OTHER)).isFalse();
    assertThat(foo.is(Symbol.Kind.OTHER, Symbol.Kind.FUNCTION)).isTrue();
  }

  @Test
  void removeUsages() {
    Symbol x = symbols("x = 42").get("x");
    assertThat(x.usages()).isNotEmpty();
    ((SymbolImpl) x).removeUsages();
    assertThat(x.usages()).isEmpty();

    FileInput fileInput = parse(
      "obj = {}",
      "obj.foo");
    QualifiedExpression qualifiedExpr = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.QUALIFIED_EXPR));
    assertThat(qualifiedExpr.symbol().usages()).isNotEmpty();

    Symbol obj = ((HasSymbol) qualifiedExpr.qualifier()).symbol();
    ((SymbolImpl) obj).removeUsages();
    assertThat(qualifiedExpr.symbol().usages()).isEmpty();
  }

  @Test
  void inferred_type_after_copy() {
    SymbolImpl symbol = (SymbolImpl) ((HasSymbol) lastExpressionInFunction(
      "e = OSError()",
      "e.errno"
    )).symbol();
    assertThat(symbol.inferredType()).isEqualTo(InferredTypes.INT);
    SymbolImpl copiedSymbol = symbol.copyWithoutUsages();
    assertThat(copiedSymbol.inferredType()).isEqualTo(InferredTypes.INT);
  }

  @Test
  void annotated_type_name_null_by_default_test() {
    var x = new SymbolImpl("x", "module.x");
    assertThat(x.annotatedTypeName()).isNull();
  }

  private Map<String, Symbol> symbols(String... code) {
    FileInput fileInput = parse(new SymbolTableBuilder("", PythonTestUtils.pythonFile("foo")), code);
    return fileInput.globalVariables().stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
  }
}
