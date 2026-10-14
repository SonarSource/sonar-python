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
package org.sonar.python.tree;

import com.sonar.sslr.api.RecognitionException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2Test;
import org.sonar.python.semantic.v2.types.TypeDependenciesCalculator;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.types.v2.TypesTestUtils.DICT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.SET_TYPE;

class UnpackingComprehensionTest {

  @ParameterizedTest
  @ValueSource(strings = {
    "[*chunk for chunk in chunks]", "{*chunk for chunk in chunks}", "(*chunk for chunk in chunks)",
    "[*chunk async for chunk in chunks]", "{*chunk async for chunk in chunks}", "(*chunk async for chunk in chunks)",
    "[*(left + right) for left, right in chunks if left for right in others]",
    "[*(left if condition else right) for left, right in chunks]"
  })
  void iterable_unpacking(String code) {
    var expression = (ComprehensionExpression) PythonTestUtils.lastExpression(code);
    var unpacking = (UnpackingExpression) expression.resultExpression();
    assertThat(unpacking.starToken().value()).isEqualTo("*");
    assertThat(unpacking.expression()).isNotNull();
    assertThat(unpacking.parent()).isSameAs(expression);
    assertThat(expression.comprehensionFor().asyncToken() != null).isEqualTo(code.contains("async"));
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "{**mapping for mapping in mappings}", "{**mapping async for mapping in mappings}",
    "{**mapping for mapping in mappings if mapping for other in mapping}",
    "{**(left if condition else right) for left, right in mappings}",
    "{**left if condition else right for left, right in mappings}",
    "{**lambda: mapping for mapping in mappings}"
  })
  void mapping_unpacking(String code) {
    var expression = (DictCompExpression) PythonTestUtils.lastExpression(code);
    assertThat(expression.getKind()).isEqualTo(Tree.Kind.DICT_COMPREHENSION);
    var unpacking = (UnpackingExpression) expression.resultElement();
    assertThat(unpacking.starToken().value()).isEqualTo("**");
    assertThat(unpacking.parent()).isSameAs(expression);
    assertThat(expression.children()).containsExactly(expression.firstToken(), unpacking, expression.comprehensionFor(), expression.lastToken());
    assertThat(expression.comprehensionFor().asyncToken() != null).isEqualTo(code.contains("async"));
    assertThat(new TypeDependenciesCalculator().getTypeDependencies(expression)).containsExactly(unpacking);
  }

  @Test
  void regular_dictionary_comprehension_has_a_key_value_head() {
    var expression = (DictCompExpression) PythonTestUtils.lastExpression("{key: value for key, value in pairs}");
    var pair = (KeyValuePair) expression.resultElement();
    assertThat(pair.parent()).isSameAs(expression);
    assertThat(pair.key().parent()).isSameAs(pair);
    assertThat(pair.value().parent()).isSameAs(pair);
    assertThat(pair.colon().parent()).isSameAs(pair);
    assertThat(pair.colon().value()).isEqualTo(":");
    assertThat(expression.children()).containsExactly(expression.firstToken(), pair, expression.comprehensionFor(), expression.lastToken());
    assertThat(pair.children()).containsExactly(pair.key(), pair.colon(), pair.value());
  }

  @Test
  @SuppressWarnings("deprecation")
  void legacy_accessors_delegate_to_the_typed_head() {
    var regular = (DictCompExpression) PythonTestUtils.lastExpression("{key: value for key, value in pairs}");
    var pair = (KeyValuePair) regular.resultElement();
    assertThat(regular.keyExpression()).isSameAs(pair.key());
    assertThat(regular.valueExpression()).isSameAs(pair.value());
    assertThat(regular.colonToken()).isSameAs(pair.colon());

    var mapping = (DictCompExpression) PythonTestUtils.lastExpression("{**item for item in items}");
    assertThat(mapping.keyExpression()).isNull();
    assertThat(mapping.valueExpression()).isNull();
    assertThat(mapping.colonToken()).isNull();
  }

  @ParameterizedTest
  @ValueSource(strings = {"f(*chunk for chunk in chunks)", "f(*chunk async for chunk in chunks)"})
  void sole_argument_is_a_generator(String code) {
    var call = (CallExpression) PythonTestUtils.lastExpression(code);
    assertThat(call.arguments()).hasSize(1);
    var argument = (RegularArgument) call.arguments().get(0);
    var generator = (ComprehensionExpression) argument.expression();
    assertThat(generator.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);
    assertThat(generator.resultExpression()).isInstanceOf(UnpackingExpression.class);
    assertThat(generator.firstToken().value()).isEqualTo("*");
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "(*left or right for left, right in pairs)", "f(*left or right for left, right in pairs)",
    "(*left or right async for left, right in pairs)", "f(*left or right async for left, right in pairs)",
    "(*left and right for left, right in pairs)", "f(*left and right for left, right in pairs)",
    "(*left and right async for left, right in pairs)", "f(*left and right async for left, right in pairs)",
    "(*left if condition else right for left, right in pairs)", "f(*left if condition else right for left, right in pairs)",
    "(*left if condition else right async for left, right in pairs)", "f(*left if condition else right async for left, right in pairs)",
    "(*lambda: left for left, right in pairs)", "f(*lambda: left for left, right in pairs)",
    "(*lambda: left async for left, right in pairs)", "f(*lambda: left async for left, right in pairs)"
  })
  void generator_unpacking_accepts_full_expression_heads(String code) {
    for (boolean notebook : new boolean[] {false, true}) {
      var parser = notebook ? PythonParser.createIPythonParser() : PythonParser.create();
      PythonTreeMaker maker = notebook ? new IPythonTreeMaker() : new PythonTreeMaker();
      var file = maker.fileInput(parser.parse(code + "\nsentinel\n"));
      assertThat(file.statements().statements()).hasSize(2);
      assertThat(file.statements().statements().get(1).firstToken().value()).isEqualTo("sentinel");
      var expression = ((ExpressionStatement) file.statements().statements().get(0)).expressions().get(0);
      if (expression instanceof CallExpression call) {
        assertThat(call.arguments()).hasSize(1);
        expression = ((RegularArgument) call.arguments().get(0)).expression();
        assertThat(expression.firstToken().value()).isEqualTo("*");
      } else {
        assertThat(expression.firstToken().value()).isEqualTo("(");
        assertThat(expression.lastToken().value()).isEqualTo(")");
      }
      var generator = (ComprehensionExpression) expression;
      assertThat(generator.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);
      assertThat(generator.comprehensionFor().asyncToken() != null).isEqualTo(code.contains("async"));
      var unpacking = (UnpackingExpression) generator.resultExpression();
      assertThat(unpacking.parent()).isSameAs(generator);
      assertThat(unpacking.starToken().value()).isEqualTo("*");
      assertThat(unpacking.expression().parent()).isSameAs(unpacking);
      assertThat(unpacking.children()).containsExactly(unpacking.starToken(), unpacking.expression());
      if (code.contains("else")) {
        var conditional = (ConditionalExpression) unpacking.expression();
        assertThat(conditional.trueExpression().firstToken().value()).isEqualTo("left");
        assertThat(conditional.condition().firstToken().value()).isEqualTo("condition");
        assertThat(conditional.falseExpression().firstToken().value()).isEqualTo("right");
      } else if (code.contains("lambda")) {
        assertThat(unpacking.expression()).isInstanceOf(LambdaExpression.class);
      } else {
        var binary = (BinaryExpression) unpacking.expression();
        assertThat(binary.operator().value()).isEqualTo(code.contains(" or ") ? "or" : "and");
        assertThat(binary.leftOperand().firstToken().value()).isEqualTo("left");
        assertThat(binary.rightOperand().firstToken().value()).isEqualTo("right");
      }
    }
  }

  @Test
  void parenthesized_generator_can_be_unpacked_as_call_arguments() {
    var call = (CallExpression) PythonTestUtils.lastExpression("f(*(*chunk for chunk in chunks))");
    var argument = (UnpackingExpression) call.arguments().get(0);
    assertThat(argument.expression()).isInstanceOf(ComprehensionExpression.class);
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "[**mapping for mapping in mappings]", "(**mapping for mapping in mappings)",
    "{*key: value for key, value in pairs}", "{key: *value for key, value in pairs}",
    "{**key: value for key, value in pairs}", "{key: **value for key, value in pairs}",
    "[*left if condition else right for left, right in chunks]", "{*left if condition else right for left, right in chunks}",
    "[*left or right for left, right in chunks]", "{*left or right for left, right in chunks}",
    "[*left and right for left, right in chunks]", "{*left and right for left, right in chunks}",
    "[*lambda: left for left in chunks]", "{*lambda: left for left in chunks}",
    "f(other, *chunk for chunk in chunks)",
    "f(**mapping for mapping in mappings)",
    "[*first, *second for first, second in chunks]", "{**first, **second for first, second in mappings}"
  })
  void invalid_unpacking(String code) {
    assertThatThrownBy(() -> PythonTestUtils.parseWithoutSymbols(code)).isInstanceOf(RecognitionException.class);
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "f(*chunk for chunk in chunks, other)",
    "f(*chunk for chunk in chunks for other in others, extra)",
    "f(*chunk for chunk in chunks if flag for other in others, extra)",
    "f(*chunk async for chunk in chunks if flag async for other in others, extra)",
    "[*chunk for chunk in chunks, extra]", "{*chunk for chunk in chunks, extra}",
    "(*chunk for chunk in chunks,)", "{**mapping for mapping in mappings, extra}",
    "[*chunk for chunk in chunks for other in others, extra]",
    "{*chunk for chunk in chunks if flag for other in others, extra}",
    "(*chunk for chunk in chunks if flag for other in others, extra)",
    "{**mapping for mapping in mappings if flag for other in others, extra}"
  })
  void unparenthesized_tuple_iterables_preserve_comprehensions_and_the_rest_of_the_file(String code) {
    // Invalid Python syntax: comma-containing comprehension iterables must be parenthesized.
    // Keep the grammar's tuple representation for analysis; it does not establish valid runtime semantics.
    for (boolean notebook : new boolean[] {false, true}) {
      var parser = notebook ? PythonParser.createIPythonParser() : PythonParser.create();
      PythonTreeMaker maker = notebook ? new IPythonTreeMaker() : new PythonTreeMaker();
      var file = maker.fileInput(parser.parse(code + "\nsentinel\n"));
      assertThat(file.statements().statements()).hasSize(2);
      assertThat(file.statements().statements().get(1).firstToken().value()).isEqualTo("sentinel");
      var expression = ((ExpressionStatement) file.statements().statements().get(0)).expressions().get(0);
      if (expression instanceof CallExpression call) {
        assertThat(call.arguments()).hasSize(1);
        expression = ((RegularArgument) call.arguments().get(0)).expression();
        assertThat(expression.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);
      }
      var head = expression instanceof DictCompExpression dict
        ? dict.resultElement() : ((ComprehensionExpression) expression).resultExpression();
      assertThat(head).isInstanceOf(UnpackingExpression.class);
      assertThat(head.parent()).isSameAs(expression);
      var clauses = PythonTestUtils.<ComprehensionFor>getAllDescendant(expression, tree -> tree.is(Tree.Kind.COMP_FOR));
      var lastClause = clauses.get(clauses.size() - 1);
      var iterable = (Tuple) lastClause.iterable();
      assertThat(iterable.parent()).isSameAs(lastClause);
      assertThat(iterable.commas()).hasSize(1);
      assertThat(iterable.commas().get(0).value()).isEqualTo(",");
      assertThat(iterable.elements()).hasSize(code.contains("chunks,)") ? 1 : 2);
      assertThat(lastClause.asyncToken() != null).isEqualTo(code.contains("async"));
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "[*chunk for left, chunk in (pairs, extra)]",
    "{*chunk for left, chunk in (pairs, extra)}",
    "(*chunk for left, chunk in (pairs, extra))",
    "{**mapping for left, mapping in (pairs, extra)}",
    "f(*chunk for left, chunk in (pairs, extra))",
    "[*chunk for chunk in chunks if flag for left, other in (pairs, extra)]",
    "{*chunk for chunk in chunks if flag for left, other in (pairs, extra)}",
    "(*chunk for chunk in chunks if flag for left, other in (pairs, extra))",
    "{**mapping for mapping in mappings if flag for left, other in (pairs, extra)}",
    "f(*chunk async for chunk in chunks if flag async for left, other in (pairs, extra))"
  })
  void parenthesized_tuple_iterables_and_tuple_targets_are_valid(String code) {
    assertThatCode(() -> new PythonTreeMaker().fileInput(PythonParser.create().parse(code))).doesNotThrowAnyException();
  }

  @Test
  void ordinary_comprehensions_retain_legacy_tuple_iterables() {
    assertThatCode(() -> new PythonTreeMaker().fileInput(PythonParser.create().parse("[chunk for chunk in chunks, extra]")))
      .doesNotThrowAnyException();
  }

  @ParameterizedTest
  @ValueSource(strings = {"[*item for item in items if item]", "{*item for item in items if item}",
    "(*item for item in items if item)", "{**item for item in items if item}",
    "[*item async for item in items if item]", "{**item async for item in items if item}",
    "{item: 0 for item in items if item}", "{item: 0 async for item in items if item}"})
  void head_expression_uses_comprehension_scope(String code) {
    var file = PythonTestUtils.parse("item = 42", "items = []", code);
    var expression = ((ExpressionStatement) file.statements().statements().get(2)).expressions().get(0);
    var head = expression instanceof DictCompExpression dict ? dict.resultElement() : ((ComprehensionExpression) expression).resultExpression();
    var clause = expression instanceof DictCompExpression dict ? dict.comprehensionFor() : ((ComprehensionExpression) expression).comprehensionFor();
    var headName = (Name) (head instanceof KeyValuePair pair ? pair.key() : ((UnpackingExpression) head).expression());
    var targetName = (Name) clause.loopExpression();
    var conditionName = (Name) ((ComprehensionIf) clause.nestedClause()).condition();
    assertThat(headName.symbol()).isSameAs(targetName.symbol()).isSameAs(conditionName.symbol());
    assertThat(headName.symbol().usages()).hasSize(3);
    var symbolTable = new SymbolTableBuilderV2(file).build();
    assertThat(headName.symbolV2()).isSameAs(targetName.symbolV2()).isSameAs(conditionName.symbolV2());
    assertThat(headName.symbolV2().usages()).hasSize(3);
    assertThat(symbolTable.getSymbolsByRootTree(expression)).hasSize(1);
    assertThat(symbolTable.getSymbolsByRootTree(file)).noneMatch(symbol -> symbol == headName.symbolV2());
  }

  @Test
  void visitor_scans_unpacked_mapping() {
    var expression = PythonTestUtils.lastExpression("{**item for item in items}");
    var names = new java.util.ArrayList<String>();
    expression.accept(new BaseTreeVisitor() {
      @Override
      public void visitName(Name name) {
        names.add(name.name());
      }
    });
    assertThat(names).containsExactly("item", "item", "items");
  }

  @Test
  void collection_types_are_preserved() {
    assertThat(PythonTestUtils.lastExpression("[*item for item in items]").type()).isEqualTo(InferredTypes.LIST);
    assertThat(PythonTestUtils.lastExpression("{*item for item in items}").type()).isEqualTo(InferredTypes.SET);
    assertThat(PythonTestUtils.lastExpression("{**item for item in items}").type()).isEqualTo(InferredTypes.DICT);
    assertThat(TypeInferenceV2Test.lastExpression("[*item for item in items]").typeV2().unwrappedType()).isEqualTo(LIST_TYPE);
    assertThat(TypeInferenceV2Test.lastExpression("{*item for item in items}").typeV2().unwrappedType()).isEqualTo(SET_TYPE);
    assertThat(TypeInferenceV2Test.lastExpression("{**item for item in items}").typeV2().unwrappedType()).isEqualTo(DICT_TYPE);
  }
}
