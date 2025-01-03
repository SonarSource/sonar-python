/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.tree;

import com.sonar.sslr.api.RecognitionException;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.AsPattern;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.ClassPattern;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.GroupPattern;
import org.sonar.plugins.python.api.tree.Guard;
import org.sonar.plugins.python.api.tree.KeywordPattern;
import org.sonar.plugins.python.api.tree.DoubleStarPattern;
import org.sonar.plugins.python.api.tree.KeyValuePattern;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.LiteralPattern;
import org.sonar.plugins.python.api.tree.MappingPattern;
import org.sonar.plugins.python.api.tree.MatchStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.OrPattern;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SequencePattern;
import org.sonar.plugins.python.api.tree.StarPattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.WildcardPattern;
import org.sonar.plugins.python.api.tree.ValuePattern;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

class PythonTreeMakerMatchStatementTest extends RuleTest {

  private final PythonTreeMaker treeMaker = new PythonTreeMaker();

  @Test
  void match_statement() {
    setRootRule(PythonGrammar.MATCH_STMT);
    MatchStatement matchStatement = parse("match command:\n  case 42:...\n", treeMaker::matchStatement);
    assertThat(matchStatement.getKind()).isEqualTo(Kind.MATCH_STMT);
    assertThat(matchStatement.matchKeyword().value()).isEqualTo("match");
    assertThat(matchStatement.subjectExpression().getKind()).isEqualTo(Kind.NAME);
    assertThat(matchStatement.colon().value()).isEqualTo(":");
    assertThat(matchStatement.caseBlocks()).hasSize(1);
    assertThat(matchStatement.children()).extracting(Tree::getKind)
      .containsExactly(Kind.TOKEN, Kind.NAME, Kind.TOKEN, Kind.TOKEN, Kind.TOKEN, Kind.CASE_BLOCK, Kind.TOKEN);

    CaseBlock caseBlock = matchStatement.caseBlocks().get(0);
    assertThat(caseBlock.caseKeyword().value()).isEqualTo("case");
    assertThat(caseBlock.guard()).isNull();
    assertThat(caseBlock.colon().value()).isEqualTo(":");
    assertThat(caseBlock.body().statements()).extracting(Tree::getKind).containsExactly(Kind.EXPRESSION_STMT);
    assertThat(caseBlock.children()).extracting(Tree::getKind)
      .containsExactly(Kind.TOKEN, Kind.NUMERIC_LITERAL_PATTERN, Kind.TOKEN, Kind.STATEMENT_LIST);

    Pattern pattern = caseBlock.pattern();
    assertThat(pattern.getKind()).isEqualTo(Kind.NUMERIC_LITERAL_PATTERN);
    LiteralPattern literalPattern = (LiteralPattern) pattern;
    assertThat(literalPattern.children()).extracting(Tree::getKind)
      .containsExactly(Kind.TOKEN);
  }

  @Test
  void match_statement_tuple_subject() {
    setRootRule(PythonGrammar.MATCH_STMT);
    MatchStatement matchStatement = parse("match (x, y):\n  case 42:...\n", treeMaker::matchStatement);
    Expression subjectExpression = matchStatement.subjectExpression();
    assertThat(subjectExpression.getKind()).isEqualTo(Kind.TUPLE);
    assertThat(((Tuple) subjectExpression).elements()).hasSize(2);

    matchStatement = parse("match x, y:\n  case 42:...\n", treeMaker::matchStatement);
    subjectExpression = matchStatement.subjectExpression();
    assertThat(subjectExpression.getKind()).isEqualTo(Kind.TUPLE);
    assertThat(((Tuple) subjectExpression).elements()).hasSize(2);

    matchStatement = parse("match x,:\n  case 42:...\n", treeMaker::matchStatement);
    subjectExpression = matchStatement.subjectExpression();
    assertThat(subjectExpression.getKind()).isEqualTo(Kind.TUPLE);
    assertThat(((Tuple) subjectExpression).elements()).hasSize(1);

    matchStatement = parse("match (x):\n  case 42:...\n", treeMaker::matchStatement);
    subjectExpression = matchStatement.subjectExpression();
    assertThat(subjectExpression.getKind()).isEqualTo(Kind.PARENTHESIZED);
  }

  @Test
  void match_statement_list_subject() {
    setRootRule(PythonGrammar.MATCH_STMT);
    MatchStatement matchStatement = parse("match [x, y]:\n  case 42:...\n", treeMaker::matchStatement);
    Expression subjectExpression = matchStatement.subjectExpression();
    assertThat(subjectExpression.getKind()).isEqualTo(Kind.LIST_LITERAL);
    assertThat(((ListLiteral) subjectExpression).elements().expressions()).hasSize(2);
  }

  @Test
  void case_block_with_guard() {
    setRootRule(PythonGrammar.CASE_BLOCK);
    CaseBlock caseBlock = parse("case 42 if x is None: ...", treeMaker::caseBlock);
    Guard guard = caseBlock.guard();
    assertThat(guard).isNotNull();
    assertThat(guard.ifKeyword().value()).isEqualTo("if");
    assertThat(guard.condition().getKind()).isEqualTo(Kind.IS);
    assertThat(guard.children()).extracting(Tree::getKind)
      .containsExactly(Kind.TOKEN, Kind.IS);
  }

  @Test
  void literal_patterns() {
    assertLiteralPattern(pattern("case \"foo\": ..."), Kind.STRING_LITERAL_PATTERN, "\"foo\"");
    assertLiteralPattern(pattern("case \"foo\" \"bar\": ..."), Kind.STRING_LITERAL_PATTERN, "\"foo\"\"bar\"");
    assertLiteralPattern(pattern("case -42: ..."), Kind.NUMERIC_LITERAL_PATTERN, "-42");
    assertLiteralPattern(pattern("case 3 + 5j: ..."), Kind.NUMERIC_LITERAL_PATTERN, "3+5j");
    assertLiteralPattern(pattern("case None: ..."), Kind.NONE_LITERAL_PATTERN, "None");
    assertLiteralPattern(pattern("case True: ..."), Kind.BOOLEAN_LITERAL_PATTERN, "True");
    assertLiteralPattern(pattern("case False: ..."), Kind.BOOLEAN_LITERAL_PATTERN, "False");
  }

  @Test
  void or_pattern() {
    OrPattern pattern = pattern("case 42 | None | True: ...");
    assertThat(pattern.patterns()).extracting(p -> ((LiteralPattern) p).valueAsString()).containsExactly("42", "None", "True");
    assertThat(pattern.separators()).extracting(Token::value).containsExactly("|", "|");

    AsPattern asPattern = pattern("case 'foo' | 'bar' as x: ...");
    assertThat(asPattern.pattern().getKind()).isEqualTo(Kind.OR_PATTERN);
  }

  @Test
  void as_pattern() {
    AsPattern asPattern = pattern("case \"foo\" as x: ...");
    assertThat(asPattern.pattern()).isInstanceOf(LiteralPattern.class);
    assertThat(asPattern.asKeyword().value()).isEqualTo("as");
    assertThat(asPattern.alias().name().name()).isEqualTo("x");
    assertThat(asPattern.children()).extracting(Tree::getKind).containsExactly(Kind.STRING_LITERAL_PATTERN, Tree.Kind.TOKEN, Kind.CAPTURE_PATTERN);

    asPattern = pattern("case value as x: ...");
    assertThat(asPattern.pattern().getKind()).isEqualTo(Tree.Kind.CAPTURE_PATTERN);
  }

  @Test
  void mapping_pattern() {
    MappingPattern mappingPattern = pattern("case {'x': 'foo', 'y': 'bar'}: ...");
    List<Pattern> keyValuePatternList = mappingPattern.elements();
    assertThat(keyValuePatternList).hasSize(2);
    KeyValuePattern keyValuePattern = (KeyValuePattern) keyValuePatternList.get(0);
    assertThat(keyValuePattern.colon().value()).isEqualTo(":");
    LiteralPattern key = (LiteralPattern) keyValuePattern.key();
    assertThat(key.getKind()).isEqualTo(Kind.STRING_LITERAL_PATTERN);
    assertThat(key.valueAsString()).isEqualTo("'x'");
    LiteralPattern value = (LiteralPattern) keyValuePattern.value();
    assertThat(value.getKind()).isEqualTo(Kind.STRING_LITERAL_PATTERN);
    assertThat(value.valueAsString()).isEqualTo("'foo'");
    assertThat(mappingPattern.lCurlyBrace().value()).isEqualTo("{");
    assertThat(mappingPattern.rCurlyBrace().value()).isEqualTo("}");
    assertThat(mappingPattern.commas()).hasSize(1);
    assertThat(mappingPattern.children())
      .extracting(Tree::getKind)
      .containsExactly(Tree.Kind.TOKEN, Tree.Kind.KEY_VALUE_PATTERN, Tree.Kind.TOKEN, Tree.Kind.KEY_VALUE_PATTERN, Tree.Kind.TOKEN);

    mappingPattern = pattern("case {}: ...");
    keyValuePatternList = mappingPattern.elements();
    assertThat(keyValuePatternList).isEmpty();

    mappingPattern = pattern("case {**d}: ...");
    keyValuePatternList = mappingPattern.elements();
    assertThat(keyValuePatternList).extracting(Tree::getKind).containsExactly(Tree.Kind.DOUBLE_STAR_PATTERN);

    mappingPattern = pattern("case {'x': 'foo', **rest}: ...");
    assertThat(mappingPattern.children()).extracting(Tree::getKind)
      .containsExactly(Tree.Kind.TOKEN, Tree.Kind.KEY_VALUE_PATTERN, Tree.Kind.TOKEN, Tree.Kind.DOUBLE_STAR_PATTERN, Tree.Kind.TOKEN);
    keyValuePatternList = mappingPattern.elements();
    assertThat(keyValuePatternList).extracting(Tree::getKind)
      .containsExactly(Tree.Kind.KEY_VALUE_PATTERN, Tree.Kind.DOUBLE_STAR_PATTERN);
    DoubleStarPattern doubleStarPattern = (DoubleStarPattern) keyValuePatternList.get(1);
    assertThat(doubleStarPattern.capturePattern().name().name()).isEqualTo("rest");
    assertThat(doubleStarPattern.doubleStarToken().value()).isEqualTo("**");

    mappingPattern = pattern("case {a.b: 'foo', c.d.e: 'bar'}: ...");

    KeyValuePattern firstKeyValuePattern = (KeyValuePattern) mappingPattern.elements().get(0);
    assertThat(((LiteralPattern) firstKeyValuePattern.value()).valueAsString()).isEqualTo("'foo'");
    QualifiedExpression firstKeyQualExpr = ((ValuePattern) firstKeyValuePattern.key()).qualifiedExpression();
    assertThat(TreeUtils.nameFromQualifiedExpression(firstKeyQualExpr)).isEqualTo("a.b");
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(firstKeyQualExpr)).contains("a.b");

    KeyValuePattern secondKeyValuePattern = (KeyValuePattern) mappingPattern.elements().get(1);
    assertThat(((LiteralPattern) secondKeyValuePattern.value()).valueAsString()).isEqualTo("'bar'");
    QualifiedExpression secondKeyQualExpr = ((ValuePattern) secondKeyValuePattern.key()).qualifiedExpression();
    assertThat(TreeUtils.nameFromQualifiedExpression(secondKeyQualExpr)).isEqualTo("c.d.e");
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(secondKeyQualExpr)).contains("c.d.e");
  }

  @Test
  void capture_pattern() {
    setRootRule(PythonGrammar.CASE_BLOCK);
    CaseBlock caseBlock = parse("case x: ...", treeMaker::caseBlock);
    CapturePattern capturePattern = (CapturePattern) caseBlock.pattern();
    assertThat(capturePattern.name().name()).isEqualTo("x");
    assertThat(capturePattern.children()).containsExactly(capturePattern.name());
  }

  @Test
  void value_pattern() {
    ValuePattern valuePattern = pattern("case a.b: ...");
    QualifiedExpression qualifiedExpression = valuePattern.qualifiedExpression();
    assertThat(((Name) qualifiedExpression.qualifier()).isVariable()).isTrue();
    assertThat(TreeUtils.nameFromQualifiedExpression(qualifiedExpression)).isEqualTo("a.b");
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(qualifiedExpression)).contains("a.b");
    assertThat(valuePattern.children()).containsExactly(valuePattern.qualifiedExpression());

    valuePattern = pattern("case a.b.c: ...");
    qualifiedExpression = valuePattern.qualifiedExpression();
    assertThat(TreeUtils.nameFromQualifiedExpression(qualifiedExpression)).isEqualTo("a.b.c");
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(qualifiedExpression)).contains("a.b.c");
  }

  @Test
  void sequence_pattern() {
    assertSequenceElements(pattern("case [x, y]: ..."), "x", "y");
    assertSequenceElements(pattern("case (x, y): ..."), "x", "y");
    assertSequenceElements(pattern("case [x]: ..."), "x");
    assertSequenceElements(pattern("case (x,): ..."), "x");
    assertSequenceElements(pattern("case [x, y,]: ..."), "x", "y");
    assertSequenceElements(pattern("case (x, y,): ..."), "x", "y");
    assertThat(((SequencePattern) pattern("case []: ...")).elements()).isEmpty();
    assertThat(((SequencePattern) pattern("case (): ...")).elements()).isEmpty();
    assertSequenceElements(pattern("case ['foo' as head]: ..."), Kind.AS_PATTERN);
  }

  @Test
  void sequence_pattern_delimiters() {
    SequencePattern pattern = pattern("case [x, y]: ...");
    assertThat(pattern.lDelimiter().value()).isEqualTo("[");
    assertThat(pattern.rDelimiter().value()).isEqualTo("]");
    assertThat(pattern.commas()).hasSize(1);

    pattern = pattern("case (x, y): ...");
    assertThat(pattern.lDelimiter().value()).isEqualTo("(");
    assertThat(pattern.rDelimiter().value()).isEqualTo(")");

    pattern = pattern("case x, y: ...");
    assertThat(pattern.lDelimiter()).isNull();
    assertThat(pattern.rDelimiter()).isNull();
  }

  @Test
  void group_pattern() {
    GroupPattern groupPattern = pattern("case (x): ...");
    assertThat(groupPattern.leftPar().value()).isEqualTo("(");
    assertThat(((CapturePattern) groupPattern.pattern()).name().name()).isEqualTo("x");
    assertThat(groupPattern.rightPar().value()).isEqualTo(")");
  }

  @Test
  void wildcard_pattern() {
    WildcardPattern wildcardPattern = pattern("case _: ...");
    assertThat(wildcardPattern.wildcard().value()).isEqualTo("_");
  }

  @Test
  void sequence_pattern_with_star_pattern() {
    SequencePattern sequencePattern = pattern("case [head, *tail]: ...");
    assertSequenceElements(sequencePattern, Kind.CAPTURE_PATTERN, Kind.STAR_PATTERN);

    StarPattern starPattern = (StarPattern) sequencePattern.elements().get(1);
    assertThat(starPattern.starToken().value()).isEqualTo("*");
    assertThat(((CapturePattern) starPattern.pattern()).name().name()).isEqualTo("tail");

    sequencePattern = pattern("case [head, *_]: ...");
    starPattern = (StarPattern) sequencePattern.elements().get(1);
    assertThat(starPattern.pattern().getKind()).isEqualTo(Kind.WILDCARD_PATTERN);
  }

  @Test
  void sequence_pattern_without_parens() {
    assertSequenceElements(pattern("case x, y: ..."), "x", "y");
    assertSequenceElements(pattern("case x,: ..."), "x");
  }

  @Test
  void class_pattern() {
    ClassPattern classPattern = pattern("case A(x, y, z,): ...");
    Name className = (Name) classPattern.targetClass();
    assertThat(className.name()).isEqualTo("A");
    assertThat(className.isVariable()).isTrue();
    assertThat(classPattern.arguments()).extracting(arg -> ((CapturePattern) arg).name().name()).containsExactly("x", "y", "z");
    assertThat(classPattern.argumentSeparators()).hasSize(3);
    assertThat(classPattern.leftPar().value()).isEqualTo("(");
    assertThat(classPattern.rightPar().value()).isEqualTo(")");

    classPattern = pattern("case A.B.C(): ...");
    assertThat(classPattern.arguments()).isEmpty();
    QualifiedExpression qualifiedExpression = (QualifiedExpression) classPattern.targetClass();
    Name qualifierA = (Name) ((QualifiedExpression) qualifiedExpression.qualifier()).qualifier();
    assertThat(qualifierA.isVariable()).isTrue();
    assertThat(TreeUtils.nameFromQualifiedExpression(qualifiedExpression)).isEqualTo("A.B.C");
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(qualifiedExpression)).contains("A.B.C");

    classPattern = pattern("case A(foo='bar'): ...");
    KeywordPattern arg = ((KeywordPattern) classPattern.arguments().get(0));
    assertThat(arg.attributeName().name()).isEqualTo("foo");
    assertThat(((LiteralPattern) arg.pattern()).valueAsString()).isEqualTo("'bar'");
    assertThat(arg.equalToken().value()).isEqualTo("=");

    classPattern = pattern("case A(x, foo='bar', baz=42,): ...");
    assertThat(classPattern.arguments()).extracting(Tree::getKind).containsExactly(Kind.CAPTURE_PATTERN, Kind.KEYWORD_PATTERN, Kind.KEYWORD_PATTERN);
    assertThat(classPattern.argumentSeparators()).hasSize(3);
  }

  @Test
  void class_pattern_positional_after_keyword() {
    try {
      pattern("case A(foo=42, x, y): ...");
      fail("Position patterns cannot follow keyword patterns");
    } catch (RecognitionException exception) {
      assertThat(exception.getMessage()).isEqualTo("Parse error at line 1: Positional patterns follow keyword patterns.");
    }
  }

  private void assertSequenceElements(SequencePattern sequencePattern, String... elements) {
    assertThat(sequencePattern.elements()).extracting(t -> ((CapturePattern) t).name().name()).containsExactly(elements);
  }

  private void assertSequenceElements(SequencePattern sequencePattern, Kind... elements) {
    assertThat(sequencePattern.elements()).extracting(Tree::getKind).containsExactly(elements);
  }

  private void assertLiteralPattern(LiteralPattern literalPattern, Kind literalKind, String valueAsString) {
    assertThat(literalPattern.getKind()).isEqualTo(literalKind);
    assertThat(literalPattern.valueAsString()).isEqualTo(valueAsString);
  }

  @SuppressWarnings("unchecked")
  private <T extends Pattern> T pattern(String code) {
    setRootRule(PythonGrammar.CASE_BLOCK);
    CaseBlock caseBlock = parse(code, treeMaker::caseBlock);
    return (T) caseBlock.pattern();
  }
}
