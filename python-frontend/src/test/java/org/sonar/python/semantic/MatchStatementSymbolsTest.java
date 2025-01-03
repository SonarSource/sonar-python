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
package org.sonar.python.semantic;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AsPattern;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.ClassPattern;
import org.sonar.plugins.python.api.tree.DoubleStarPattern;
import org.sonar.plugins.python.api.tree.GroupPattern;
import org.sonar.plugins.python.api.tree.KeyValuePattern;
import org.sonar.plugins.python.api.tree.KeywordPattern;
import org.sonar.plugins.python.api.tree.MappingPattern;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.OrPattern;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SequencePattern;
import org.sonar.plugins.python.api.tree.StarPattern;
import org.sonar.plugins.python.api.tree.ValuePattern;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.api.symbols.Usage.Kind.CLASS_DECLARATION;
import static org.sonar.plugins.python.api.symbols.Usage.Kind.OTHER;
import static org.sonar.plugins.python.api.symbols.Usage.Kind.PATTERN_DECLARATION;
import static org.sonar.plugins.python.api.tree.Tree.Kind.CASE_BLOCK;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.python.PythonTestUtils.getLastDescendant;

class MatchStatementSymbolsTest {

  @Test
  void capture_pattern() {
    CapturePattern capturePattern = patternFromCase("case others: print(others)");
    Symbol others = capturePattern.name().symbol();
    assertThat(others.name()).isEqualTo("others");
    assertThat(others.fullyQualifiedName()).isNull();
    assertThat(others.usages()).extracting(Usage::kind).containsExactly(PATTERN_DECLARATION, OTHER);
  }

  @Test
  void wildcard_pattern() {
    Name wildcard = getLastDescendant(PythonTestUtils.parse(
      "def foo(value):",
      "  match(value):",
      "    case _: print(_)"
    ), t -> t.is(NAME) && ((Name) t).name().equals("_"));
    assertThat(wildcard.symbol()).isNull();
  }

  @Test
  void class_pattern() {
    ClassPattern classPattern = pattern(
      "class A:",
      "  foo = 42",
      "def foo(value):",
      "  match(value):",
      "    case A(foo=x):",
      "      print(x)");
    Name name = (Name) classPattern.targetClass();
    assertThat(name.symbol().kind()).isEqualTo(Symbol.Kind.CLASS);
    KeywordPattern keywordPattern = ((KeywordPattern) classPattern.arguments().get(0));
    Symbol x = ((CapturePattern) keywordPattern.pattern()).name().symbol();
    assertThat(x.usages()).extracting(Usage::kind).containsExactly(PATTERN_DECLARATION, OTHER);

    classPattern = pattern(
      "import mod",
      "def foo(value):",
      "  match(value):",
      "    case mod.A(foo=x):",
      "      print(x)");
    QualifiedExpression qualifiedExpression = (QualifiedExpression) classPattern.targetClass();
    assertThat(qualifiedExpression.symbol().kind()).isEqualTo(Symbol.Kind.OTHER);
  }

  @Test
  void value_pattern() {
    ValuePattern valuePattern = pattern(
      "import command",
      "def foo(value):",
      "  match(value):",
      "    case command.QUIT:",
      "      print(x)");
    assertThat(valuePattern.qualifiedExpression().symbol().kind()).isEqualTo(Symbol.Kind.OTHER);
  }

  @Test
  void as_pattern() {
    AsPattern asPattern = patternFromCase("case 42 as x: ...");
    Symbol symbol = asPattern.alias().name().symbol();
    assertThat(symbol.name()).isEqualTo("x");
    assertThat(symbol.fullyQualifiedName()).isNull();
    assertThat(symbol.usages()).extracting(Usage::kind).containsExactly(PATTERN_DECLARATION);

    asPattern = patternFromCase("case z as x: return x + z");
    Symbol z = ((CapturePattern) asPattern.pattern()).name().symbol();
    Symbol x = asPattern.alias().name().symbol();
    assertThat(z.name()).isEqualTo("z");
    assertThat(x.name()).isEqualTo("x");
    assertThat(z.usages()).extracting(Usage::kind).containsExactly(PATTERN_DECLARATION, OTHER);
    assertThat(x.usages()).extracting(Usage::kind).containsExactly(PATTERN_DECLARATION, OTHER);
  }

  @Test
  void or_pattern() {
    OrPattern orPattern = pattern(
      "class A: ...",
      "class B: ...",
      "def foo(value):",
      "  match(value):",
      "    case A() | B(): ...");
    Symbol symbolA = ((Name) ((ClassPattern) orPattern.patterns().get(0)).targetClass()).symbol();
    Symbol symbolB = ((Name) ((ClassPattern) orPattern.patterns().get(1)).targetClass()).symbol();
    assertThat(symbolA.usages()).extracting(Usage::kind).containsExactly(CLASS_DECLARATION, OTHER);
    assertThat(symbolB.usages()).extracting(Usage::kind).containsExactly(CLASS_DECLARATION, OTHER);
  }

  @Test
  void sequence_pattern() {
    SequencePattern sequencePattern = patternFromCase("case [42, x, *others]: ...");
    CapturePattern capturePattern = (CapturePattern) sequencePattern.elements().get(1);
    StarPattern starPattern = (StarPattern) sequencePattern.elements().get(2);
    assertThat(capturePattern.name().symbol()).isNotNull();
    assertThat(((CapturePattern) starPattern.pattern()).name().symbol()).isNotNull();
  }

  @Test
  void mapping_pattern() {
    MappingPattern mappingPattern = patternFromCase("case {'x': 'foo', 'y': val, **others}: ...");
    KeyValuePattern keyValuePattern = (KeyValuePattern) mappingPattern.elements().get(1);
    CapturePattern capturePattern = (CapturePattern) keyValuePattern.value();
    DoubleStarPattern doubleStarPattern = (DoubleStarPattern) mappingPattern.elements().get(2);
    assertThat(capturePattern.name().symbol()).isNotNull();
    assertThat(doubleStarPattern.capturePattern().name().symbol()).isNotNull();
  }

  @Test
  void group_pattern() {
    GroupPattern groupPattern = patternFromCase("case (x): ...");
    CapturePattern capturePattern = (CapturePattern) groupPattern.pattern();
    assertThat(capturePattern.name().symbol()).isNotNull();
  }

  @Test
  void guard() {
    CapturePattern capturePattern = patternFromCase("case x if x > 42: ...");
    assertThat(capturePattern.name().symbol().usages()).extracting(Usage::kind).containsExactly(PATTERN_DECLARATION, OTHER);
  }

  @SuppressWarnings("unchecked")
  private static <T extends Pattern> T pattern(String... lines) {
    String code = String.join(System.getProperty("line.separator"), lines);
    CaseBlock caseBlock = getLastDescendant(PythonTestUtils.parse(code), t -> t.is(CASE_BLOCK));
    return ((T) caseBlock.pattern());
  }

  @SuppressWarnings("unchecked")
  private static <T extends Pattern> T patternFromCase(String code) {
    CaseBlock caseBlock = getLastDescendant(PythonTestUtils.parse(
      "def foo(value):",
      "  match(value):",
      "    " + code
    ), t -> t.is(CASE_BLOCK));
    return (T) caseBlock.pattern();
  }
}
