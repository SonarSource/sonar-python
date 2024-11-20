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
package org.sonar.python.parser;

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.RecognitionException;
import com.sonar.sslr.impl.Parser;
import com.sonar.sslr.impl.matcher.RuleDefinition;
import org.fest.assertions.GenericAssert;
import org.sonar.sslr.grammar.GrammarRuleKey;
import org.sonar.sslr.internal.vm.EndOfInputExpression;
import org.sonar.sslr.internal.vm.FirstOfExpression;
import org.sonar.sslr.internal.vm.lexerful.TokenTypeExpression;
import org.sonar.sslr.tests.Assertions;
import org.sonar.sslr.tests.ParsingResultComparisonFailure;

// Mostly copied from SSLR: org.sonar.sslr.tests.ParserAssert
// We couldn't use org.sonar.sslr.tests.ParserAssert because it creates a new instance of Parser
// and SonarPython's parser now has its own Parser class.
public class PythonParserAssert extends GenericAssert<PythonParserAssert, PythonParser> {

  public static PythonParserAssert assertThat(PythonParser actual) {
    return new PythonParserAssert(actual);
  }

  public PythonParserAssert(PythonParser actual) {
    super(PythonParserAssert.class, actual);
  }

  private PythonParser createParserWithEofMatcher() {
    RuleDefinition rule = actual.getRootRule();
    RuleDefinition endOfInput = new RuleDefinition(new EndOfInput())
      .is(new FirstOfExpression(EndOfInputExpression.INSTANCE, new TokenTypeExpression(GenericTokenType.EOF)));
    RuleDefinition withEndOfInput = new RuleDefinition(new WithEndOfInput(actual.getRootRule().getRuleKey()))
      .is(rule, endOfInput);

    PythonParser parser = PythonParser.create();
    parser.setRootRule(withEndOfInput);

    return parser;
  }

  /**
   * Verifies that the actual <code>{@link Parser}</code> fully matches a given input.
   * @return this assertion object.
   */
  public PythonParserAssert matches(String input) {
    isNotNull();
    hasRootRule();
    PythonParser parser = createParserWithEofMatcher();
    String expected = "Rule '" + getRuleName() + "' should match:\n" + input;
    try {
      parser.parse(input);
    } catch (RecognitionException e) {
      String actual = e.getMessage();
      throw new ParsingResultComparisonFailure(expected, actual);
    }
    return this;
  }

  /**
   * Verifies that the actual <code>{@link Parser}</code> not matches a given input.
   * @return this assertion object.
   */
  public PythonParserAssert notMatches(String input) {
    isNotNull();
    hasRootRule();
    PythonParser parser = createParserWithEofMatcher();
    try {
      parser.parse(input);
    } catch (RecognitionException e) {
      // expected
      return this;
    }
    throw new AssertionError("Rule '" + getRuleName() + "' should not match:\n" + input);
  }

  private void hasRootRule() {
    Assertions.assertThat(actual.getRootRule())
      .overridingErrorMessage("Root rule of the parser should not be null")
      .isNotNull();
  }

  private String getRuleName() {
    return actual.getRootRule().getName();
  }

  static class EndOfInput implements GrammarRuleKey {
    @Override
    public String toString() {
      return "end of input";
    }
  }

  static class WithEndOfInput implements GrammarRuleKey {
    private final GrammarRuleKey ruleKey;

    public WithEndOfInput(GrammarRuleKey ruleKey) {
      this.ruleKey = ruleKey;
    }

    @Override
    public String toString() {
      return ruleKey + " with end of input";
    }
  }


}
