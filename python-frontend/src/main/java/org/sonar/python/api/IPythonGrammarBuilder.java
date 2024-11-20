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
package org.sonar.python.api;

import org.sonar.sslr.grammar.LexerfulGrammarBuilder;

import static com.sonar.sslr.api.GenericTokenType.EOF;
import static org.sonar.python.api.IPythonGrammar.CELL;
import static org.sonar.python.api.IPythonGrammar.CELL_MAGIC_STATEMENT;
import static org.sonar.python.api.IPythonGrammar.DYNAMIC_OBJECT_INFO_STATEMENT;
import static org.sonar.python.api.IPythonGrammar.LINE_MAGIC;
import static org.sonar.python.api.IPythonGrammar.LINE_MAGIC_STATEMENT;
import static org.sonar.python.api.IPythonGrammar.MAGIC_CELL;
import static org.sonar.python.api.PythonGrammar.ANNOTATED_RHS;
import static org.sonar.python.api.PythonGrammar.ASSERT_STMT;
import static org.sonar.python.api.PythonGrammar.BREAK_STMT;
import static org.sonar.python.api.PythonGrammar.CONTINUE_STMT;
import static org.sonar.python.api.PythonGrammar.DEL_STMT;
import static org.sonar.python.api.PythonGrammar.EXEC_STMT;
import static org.sonar.python.api.PythonGrammar.EXPRESSION_STMT;
import static org.sonar.python.api.PythonGrammar.FILE_INPUT;
import static org.sonar.python.api.PythonGrammar.GLOBAL_STMT;
import static org.sonar.python.api.PythonGrammar.IMPORT_STMT;
import static org.sonar.python.api.PythonGrammar.NONLOCAL_STMT;
import static org.sonar.python.api.PythonGrammar.PASS_STMT;
import static org.sonar.python.api.PythonGrammar.PRINT_STMT;
import static org.sonar.python.api.PythonGrammar.RAISE_STMT;
import static org.sonar.python.api.PythonGrammar.RETURN_STMT;
import static org.sonar.python.api.PythonGrammar.SIMPLE_STMT;
import static org.sonar.python.api.PythonGrammar.STATEMENT;
import static org.sonar.python.api.PythonGrammar.TEST;
import static org.sonar.python.api.PythonGrammar.TESTLIST_STAR_EXPR;
import static org.sonar.python.api.PythonGrammar.YIELD_EXPR;
import static org.sonar.python.api.PythonGrammar.YIELD_STMT;
import static org.sonar.python.api.PythonTokenType.IPYNB_CELL_DELIMITER;
import static org.sonar.python.api.PythonTokenType.NEWLINE;

public class IPythonGrammarBuilder extends PythonGrammarBuilder {

  public static final String QUESTION_MARK = "?";

  @Override
  protected void fileInput(LexerfulGrammarBuilder b) {
    b.rule(FILE_INPUT).is(b.zeroOrMore(b.firstOf(CELL, MAGIC_CELL, NEWLINE, IPYNB_CELL_DELIMITER)), EOF);
  }

  @Override
  protected void setupRules(LexerfulGrammarBuilder b) {
    iPythonRules(b);
    super.setupRules(b);
  }

  protected void iPythonRules(LexerfulGrammarBuilder b) {
    b.rule(CELL).is(b.oneOrMore(b.firstOf(NEWLINE, STATEMENT)));
    b.rule(MAGIC_CELL).is(CELL_MAGIC_STATEMENT);
    b.rule(CELL_MAGIC_STATEMENT).is(PythonPunctuator.MOD, PythonPunctuator.MOD, b.zeroOrMore(b.anyTokenButNot(b.firstOf(IPYNB_CELL_DELIMITER, EOF))));
    b.rule(LINE_MAGIC_STATEMENT).is(LINE_MAGIC);
    b.rule(LINE_MAGIC).is(
      b.firstOf(PythonPunctuator.MOD, "!", PythonPunctuator.DIV),
      b.anyTokenButNot(PythonPunctuator.MOD),
      b.zeroOrMore(b.anyTokenButNot(b.firstOf(NEWLINE, b.next(EOF))))
    );
    b.rule(DYNAMIC_OBJECT_INFO_STATEMENT).is(b.firstOf(
      b.sequence(
        QUESTION_MARK,
        b.optional(QUESTION_MARK),
        TEST,
        b.optional(QUESTION_MARK),
        b.optional(QUESTION_MARK)
      ),
      b.sequence(
        TEST,
        QUESTION_MARK,
        b.optional(QUESTION_MARK)
      ),
      QUESTION_MARK
    ));
  }

  @Override
  protected void simpleStatement(LexerfulGrammarBuilder b) {
    b.rule(SIMPLE_STMT).is(b.firstOf(
      DYNAMIC_OBJECT_INFO_STATEMENT,
      PRINT_STMT,
      EXEC_STMT,
      EXPRESSION_STMT,
      ASSERT_STMT,
      PASS_STMT,
      DEL_STMT,
      RETURN_STMT,
      YIELD_STMT,
      RAISE_STMT,
      BREAK_STMT,
      CONTINUE_STMT,
      IMPORT_STMT,
      GLOBAL_STMT,
      NONLOCAL_STMT,
      LINE_MAGIC_STATEMENT));
  }

  @Override
  protected void annotatedRhs(LexerfulGrammarBuilder b) {
    b.rule(ANNOTATED_RHS).is(b.firstOf(LINE_MAGIC, YIELD_EXPR, TESTLIST_STAR_EXPR));
  }
}
