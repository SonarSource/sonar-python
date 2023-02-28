/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.api;

import org.sonar.sslr.grammar.LexerfulGrammarBuilder;

import static com.sonar.sslr.api.GenericTokenType.EOF;
import static org.sonar.python.api.IPythonGrammar.LINE_MAGIC_STATEMENT;
import static org.sonar.python.api.IPythonGrammar.LINE_MAGIC;
import static org.sonar.python.api.PythonGrammar.ASSERT_STMT;
import static org.sonar.python.api.PythonGrammar.ASSIGNMENT_VALUE;
import static org.sonar.python.api.PythonGrammar.BREAK_STMT;
import static org.sonar.python.api.PythonGrammar.CONTINUE_STMT;
import static org.sonar.python.api.PythonGrammar.DEL_STMT;
import static org.sonar.python.api.PythonGrammar.EXEC_STMT;
import static org.sonar.python.api.PythonGrammar.EXPRESSION_STMT;
import static org.sonar.python.api.PythonGrammar.GLOBAL_STMT;
import static org.sonar.python.api.PythonGrammar.IMPORT_STMT;
import static org.sonar.python.api.PythonGrammar.NAME;
import static org.sonar.python.api.PythonGrammar.NONLOCAL_STMT;
import static org.sonar.python.api.PythonGrammar.PASS_STMT;
import static org.sonar.python.api.PythonGrammar.PRINT_STMT;
import static org.sonar.python.api.PythonGrammar.RAISE_STMT;
import static org.sonar.python.api.PythonGrammar.RETURN_STMT;
import static org.sonar.python.api.PythonGrammar.SIMPLE_STMT;
import static org.sonar.python.api.PythonGrammar.TESTLIST_STAR_EXPR;
import static org.sonar.python.api.PythonGrammar.YIELD_EXPR;
import static org.sonar.python.api.PythonGrammar.YIELD_STMT;
import static org.sonar.python.api.PythonTokenType.NEWLINE;

public class IPythonGrammarBuilder extends PythonGrammarBuilder {

  @Override
  protected void setupRules(LexerfulGrammarBuilder b) {
    lineMagic(b);
    super.setupRules(b);
  }

  protected void lineMagic(LexerfulGrammarBuilder b) {
    b.rule(LINE_MAGIC_STATEMENT).is(LINE_MAGIC);
    b.rule(LINE_MAGIC).is(
      b.firstOf("%", "!", "/"),
      NAME,
      b.zeroOrMore(b.anyTokenButNot(b.firstOf(NEWLINE, b.next(EOF))))
    );
  }

  @Override
  protected void simpleStatement(LexerfulGrammarBuilder b) {
    b.rule(SIMPLE_STMT).is(b.firstOf(
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
  protected void assignmentValue(LexerfulGrammarBuilder b) {
    b.rule(ASSIGNMENT_VALUE).is(b.firstOf(LINE_MAGIC, YIELD_EXPR, TESTLIST_STAR_EXPR));
  }
}
