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

import com.sonar.sslr.api.Grammar;
import org.sonar.sslr.grammar.GrammarRuleKey;
import org.sonar.sslr.grammar.LexerfulGrammarBuilder;

import static com.sonar.sslr.api.GenericTokenType.EOF;
import static org.sonar.python.api.PythonGrammar.FILE_INPUT;
import static org.sonar.python.api.PythonGrammar.STATEMENT;
import static org.sonar.python.api.PythonGrammar.compoundStatements;
import static org.sonar.python.api.PythonGrammar.expressions;
import static org.sonar.python.api.PythonGrammar.grammar;
import static org.sonar.python.api.PythonGrammar.simpleStatements;
import static org.sonar.python.api.PythonTokenType.NEWLINE;

public enum IPythonGrammar implements GrammarRuleKey {

  LINE_MAGIC_COMMAND;

  public static Grammar create() {
    LexerfulGrammarBuilder b = LexerfulGrammarBuilder.create();

    b.rule(FILE_INPUT).is(b.zeroOrMore(b.firstOf(NEWLINE, STATEMENT)), EOF);

    grammar(b);
    compoundStatements(b);
    simpleStatements(b);
    magicCommands(b);
    expressions(b);

    b.setRootRule(FILE_INPUT);
    return b.buildWithMemoizationOfMatchesForAllRules();
  }

  public static void magicCommands(LexerfulGrammarBuilder b) {
    b.rule(LINE_MAGIC_COMMAND).is(PythonPunctuator.MOD, b.anyTokenButNot(PythonPunctuator.MOD), b.tillNewLine());
  }
}
