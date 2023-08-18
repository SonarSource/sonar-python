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
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.utils.CodeTestUtils.code;

public class TrailingCommentCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/trailingComment.py", new TrailingCommentCheck());
  }

  @Test
  public void testQuickFixSimple() {
    String code = "print(1) # More than one word";
    String fixedCode = code(
      "# More than one word",
      "print(1)"
      );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixMoreCodeAfter() {
    String code = code(
      "print(1) # More Words",
      "print(\"aaa\")"
    );
    String fixedCode = code(
      "# More Words",
      "print(1)",
      "print(\"aaa\")"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixInIfCondition() {
    String code = code(
      "def func(self):",
        "    if a==b: #Some comment",
        "        return None",
        "    self.bar()"
    );
    String fixedCode = code(
      "def func(self):",
      "    #Some comment",
      "    if a==b:",
      "        return None",
      "    self.bar()"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixInIfReturn() {
    String code = code(
      "def func(self):",
        "    if a==b:",
        "        return None #This should not happen",
        "    self.bar()"
    );
    String fixedCode = code(
      "def func(self):",
      "    if a==b:",
      "        #This should not happen",
      "        return None",
      "    self.bar()"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixInArray() {
    String code = code(
      "SOMEVAR = [",
        "  'asd', 'asd', # comment more than one word",
        "  'asdpj'",
        "]"
    );
    String fixedCode = code(
      "SOMEVAR = [",
      "  # comment more than one word",
      "  'asd', 'asd',",
      "  'asdpj'",
      "]"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixArgumentsMultiline() {
    String code = code(
      "changes = sa.Table('foo', bar,",
        "    sa.Column('id', sa.Integer), # yet another comment",
        "    # some comment",
        "    sa.Column('name', sa.String(256)),",
        "    )"
    );
    String fixedCode = code(
      "changes = sa.Table('foo', bar,",
      "    # yet another comment",
      "    sa.Column('id', sa.Integer),",
      "    # some comment",
      "    sa.Column('name', sa.String(256)),",
      "    )"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixInTuple() {
    String code = code(
      "toto = (state, None, # a comment",
        "        [foo(b) for b in bar])"
    );
    String fixedCode = code(
      "# a comment",
      "toto = (state, None,",
      "        [foo(b) for b in bar])"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixInKeyWordArguments() {
    String code = code(
      "foo.bar(",
        "         SomeClass(workdir='wkdir',",
        "                      command=['cmd',",
        "                               'foo'], # note extra param",
        "                      env=some.method(",
        "                          r'sf',",
        "                          l='l', p='p', i='i'))",
        "          + 0",
        "      )"
    );
    String fixedCode = code(
      "foo.bar(",
      "         SomeClass(workdir='wkdir',",
      "                      command=['cmd',",
      "                               # note extra param",
      "                               'foo'],",
      "                      env=some.method(",
      "                          r'sf',",
      "                          l='l', p='p', i='i'))",
      "          + 0",
      "      )"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }
  
  @Test
  public void testQuickFixInIfAsArgument() {
    String code = code(
      "var = (some.method(arg1, arg2)",
        "        if hasattr(umath, 'nextafter')  # Missing on some platforms?",
        "        else float64_ma.huge)"
    );
    String fixedCode = code(
      "var = (some.method(arg1, arg2)",
      "        # Missing on some platforms?",
      "        if hasattr(umath, 'nextafter')",
      "        else float64_ma.huge)"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }
  @Test
  public void testQuickFixInWithStatement() {
    String code = code(
      "with errstate(over='ignore'): #some comment",
        "    if bar:",
        "        print(\"hello\")"
    );
    String fixedCode = code(
      "#some comment",
      "with errstate(over='ignore'):",
      "    if bar:",
      "        print(\"hello\")"
    );
    PythonQuickFixVerifier.verify(new TrailingCommentCheck(), code, fixedCode);
  }
}
