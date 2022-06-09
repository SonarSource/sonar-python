/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import org.junit.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class AllBranchesAreIdenticalCheckTest {

  private PythonCheck check = new AllBranchesAreIdenticalCheck();
  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/allBranchesAreIdentical.py", check);
  }

  @Test
  public void quickfix_one_statement() {
    String codeWithIssue =
      "def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "    else:\n" +
      "        doSomething()\n";
    String codeFixed =
      "def func():\n" +
      "    doSomething()\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }
  @Test
  public void quickfix_semicolons() {
    String codeWithIssue =
      "def func():\n" +
        "    if b == 0:\n" +
        "        doSomething(); doOneMoreThing()\n"+
        "    else:\n" +
        "        doSomething(); doOneMoreThing()\n";
    String codeFixed =
      "def func():\n" +
        "    doSomething(); doOneMoreThing()\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void if_enclosed() {
    String codeWithIssue =
      "def func():\n" +
        "    if b == 0:\n" +
        "        if a == 1:\n"+
        "            doSomething()\n"+
        "    else:\n" +
        "        if a == 1:\n"+
        "            doSomething()\n";
    String codeFixed =
      "def func():\n" +
      "    if a == 1:\n"+
      "        doSomething()\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }


  @Test
  public void oneline(){
    String codeWithIssue = "a = 1 if x else 1";
    String codeFixed = "a = 1";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void test_multiple_statement(){
    String codeWithIssue ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n" +
      "    else:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n";
    String codeFixed = "def func():\n" +
      "    doSomething()\n" +
      "    doOneMoreThing()\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void lambda(){
    String codeWithIssue = "a = (lambda x: x+1\n" +
      "     if x > 0 # Noncompliant\n" +
      "     else x+1)";
    String codeFixed = "a = (lambda x: x+1)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void mutiple_conditional_statements(){
    String codeWithIssue = "a = 1 if x else 1 if y else 1 if z else 1";
    String codeFixed = "a = 1";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void test_elseif(){
    String codeWithIssue ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "    elif b == 1:\n" +
      "        doSomething()\n" +
      "    else:\n" +
      "        doSomething()\n";
    String codeFixed = "def func():\n" +
      "    doSomething()\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void test_elseif_multiple(){
    String codeWithIssue ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    elif b == 1:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    else:\n" +
      "        doSomething()\n"+
      "        doOneMoreThing()\n";;
    String codeFixed = "def func():\n" +
      "    doSomething()\n"+
      "    doOneMoreThing()\n";;
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void test_elseif_more(){
    String codeWithIssue ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    elif b == 1:\n" +
      "        doSomething()\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    else:\n" +
      "        doSomething()\n"+
      "        doSomething()\n" +
      "        doOneMoreThing()\n";;
    String codeFixed = "def func():\n" +
      "    doSomething()\n"+
      "    doSomething()\n" +
      "    doOneMoreThing()\n";;
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }
}
