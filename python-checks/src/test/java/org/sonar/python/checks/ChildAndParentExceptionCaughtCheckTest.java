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

import org.junit.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class ChildAndParentExceptionCaughtCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/childAndParentExceptionCaughtCheck.py", new ChildAndParentExceptionCaughtCheck());
  }

  @Test
  public void childWithParentQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = "def child_with_parent():\n" +
      "  try:\n" +
      "      raise NotImplementedError()\n" +
      "  except (RuntimeError, RecursionError):\n" +
      "      print(\"Foo\")";
    String after = "def child_with_parent():\n" +
      "  try:\n" +
      "      raise NotImplementedError()\n" +
      "  except (RuntimeError):\n" +
      "      print(\"Foo\")";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  public void parentWithChildQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = "def parent_with_child():\n" +
      "  try:\n" +
      "      raise NotImplementedError()\n" +
      "  except (RecursionError, RuntimeError):\n" +
      "      print(\"Foo\")";
    String after = "def parent_with_child():\n" +
      "  try:\n" +
      "      raise NotImplementedError()\n" +
      "  except (RuntimeError):\n" +
      "      print(\"Foo\")";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  public void duplicateExceptionQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = "def duplicate_exception_caught():\n" +
      "  try:\n" +
      "      raise NotImplementedError()\n" +
      "  except (RuntimeError, RuntimeError):\n" +
      "      print(\"Foo\")";
    String after = "def duplicate_exception_caught():\n" +
      "  try:\n" +
      "      raise NotImplementedError()\n" +
      "  except (RuntimeError):\n" +
      "      print(\"Foo\")";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }
}
