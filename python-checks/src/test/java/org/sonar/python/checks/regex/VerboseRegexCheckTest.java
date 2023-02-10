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
package org.sonar.python.checks.regex;

import org.junit.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class VerboseRegexCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/regex/verboseRegexCheck.py", new VerboseRegexCheck());
  }

  @Test
  public void dotReplacementQuickFixTest() {
    var before = "import re\n" +
      "\n" +
      "def non_compliant(input):\n" +
      "    re.match(r\"[\\s\\S]\", input, re.DOTALL)";
    var after = "import re\n" +
      "\n" +
      "def non_compliant(input):\n" +
      "    re.match(r\".\", input, re.DOTALL)";
    var check = new VerboseRegexCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, String.format(VerboseRegexCheck.QUICK_FIX_FORMAT, "."));
  }

  @Test
  public void digitReplacementQuickFixTest() {
    var before = "import re\n" +
      "\n" +
      "def non_compliant(input):\n" +
      "    re.match(r\"foo[0-9]barr\", input)";
    var after = "import re\n" +
      "\n" +
      "def non_compliant(input):\n" +
      "    re.match(r\"foo\\dbarr\", input)";
    var check = new VerboseRegexCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, String.format(VerboseRegexCheck.QUICK_FIX_FORMAT, "\\d"));
  }

  @Test
  public void plusReplacementQuickFixTest() {
    var before = "import re\n" +
      "\n" +
      "def non_compliant(input):\n" +
      "    re.match(r\"x{1,}\", input)";
    var after = "import re\n" +
      "\n" +
      "def non_compliant(input):\n" +
      "    re.match(r\"x+\", input)";
    var check = new VerboseRegexCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, String.format(VerboseRegexCheck.QUICK_FIX_FORMAT, "+"));
  }
}
