/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ShallowCopyEnvironCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/shallowCopyEnviron.py", new ShallowCopyEnvironCheck());
  }

  @Test
  void quickFix() {
    String before = """
      import copy
      import os
      env = copy.copy(os.environ)
      """;
    String after = """
      import copy
      import os
      env = os.environ.copy()
      """;
    PythonQuickFixVerifier.verify(new ShallowCopyEnvironCheck(), before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(new ShallowCopyEnvironCheck(), before, "Replace with \"os.environ.copy()\"");
  }

  @Test
  void quickFixFromCopyImport() {
    String before = """
      import os
      from copy import copy
      env = copy(os.environ)
      """;
    String after = """
      import os
      from copy import copy
      env = os.environ.copy()
      """;
    PythonQuickFixVerifier.verify(new ShallowCopyEnvironCheck(), before, after);
  }

  @Test
  void quickFixKeywordArgument() {
    String before = """
      import copy
      import os
      env = copy.copy(x=os.environ)
      """;
    String after = """
      import copy
      import os
      env = os.environ.copy()
      """;
    PythonQuickFixVerifier.verify(new ShallowCopyEnvironCheck(), before, after);
  }

  @Test
  void noQuickFixForMultilineArgument() {
    String before = """
      import copy
      import os
      env = copy.copy((os
                       .environ))
      """;
    PythonQuickFixVerifier.verifyNoQuickFixes(new ShallowCopyEnvironCheck(), before);
  }
}
