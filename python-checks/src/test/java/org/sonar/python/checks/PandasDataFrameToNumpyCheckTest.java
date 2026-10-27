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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class PandasDataFrameToNumpyCheckTest {

  PandasDataFrameToNumpyCheck check = new PandasDataFrameToNumpyCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/pandasDataFrameToNumpy.py", check);
  }

  @Test
  void does_not_raise_on_the_dataframe_implementation_itself() {
    PythonCheckVerifier.verifyNoIssue(List.of(
      "src/test/resources/checks/s6741/package_root.py",
      "src/test/resources/checks/s6741/pandas/__init__.py",
      "src/test/resources/checks/s6741/pandas/core/__init__.py",
      "src/test/resources/checks/s6741/pandas/core/frame.py"), check);
  }

  @Test
  void quick_fix_test_1() {
    final String non_compliant = """
      def non_compliant_1(xx):
          import pandas as pd
      
          df = pd.DataFrame({
              'X': ['A', 'B', 'A', 'C'],
              'Y': [10, 7, 12, 5]
          })
      
          _ = df.values""";
    final String compliant = """
      def non_compliant_1(xx):
          import pandas as pd
      
          df = pd.DataFrame({
              'X': ['A', 'B', 'A', 'C'],
              'Y': [10, 7, 12, 5]
          })
      
          _ = df.to_numpy()""";
    performVerification(non_compliant, compliant);
  }

  @Test
  void quick_fix_test_2() {

    final String non_compliant = """
      def non_compliant_1(xx):
          import pandas as pd
      
          _ = pd.DataFrame({
              'X': ['A', 'B', 'A', 'C'],
              'Y': [10, 7, 12, 5]
          }).values""";
    final String compliant = """
      def non_compliant_1(xx):
          import pandas as pd
      
          _ = pd.DataFrame({
              'X': ['A', 'B', 'A', 'C'],
              'Y': [10, 7, 12, 5]
          }).to_numpy()""";
    performVerification(non_compliant, compliant);
  }

  @Test
  void quick_fix_test_3() {

    final String non_compliant = """
      def dataframe_from_read_csv():
          import pandas as pd
      
          my_df = pd.read_csv("some_csv.csv")
          my_df.values.astype(str)""";
    final String compliant = """
      def dataframe_from_read_csv():
          import pandas as pd
      
          my_df = pd.read_csv("some_csv.csv")
          my_df.to_numpy().astype(str)""";
    performVerification(non_compliant, compliant);
  }

  private void performVerification(String non_compliant, String compliant) {
    PythonQuickFixVerifier.verify(check, non_compliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, non_compliant, "Replace with \"DataFrame.to_numpy()\"");
  }
}
