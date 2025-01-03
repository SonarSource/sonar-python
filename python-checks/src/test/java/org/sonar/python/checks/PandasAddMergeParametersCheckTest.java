/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class PandasAddMergeParametersCheckTest {
  public static final PandasAddMergeParametersCheck CHECK = new PandasAddMergeParametersCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/pandasAddMergeParameters.py", CHECK);
  }

  @Test
  void quickfix_test_1() {
    final String non_compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = age_df.merge(name_df)";

    final String compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = age_df.merge(name_df, how=\"inner\", on=None, validate=\"many_to_many\")";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(CHECK, non_compliant, "Add the missing parameters");
  }

  @Test
  void quickfix_test_2() {
    final String non_compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = age_df.join(name_df)";

    final String compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = age_df.join(name_df, how=\"left\", on=None, validate=\"many_to_many\")";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(CHECK, non_compliant, "Add the missing parameters");
  }

  @Test
  void quickfix_test_3() {
    final String non_compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = age_df.merge(name_df, on=\"user_id\")";

    final String compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = age_df.merge(name_df, on=\"user_id\", how=\"inner\", validate=\"many_to_many\")";

    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(CHECK, non_compliant, "Add the missing parameters");
  }


  @Test
  void quickfix_test_4() {
    final String non_compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = pd.merge(\n" +
      "          age_df, \n" +
      "          name_df, \n" +
      "          on=\"user_id\")";

    final String compliant = "def non_compliant_merge_1():\n" +
      "    import pandas as pd\n" +
      "\n" +
      "    age_df = pd.read_csv(\"age_csv.csv\")\n" +
      "    name_df = pd.read_csv(\"name_csv.csv\")\n" +
      "\n" +
      "    _ = pd.merge(\n" +
      "          age_df, \n" +
      "          name_df, \n" +
      "          on=\"user_id\", how=\"inner\", validate=\"many_to_many\")";

    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(CHECK, non_compliant, "Add the missing parameters");
  }

}
