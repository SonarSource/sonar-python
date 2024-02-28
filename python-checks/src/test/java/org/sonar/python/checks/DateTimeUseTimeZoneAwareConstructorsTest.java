/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

class DateTimeUseTimeZoneAwareConstructorsTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/datetime_constructor_use_timezone_aware.py", new DateTimeUseTimeZoneAwareConstructors());
  }

  @Test
  void UTCNowQuickFixTest() {
    var check = new DateTimeUseTimeZoneAwareConstructors();

    var before = "from datetime import datetime\n" +
      "datetime.utcnow()\n";
    var after = "from datetime import datetime\n" +
      "from datetime import timezone\n" +
      "datetime.now(timezone.utc)\n";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Change the utcnow call to construct a timezone aware datetime instead");
  }

  @Test
  void NoQuickFixOnAssignedTimeZoneTest() {
    var check = new DateTimeUseTimeZoneAwareConstructors();

    var before = "from datetime import datetime\n" +
      "timezone = True\n" +
      "datetime.utcnow()\n";

    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  @Test
  void UTCFromTimestampQuickFixTest() {
    var check = new DateTimeUseTimeZoneAwareConstructors();

    var before = "from datetime import datetime\n" +
      "datetime.utcfromtimestamp(156461321)\n";
    var after = "from datetime import datetime\n" +
      "from datetime import timezone\n" +
      "datetime.fromtimestamp(156461321, timezone.utc)\n";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Change the utcfromtimestamp call to construct a timezone aware datetime instead");
  }

  @Test
  void NoQuickFixOnAssignedTimeZoneTest2() {
    var check = new DateTimeUseTimeZoneAwareConstructors();

    var before = "from datetime import datetime\n" +
      "timezone = True\n" +
      "datetime.utcfromtimestamp(156461321)\n";

    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  @Test
  void NoQuickFixOnInvalidParameters() {
    var check = new DateTimeUseTimeZoneAwareConstructors();

    var before1 = "from datetime import datetime\n" +
      "datetime.utcfromtimestamp()\n";
    var before2 = "from datetime import datetime\n" +
      "datetime.utcnow(156)\n";

    PythonQuickFixVerifier.verifyNoQuickFixes(check, before1);
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before2);
  }
}
