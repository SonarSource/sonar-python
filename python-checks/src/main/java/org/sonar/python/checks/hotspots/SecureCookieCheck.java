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
package org.sonar.python.checks.hotspots;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.sonar.check.Rule;

@Rule(key = "S2092")
public class SecureCookieCheck extends AbstractCookieFlagCheck {

  private static Map<String, Integer> sensitiveArgumentByFQN;
  static {
    sensitiveArgumentByFQN = new HashMap<>();
    sensitiveArgumentByFQN.put("django.http.response.HttpResponseBase.set_cookie", 6);
    sensitiveArgumentByFQN.put("django.http.response.HttpResponseBase.set_signed_cookie", 7);
    sensitiveArgumentByFQN.put("flask.wrappers.Response.set_cookie", 6);
    sensitiveArgumentByFQN.put("werkzeug.wrappers.BaseResponse.set_cookie", 6);
    sensitiveArgumentByFQN.put("werkzeug.sansio.response.Response.set_cookie", 7);
    sensitiveArgumentByFQN = Collections.unmodifiableMap(sensitiveArgumentByFQN);
  }

  @Override
  String flagName() {
    return "secure";
  }

  @Override
  String message() {
    return "Make sure creating this cookie without the \"secure\" flag is safe.";
  }

  @Override
  Map<String, Integer> sensitiveArgumentByFQN() {
    return sensitiveArgumentByFQN;
  }
}
