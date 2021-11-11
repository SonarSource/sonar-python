/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.plugins.python.api;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;

import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_27;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_310;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_35;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_36;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_37;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_38;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_39;

public class PythonVersionUtils {

  public enum Version {
    V_27(2.7, "27"), V_35(3.5, "35"), V_36(3.6, "36"), V_37(3.7, "37"), V_38(3.8, "38"), V_39(3.9, "39"), V_310(3.10, "310");

    private final double value;
    private final String serializedValue;

    Version(double value, String serializedValue) {
      this.value = value;
      this.serializedValue = serializedValue;
    }

    public double value() {
      return value;
    }

    public String serializedValue() {
      return serializedValue;
    }
  }

  /**
   * Note that versions between 3 and 3.5 are currently mapped to 3.5 because
   * we don't take into account those version during typeshed symbols serialization
   */
  private static final Map<String, Version> STRING_VERSION_MAP = new HashMap<>();
  static {
    STRING_VERSION_MAP.put("2", V_27);
    STRING_VERSION_MAP.put("2.7", V_27);
    STRING_VERSION_MAP.put("3", V_35);
    STRING_VERSION_MAP.put("3.0", V_35);
    STRING_VERSION_MAP.put("3.1", V_35);
    STRING_VERSION_MAP.put("3.2", V_35);
    STRING_VERSION_MAP.put("3.3", V_35);
    STRING_VERSION_MAP.put("3.4", V_35);
    STRING_VERSION_MAP.put("3.5", V_35);
    STRING_VERSION_MAP.put("3.6", V_36);
    STRING_VERSION_MAP.put("3.7", V_37);
    STRING_VERSION_MAP.put("3.8", V_38);
    STRING_VERSION_MAP.put("3.9", V_39);
    STRING_VERSION_MAP.put("3.10", V_310);
  }
  private static final Version MIN_SUPPORTED_VERSION = V_27;
  private static final Version MAX_SUPPORTED_VERSION = V_39;
  private static final Logger LOG = Loggers.get(PythonVersionUtils.class);
  public static final String PYTHON_VERSION_KEY = "sonar.python.version";

  private PythonVersionUtils() {
  }

  public static Set<Version> fromString(String propertyValue) {
    String[] versions = propertyValue.split(",");
    if (versions.length == 0) {
      return allVersions();
    }
    Set<Version> pythonVersions = EnumSet.noneOf(Version.class);
    for (String versionValue : versions) {
      versionValue = versionValue.trim();
      Version version = STRING_VERSION_MAP.get(versionValue);
      if (version != null) {
        pythonVersions.add(version);
      } else {
        boolean isGuessSuccessful = guessPythonVersion(pythonVersions, versionValue);
        if (!isGuessSuccessful) {
          return allVersions();
        }
      }
    }
    return pythonVersions;
  }

  public static Set<Version> allVersions() {
    return EnumSet.allOf(Version.class);
  }

  private static boolean guessPythonVersion(Set<Version> pythonVersions, String versionValue) {
    try {
      double parsedVersion = Double.parseDouble(versionValue);
      if (parsedVersion < MIN_SUPPORTED_VERSION.value()) {
        pythonVersions.add(MIN_SUPPORTED_VERSION);
      } else if (parsedVersion > MAX_SUPPORTED_VERSION.value()) {
        pythonVersions.add(MAX_SUPPORTED_VERSION);
      } else {
        logErrorMessage(versionValue);
        return false;
      }
    } catch (NumberFormatException nfe) {
      logErrorMessage(versionValue);
      return false;
    }
    return true;
  }

  private static void logErrorMessage(String propertyValue) {
    String prefix = "Error while parsing value of parameter '%s' (%s). Versions must be specified as MAJOR_VERSION.MIN.VERSION (e.g. \"3.7, 3.8\")";
    LOG.warn(String.format(Locale.ROOT, prefix, PYTHON_VERSION_KEY, propertyValue));
  }
}
