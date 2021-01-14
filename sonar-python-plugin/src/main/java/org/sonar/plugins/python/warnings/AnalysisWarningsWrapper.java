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
package org.sonar.plugins.python.warnings;

import org.sonar.api.batch.InstantiationStrategy;
import org.sonar.api.batch.ScannerSide;

/**
 * As {@link org.sonar.api.notifications.AnalysisWarnings} has been added in SQ 7.4, previous version of the API
 * do not have the class. Thus, in order to avoid a {@link ClassNotFoundException} at startup, we use a wrapper class
 * we know for sure will always be present. Depending on the sonar runtime, this wrapper will either forward the
 * warnings to the underlying {@link org.sonar.api.notifications.AnalysisWarnings} or do nothing when not available.
 */
@ScannerSide
@InstantiationStrategy("PER_BATCH")
public interface AnalysisWarningsWrapper {
  void addWarning(String text);
}
