/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python.toolkit;

import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.impl.Parser;
import java.util.Arrays;
import java.util.Collections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.colorizer.KeywordsTokenizer;
import org.sonar.colorizer.Tokenizer;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.parser.PythonParser;
import org.sonar.sslr.toolkit.AbstractConfigurationModel;
import org.sonar.sslr.toolkit.ConfigurationProperty;
import org.sonar.sslr.toolkit.Validators;

import java.nio.charset.Charset;
import java.util.List;

public class PythonConfigurationModel extends AbstractConfigurationModel {

  private static final Logger LOG = LoggerFactory.getLogger(PythonConfigurationModel.class);

  private static final String CHARSET_PROPERTY_KEY = "sonar.sourceEncoding";

  // VisibleForTesting
  ConfigurationProperty charsetProperty = new ConfigurationProperty("Charset", CHARSET_PROPERTY_KEY,
    getPropertyOrDefaultValue(CHARSET_PROPERTY_KEY, "UTF-8"),
    Validators.charsetValidator());

  @Override
  public Charset getCharset() {
    return Charset.forName(charsetProperty.getValue());
  }

  @Override
  public List<ConfigurationProperty> getProperties() {
    return Collections.singletonList(charsetProperty);
  }

  @Override
  public Parser<Grammar> doGetParser() {
    return PythonParser.create(getConfiguration());
  }

  @Override
  public List<Tokenizer> doGetTokenizers() {
    return Arrays.asList(
      (Tokenizer) new KeywordsTokenizer("<span class=\"k\">", "</span>", PythonKeyword.keywordValues()));
  }

  // VisibleForTesting
  PythonConfiguration getConfiguration() {
    return new PythonConfiguration(Charset.forName(charsetProperty.getValue()));
  }

  // VisibleForTesting
  static String getPropertyOrDefaultValue(String propertyKey, String defaultValue) {
    String propertyValue = System.getProperty(propertyKey);

    if (propertyValue == null) {
      LOG.info("Property \"{}\" is not set, using the default value \"{}\".", propertyKey, defaultValue);
      return defaultValue;
    } else {
      LOG.info("Property \"{}\" is set, using its value \"{}\".", propertyKey, propertyValue);
      return propertyValue;
    }
  }

}
