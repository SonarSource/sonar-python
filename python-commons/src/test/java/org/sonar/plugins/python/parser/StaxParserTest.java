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
package org.sonar.plugins.python.parser;

import java.io.InputStream;
import java.util.stream.Stream;
import javax.xml.stream.XMLStreamException;
import org.codehaus.staxmate.in.SMHierarchicCursor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.parser.StaxParser.XmlStreamHandler;

class StaxParserTest {

  public static Stream<Arguments> data() {
    return Stream.of(
      Arguments.of("org/sonar/plugins/python/parser/dtd-test.xml"),
      Arguments.of("org/sonar/plugins/python/parser/xsd-test.xml"),
      Arguments.of("org/sonar/plugins/python/parser/xsd-test-with-entity.xml")
    );
  }
  
  @ParameterizedTest
  @MethodSource("data")
  void testParser(String resource) throws XMLStreamException {
    parse(resource);
  }

  private void parse(String resource) {
    StaxParser parser = new StaxParser(getTestHandler());
    InputStream stream = getClass().getClassLoader().getResourceAsStream(resource);
    Assertions.assertNotNull(stream);
    Assertions.assertDoesNotThrow(() -> parser.parse(stream));
  }

  private static XmlStreamHandler getTestHandler() {
    return new XmlStreamHandler() {
      public void stream(SMHierarchicCursor rootCursor) throws XMLStreamException {
        rootCursor.advance();
        while (rootCursor.getNext() != null) {
          // do nothing intentionally
        }
      }
    };
  }

}
