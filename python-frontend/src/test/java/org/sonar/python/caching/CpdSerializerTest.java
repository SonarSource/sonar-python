/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.caching;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.tree.TokenImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

public class CpdSerializerTest {

  @Test
  public void to_bytes_from_bytes() throws IOException {
    var sslrToken = com.sonar.sslr.api.Token.builder()
      .setLine(1)
      .setColumn(0)
      .setValueAndOriginalValue("pass")
      .setURI(URI.create(""))
      .setType(PythonKeyword.PASS)
      .build();

    List<Token> tokens = List.of(new TokenImpl(sslrToken));
    var result = CpdSerializer.toBytes(tokens);

    List<CpdSerializer.TokenInfo> tokenInfos = CpdSerializer.fromBytes(result.data, result.stringTable);

    assertThat(tokenInfos)
      .hasSize(1);
    assertThat(tokenInfos.get(0))
      .usingRecursiveComparison().isEqualTo(new CpdSerializer.TokenInfo(1, 0, 1, 4, "pass"));
  }

  @Test
  public void corrupted_string_table_format() {
    // A string table with zero elements and an invalid terminator
    byte[] stringTable = new byte[] {0, 1, 2, 3};
    byte[] data = new byte[] {0};

    assertThatCode(() -> CpdSerializer.fromBytes(data, stringTable))
      .isInstanceOf(IOException.class)
      .hasMessageStartingWith("Can't read data from cache, format corrupted");
  }

  @Test
  public void corrupted_data_format() {
    // A string table with zero elements and a valid terminator string
    byte[] stringTable = new byte[] {0, 3, 'E', 'N', 'D'};
    byte[] data = new byte[] {0, 1, 2, 3};

    assertThatCode(() -> CpdSerializer.fromBytes(data, stringTable))
      .isInstanceOf(IOException.class)
      .hasMessageStartingWith("Can't read data from cache, format corrupted");
  }
}
