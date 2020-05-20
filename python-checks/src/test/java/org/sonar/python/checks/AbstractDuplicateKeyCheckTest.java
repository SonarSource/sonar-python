/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import java.math.BigDecimal;
import java.util.function.Supplier;
import org.junit.Test;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.sonar.python.checks.AbstractDuplicateKeyCheck.NumericValue;
import static org.sonar.python.checks.AbstractDuplicateKeyCheck.StringValue;

@SuppressWarnings("squid:S2701")
public class AbstractDuplicateKeyCheckTest {

    private static <A> void testHashEquals(
      Supplier<A> firstColliding,
      Supplier<A> secondColliding,
      Supplier<A> unrelated,
      boolean canReturnDifferentValues
    ) {
        assertEquals(firstColliding.get().hashCode(), secondColliding.get().hashCode());
        assertNotEquals(firstColliding.get(), secondColliding.get());
        assertNotEquals(firstColliding.get(), unrelated.get());
        if (!canReturnDifferentValues) {
            assertEquals(firstColliding.get(), firstColliding.get());
        } else {
            assertNotEquals(firstColliding.get(), firstColliding.get());
        }
        A a = firstColliding.get();
        assertEquals(a, a);
        // This is not same as `assertNotNull`, it must invoke `equals`.
        assertNotEquals(a, null);
        assertNotEquals(a, "that can't be the same value");
    }

    private static final int INT_COLLISION_A = 138563169;
    private static final int INT_COLLISION_B = 158370000;

    @Test
    public void testHashEqualsNumeric() {
        testHashEquals(
          () -> new NumericValue(new BigDecimal(INT_COLLISION_A)),
          () -> new NumericValue(new BigDecimal(INT_COLLISION_B)),
          () -> new NumericValue(new BigDecimal(0)),
          false
        );
    }

    @Test
    public void testHashEqualsString() {
        testHashEquals(
          () -> new StringValue("Aa", "Aa"),
          () -> new StringValue("BB", "BB"),
          () -> new StringValue("", ""),
          false
        );

        testHashEquals(
          () -> new StringValue("Aa", "Aa"),
          () -> new StringValue("BB", "Aa"),
          () -> new StringValue("", ""),
          false
        );

        testHashEquals(
          () -> new StringValue("Aa", "Aa"),
          () -> new StringValue("Aa", "BB"),
          () -> new StringValue("", ""),
          false
        );
    }


    @Test
    public void testHashEqualsTree() {
        testHashEquals(
          () -> eval("('a', 'Aa')"),
          () -> eval("('a', 'BB')"),
          () -> eval("(x, y)"),
          false
        );

        testHashEquals(
          () -> eval("f('Aa')"),
          () -> eval("f('BB')"),
          () -> eval("(x, y)"),
          true
        );
    }

    @Test
    public void testHashEqualsToken() {
        testHashEquals(
          () -> eval("Aa"),
          () -> eval("BB"),
          () -> eval("sthElse"),
          false
        );

        // The string with colliding hash was found using
        // unhash(n: Int): String = if (n < 31) "" + n.toChar else (unhash(n / 31) + (n % 31).toChar)
        // and some regexes to filter out only legible strings.
        assertNotEquals(eval("Z"), eval("89558118"));
    }

    @Test
    public void debugThis() {
        testHashEquals(
          () -> evalFirstToken("Aa"),
          () -> evalFirstToken("BB"),
          () -> evalFirstToken("sthElse"),
          false
        );
        assertNotEquals(evalFirstToken("Z"), evalFirstToken("89558118"));
    }

    private Expression parse(String content) {
        PythonParser parser = PythonParser.create();
        AstNode astNode = parser.parse(content);
        FileInput fileInput = new PythonTreeMaker().fileInput(astNode);
        return ((ExpressionStatement) (fileInput.statements().statements().get(0))).expressions().get(0);
    }

    private AbstractDuplicateKeyCheck.Value eval(String content) {
        return AbstractDuplicateKeyCheck.evaluate(parse(content));
    }

    private AbstractDuplicateKeyCheck.Value evalFirstToken(String content) {
        return AbstractDuplicateKeyCheck.evaluate(parse(content).firstToken());
    }

}
