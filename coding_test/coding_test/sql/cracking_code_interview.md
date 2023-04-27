
## What are different types of statements supported by SQL?

**DDL** (Data Definition Language): It is used to define the database structure such as tables. It includes three statements such as *Create*, *Alter*, and *Drop*.

**DML** (Data Manipulation Language): These statements are used to manipulate the data in records. Commonly used DML statements are *Select*, *Insert*, *Update*, and *Delete*.

Note: Some people prefer to assign the SELECT statement to a category of its own called: DQL. Data Query Language.

**DCL** (Data Control Language): These statements are used to set privileges such as Grant and Revoke database access permission to the specific user.

## SQL was developed as an integral part of

**SQL** - Structured Query Language is a domain-specific language used in programming and designed for managing data held in a relational database management system (RDBMS), or for stream processing in a relational data stream management system (RDSMS).

Although **SQL** is an **ANSI** (American National Standards Institute) standard, there are different versions of the SQL language.

However, to be compliant with the **ANSI** standard, they all support at least the major commands (such as *SELECT*, *UPDATE*, *DELETE*, *INSERT*, *WHERE*) in a similar manner.

Note: Most of the SQL database programs also have their own proprietary extensions in addition to the **SQL** standard!

## What SQL is? (referring to it as a Programming Language)

**SQL** is a declarative language in which the expected result or operation is given without the specific details about how to accomplish the task. The steps required to execute **SQL** statements are handled transparently by the **SQL** database. Sometimes **SQL** is characterized as non-procedural because procedural languages generally require the details of the operations to be specified, such as opening and closing tables, loading and searching indexes, or flushing buffers and writing data to filesystems. Therefore, **SQL** is considered to be designed at a higher conceptual level of operation than procedural languages because the lower level logical and physical operations aren't specified and are determined by the SQL engine or server process that executes it.

## Which of the following are Aggregate Functions?

SQL Aggregate functions are:

* COUNT counts how many rows are in a particular column.
* SUM adds together all the values in a particular column.
* MIN and MAX return the lowest and highest values in a particular column, respectively.
* AVG calculates the average of a group of selected values.

## The SQL SELECT TOP Clause?

The **SELECT TOP** clause is used to specify the number of records to return.

The **SELECT TOP** clause is useful on large tables with thousands of records. Returning a large number of records can impact on performance.

Not all database systems support the **SELECT TOP** clause.MySQL supports the **LIMIT** clause to select a limited number of records, while Oracle uses **ROWNUM**.

## What is the difference between UNION and UNION ALL?

UNION – returns all distinct rows selected by either query

UNION ALL – returns all rows selected by either query, including all duplicates.

## What is the difference between DELETE and TRUNCATE?

The basic difference in both is DELETE is DML command and TRUNCATE is DDL.

DELETE is used to delete a specific row from the table whereas TRUNCATE is used to remove all rows from the table

We can use DELETE with WHERE clause but cannot use TRUNCATE with it.

## What is Collation in SQL?

A collation is a set of rules that defines how to compare and sort character strings.

Detailed Explanation:

A character set is a set of symbols and encodings. A collation is a set of rules for comparing characters in a character set. Let's make the distinction clear with an example of an imaginary character set.

Suppose that we have an alphabet with four letters: A, B, a, b. We give each letter a number: A = 0, B = 1, a = 2, b = 3. The letter A is a symbol, the number 0 is the encoding for A, and the combination of all four letters and their encodings is a character set.

Suppose that we want to compare two string values, A and B. The simplest way to do this is to look at the encodings: 0 for A and 1 for B. Because 0 is less than 1, we say A is less than B. What we've just done is apply a collation to our character set. The collation is a set of rules (only one rule in this case): “compare the encodings.” We call this simplest of all possible collations a binary collation.

But what if we want to say that the lowercase and uppercase letters are equivalent? Then we would have at least two rules: (1) treat the lowercase letters a and b as equivalent to A and B; (2) then compare the encodings. We call this a case-insensitive collation. It is a little more complex than a binary collation.

In real life, most character sets have many characters: not just A and B but whole alphabets, sometimes multiple alphabets or eastern writing systems with thousands of characters, along with many special symbols and punctuation marks. Also in real life, most collations have many rules, not just for whether to distinguish lettercase, but also for whether to distinguish accents (an “accent” is a mark attached to a character as in German Ö), and for multiple-character mappings (such as the rule that Ö = OE in one of the two German collations).

## What are valid constraints in MySQL?

SQL constraints are used to specify rules for the data in a table.

Constraints are used to limit the type of data that can go into a table. This ensures the accuracy and reliability of the data in the table. If there is any violation between the constraint and the data action, the action is aborted.

Constraints can be column level or table level. Column level constraints apply to a column, and table level constraints apply to the whole table.

The following constraints are commonly used in SQL:

* NOT NULL - Ensures that a column cannot have a NULL value
* UNIQUE - Ensures that all values in a column are different
* PRIMARY KEY - A combination of a NOT NULL and UNIQUE . Uniquely identifies each row in a table
* FOREIGN KEY - Uniquely identifies a row/record in another table
* CHECK - Ensures that all values in a column satisfies a specific condition
* DEFAULT - Sets a default value for a column when no value is specified
* INDEX - Used to create and retrieve data from the database very quickly
