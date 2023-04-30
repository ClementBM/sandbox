
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

**Detailed Explanation:**

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

## **SQL vs. PL-SQL**

- SQL is used to write queries, DDL and DML statements.
- PL/SQL is used to write program blocks, functions, procedures triggers,and packages.
- SQL is executed one statement at a time.
- PL/SQL is executed as a block of code.
- SQL is declarative, i.e., it tells the database what to do but not how to do it. Whereas, PL/SQL is procedural, i.e., it tells the database how to do things.
- SQL can be embedded within a PL/SQL program. But PL/SQL cant be embedded within a SQL statement.


## Examine the following code. What will the value of price be if the statement finds a NULL value? 

`SELECT name, ISNULL(price, 50) FROM PRODUCTS`

What is a NULL Value?

A field with a NULL value is a field with no value.

If a field in a table is optional, it is possible to insert a new record or update a record without adding a value to this field. Then, the field will be saved with a NULL value.

Note: It is very important to understand that a NULL value is different from a zero value or a field that contains spaces. A field with a NULL value is one that has been left blank during record creation!

**How can you return a default value for a NULL?**

**MySQL**

The MySQL IFNULL() function lets you return an alternative value if an expression is NULL:

```sql
SELECT ProductName, UnitPrice * (UnitsInStock + IFNULL(UnitsOnOrder, 0))
FROM Products
```

or we can use the COALESCE() function, like this:

```sql
SELECT ProductName, UnitPrice * (UnitsInStock + COALESCE(UnitsOnOrder, 0))
FROM Products
```

**SQL Server**

The **SQL Server** ISNULL() function lets you return an alternative value when an expression is NULL :

-   SELECT ProductName, UnitPrice * (UnitsInStock + ISNULL(UnitsOnOrder, 0))
-   FROM Products

**MS Access**

The MS Access IsNull() function returns TRUE (-1) if the expression is a null value, otherwise FALSE (0) :

-   SELECT ProductName, UnitPrice * (UnitsInStock + IIF(IsNull(UnitsOnOrder), 0, UnitsOnOrder))
-   FROM Products

**Oracle**

The Oracle NVL() function achieves the same result:

-   SELECT ProductName, UnitPrice * (UnitsInStock + NVL(UnitsOnOrder, 0))
-   FROM Products

## Which operator is used to search for a specified text pattern in a column?

The LIKE operator is used in a WHERE clause to search for a specified pattern in a column.

There are two wildcards used in conjunction with the LIKE operator:

% - The percent sign represents zero, one, or multiple characters

_ - The underscore represents a single character

Examples:

-   WHERE CustomerName LIKE 'a%' -- Finds any values that starts with  "a"
-   WHERE CustomerName LIKE '%a' -- Finds any values that ends with  "a"
-   WHERE CustomerName LIKE '%or%' -- Finds any values that have "or"  in any position
-   WHERE CustomerName LIKE '_r%' -- Finds any values that have "r"  in the second position
-   WHERE CustomerName LIKE 'a_%_%' -- Finds any values that starts with  "a"  and are at least 3 characters in length
-   WHERE ContactName LIKE 'a%o' -- Finds any values that starts with  "a"  and ends with  "o"

## How to write a query to show the details of a student from Students table whose FirstName starts with 'K'?

```sql
SELECT * FROM Students WHERE FirstName LIKE 'K%'.
```

Explanation from previous question applies.

## Which operator is used to select values within a range?

The BETWEEN operator selects values within a given range. The values can be numbers, text, or dates.

The BETWEEN operator is inclusive: begin and end values are included.

**BETWEEN Syntax**

```sql
SELECT column_name(s)
FROM table_name
WHERE column_name BETWEEN value1 AND value2;
```

## How to get current date in MySQL (without time)?

The CURDATE() function returns the current date. This function returns the current date as a YYYY-MM-DD format if used in a string context, and as a YYYYMMDD format if used in a numeric context.

The CURRENT_DATE() function is a synonym for the CURDATE() function.

## What is the difference between HAVING clause and WHERE clause?

Both specify a search condition but HAVING clause is used only with the SELECT statement and typically used with GROUP BY clause.

If GROUP BY clause is not used then HAVING behaves like WHERE clause only.

Here are some other differences:

- HAVING filters records that work on summarized GROUP BY results.

- HAVING applies to summarized group records, whereas WHERE applies to individual records.

- Only the groups that meet the HAVING criteria will be returned.

- HAVING requires that a GROUP BY clause is present.

- WHERE and HAVING can be in the same query.

## What are different JOINS used in SQL?

There are several types of joins:

**SQL INNER JOIN Keyword**

The INNER JOIN keyword selects records that have matching values in both tables.

**SQL LEFT JOIN Keyword**

The LEFT JOIN keyword returns all records from the left table (table1), and the matched records from the right table (table2). The result is NULL from the right side, if there is no match.

**SQL RIGHT JOIN Keyword**

The RIGHT JOIN keyword returns all records from the right table (table2), and the matched records from the left table (table1). The result is NULL from the left side, when there is no match.

**SQL FULL OUTER JOIN Keyword**

The FULL OUTER JOIN keyword return all records when there is a match in either left (table1) or right (table2) table records.

Note: FULL OUTER JOIN can potentially return very large result-sets!

**SQL Self JOIN**

A self JOIN is a regular join, but the table is joined with itself.

**SQL CROSS JOIN**

The SQL CROSS JOIN produces a result set which is the number of rows in the first table multiplied by the number of rows in the second table if no WHERE clause is used along with CROSS JOIN .This kind of result is called as Cartesian Product.

If WHERE clause is used with CROSS JOIN , it functions like an INNER JOIN .

## Having a list of Customer Names that searched for product 'X' and a list of customer Names that bought the product 'X'. What set operator would you use to get only those who are interested but did not bought product 'X' yet?

**MINUS**

The **SQL** MINUS operator is used to return all rows in the first SELECT statement that are not returned by the second SELECT statement. Each SELECT statement will define a dataset. The MINUS operator will retrieve all records from the first dataset and then remove from the results all records from the second dataset.

**MINUS Syntax**

```sql
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions]
MINUS
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions];
```

## What is Case Function?

The CASE function lets you evaluate conditions and return a value when the first condition is met (like an IF-THEN-ELSE statement).

**CASE Syntax**

```sql
CASE expression
WHEN condition1 THEN result1
WHEN condition2 THEN result2
...
WHEN conditionN THEN resultN
ELSE result
END  
```

## How do you create a temporary table in MySQL?

To create a temporary table, you just need to add the TEMPORARY keyword to the CREATE TABLE statement.

Example:

```sql
CREATE TEMPORARY TABLE top10customers
SELECT customer.fname, customer.lname
FROM customers
```

-   /* all the conditions to fecth the top 10 customers */

## Which SQL statement is used to update data in a database?

The UPDATE statement is used to modify the existing records in a table.

**UPDATE Syntax**

```
-   UPDATE table_name
-   SET column1 = value1, column2 = value2, ...
-   WHERE condition;
```

## What is the difference between DELETE and TRUNCATE?

The basic difference in both is DELETE is DML command and TRUNCATE is DDL.

DELETE is used to delete a specific row from the table whereas TRUNCATE is used to remove all rows from the table

We can use DELETE with WHERE clause but cannot use TRUNCATE with it.

## What are Indexes in SQL?

Indexes are used to retrieve data from the database very fast. The users cannot see the indexes, they are just used to speed up searches/queries.

Note: Updating a table with indexes takes more time than updating a table without (because the indexes also need an update). So, only create indexes on columns that will be frequently searched against.

**CREATE INDEX Syntax**

Creates an index on a table. Duplicate values are allowed:

```sql
CREATE INDEX index_name
ON table_name (column1, column2, ...)
```

## What is the difference between clustered and non-clustered indexes? Which of the following statements are true?

One table can have only one clustered index but multiple nonclustered indexes.

Clustered indexes can be read rapidly rather than non-clustered indexes.

Clustered indexes store data physically in the table or view and non-clustered indexes do not store data in table as it has separate structure from data row.

## What does the term 'locking' refer to?

Locking is a process preventing users from reading data being changed by other users, and prevents concurrent users from changing the same data at the same time.  

## What are a transaction's main controls?

COMMIT command is used to permanently save any transaction into the database.

When we use any DML command like INSERT , UPDATE or DELETE , the changes made by these commands are not permanent, until the current session is closed, the changes made by these commands can be rolled back.

To avoid that, we use the COMMIT command to mark the changes as permanent.

Following is commit command syntax:

```sql
COMMIT;
```

ROLLBACK command

This command restores the database to last committed state. It is also used with SAVEPOINT command to jump to a savepoint in an ongoing transaction.

If we have used the UPDATE command to make some changes into the database, and realise that those changes were not required, then we can use the ROLLBACK command to rollback those changes, if they were not committed using the COMMIT command.

Following is rollback command syntax:

- _ROLLBACK TO savepoint_name;_

- SAVEPOINT command

- SAVEPOINT command is used to temporarily save a transaction so that you can rollback to that point whenever required.  

Following is savepoint command syntax:

- _SAVEPOINT savepoint_name;_

In short, using this command we can name the different states of our data in any table and then rollback to that state using the ROLLBACK command whenever required.

## Inside a stored procedure, you iterate over a set of rows returned by a query using.. ?

A CURSOR is a database object which is used to manipulate data in a row-to-row manner.

Cursor follows steps as given below:

-   Declare Cursor
-   Open Cursor
-   Retrieve row from the Cursor
-   Process the row
-   Close Cursor
-   Deallocate Cursor  

## Are views updatable using INSERT, DELETE or UPDATE?

The SQL UPDATE VIEW command can be used to modify the data of a view.

All views are not updatable. So, UPDATE command is not applicable to all views. An updatable view is one which allows performing a UPDATE command on itself without affecting any other table.

When can a view be updated?

1. The view is defined based on one and only one table.

2. The view must include the PRIMARY KEY of the table based upon which the view has been created.

3. The view should not have any field made out of aggregate functions.

4. The view must not have any DISTINCT clause in its definition.

5. The view must not have any GROUP BY or HAVING clause in its definition.

6. The view must not have any SUBQUERIES in its definitions.

7. If the view you want to update is based upon another view, the later should be updatable.

8. Any of the selected output fields (of the view) must not use constants, strings or value expressions.

## What is the default isolation level used in MySQL?

**REPEATABLE READ**

## What are valid properties of the transaction?

The characteristics of these four properties as defined by Reuter and Härder are as follows:

**Atomicity**

​	Atomicity requires that each transaction be "all or nothing": if one part of the transaction fails, then the entire transaction fails, and the database state is left unchanged. An atomic system must guarantee atomicity in each and every situation, including power failures, errors and crashes. To the outside world, a committed transaction appears (by its effects on the database) to be indivisible ("atomic"), and an aborted transaction does not happen.

**Consistency**

​	The consistency property ensures that any transaction will bring the database from one valid state to another. Any data written to the database must be valid according to all defined rules, including constraints, cascades, triggers, and any combination thereof. This does not guarantee correctness of the transaction in all ways the application programmer might have wanted (that is the responsibility of application-level code), but merely that any programming errors cannot result in the violation of any defined rules.

**Isolation**

​	The isolation property ensures that the concurrent execution of transactions results in a system state that would be obtained if transactions were executed sequentially, i.e., one after the other. Providing isolation is the main goal of concurrency control. Depending on the concurrency control method (i.e., if it uses strict - as opposed to relaxed - serializability), the effects of an incomplete transaction might not even be visible to another transaction.

**Durability**

​	The durability property ensures that once a transaction has been committed, it will remain so, even in the event of power loss, crashes, or errors. In a relational database, for instance, once a group of SQL statements execute, the results need to be stored permanently (even if the database crashes immediately thereafter). To defend against power loss, transactions (or their effects) must be recorded in a non-volatile memory.

## Difference between "read commited" and "repeatable read"
Read committed is an isolation level that guarantees that any data read was committed at the moment is read. It simply restricts the reader from seeing any intermediate, uncommitted, 'dirty' read. It makes no promise whatsoever that if the transaction re-issues the read, will find the Same data, data is free to change after it was read.

Repeatable read is a higher isolation level, that in addition to the guarantees of the read committed level, it also guarantees that any data read cannot change, if the transaction reads the same data again, it will find the previously read data in place, unchanged, and available to read.

The next isolation level, serializable, makes an even stronger guarantee: in addition to everything repeatable read guarantees, it also guarantees that no new data can be seen by a subsequent read.

## Concurrency side effects enabled by the different isolation levels
| Isolation level | Dirty read | Nonrepeatable read | Phantom |
|--|--|--|--|
|Read uncommitted|Yes|Yes|Yes|
|Read committed|No|Yes|Yes|
|Repeatable read|No|No|Yes|
|Snapshot|No|No|No|
|Serializable|No|No|No|