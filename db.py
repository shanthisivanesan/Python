import sqlite3
#from employee import Employee

conn = sqlite3.connect(':memory:')

c = conn.cursor()

c.execute("""CREATE TABLE employees (
            first text,
            last text,
            pay integer
            )""")
c.execute("INSERT INTO employees VALUES ('ss','bb',1111) ")
c.execute("INSERT INTO employees VALUES ('ss1','bb1',2222) ")
c.execute("INSERT INTO employees VALUES ('ss2','bb2',3333) ")
c.execute("SELECT * FROM employees")
c.execute("UPDATE employees set pay=4444 WHERE first='ss'")
c.execute("SELECT * FROM employees")
print(c.fetchall())
c.close()