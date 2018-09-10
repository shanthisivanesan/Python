import sqlite3

conn = sqlite3.connect(':memory:')

c = conn.cursor()

c.execute("""CREATE TABLE employees (
            first text,
            last text,
            pay integer
            )""")


def insert_emp(first,last,pay):
    with conn:
        c.execute("INSERT INTO employees VALUES (:first, :last, :pay)", {'first': first, 'last': last, 'pay': pay})


def get_emps_by_name(lastname):
    #c.execute("SELECT * FROM employees WHERE last=:last", {'last': lastname})
    c.execute("SELECT * FROM employees")
    return c.fetchall()


def update_pay(first,last, pay):
    with conn:
        c.execute("""UPDATE employees SET pay = :pay
                    WHERE first = :first AND last = :last""",
                  {'first': first, 'last': last, 'pay': pay})


def remove_emp(first,last):
    with conn:
        c.execute("DELETE from employees WHERE first = :first AND last = :last",
                  {'first': first, 'last': last})

insert_emp('shanthi', 'siva', 180000)
insert_emp('nive', 'jay', 90000)
insert_emp('jay', 'raj', 190000)
insert_emp('ss', 'aa', 90000)
emps = get_emps_by_name('raj')
print(emps)

update_pay('nive', 'jay', 195000)
remove_emp('ss', 'aa')

emps = get_emps_by_name('raj')
print(emps)

conn.close()