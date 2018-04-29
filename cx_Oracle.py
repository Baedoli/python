import cx_Oracle

con = cx_Oracle.connect('premier/vmflaldj@qms.wamc.co.kr:1521/AQMS')

cur = con.cursor()

cur.execute('select * from c_0010')

for result in cur:
  print(result)

cur.close()

con.close()

msg = "Hello World"

print(msg)
