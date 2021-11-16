
def open_account():
    print("새로운 계좌가 생성되었습니다.")

def deposit(balance, money):
    print("입금이 완료 되었습니다. 잔액은 {0} 원 입니다.".format(balance+money))
    return balance+money

def withdraw(balance, money):
    if balance >= money:
        print("출금이 완료 되었습니다. 잔액은 {0} 원 입니다.".format(balance-money))
        return balance-money
    else:
        print("잔액이 부족합니다. 출금할 수 없습니다. 현재 잔액은 {0} 입니다.".format(balance))
        return balance

def withdraw_night(balance, money): # 야간 출금 이며 수수료는 100원 이 계산 된다.
    commission = 100
    if balance >= money+commission:
        print("출금이 완료 되었습니다. 잔액은 {0} 원 입니다.".format(balance-money-commission))
        balance = balance - money - commission
        return commission, balance
    else:
        print("잔액이 부족합니다. 현재 잔액은 {0} 원 입니다.".format(balance))
        return commission, balance

balance = 0
open_account()
balance = deposit(balance,1000)
print(balance)
balance = withdraw(balance,500)
commission, balance = withdraw_night(balance,100)
print(" 현재 수수료는 {0} 원입니다.".format(commission))

