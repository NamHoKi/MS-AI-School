import random

class NumberGame():
    def __init__(self):
        self.random_number = random.randint(1,100)
        self.count = 1
        self.start()

    def start(self):
        while(True):
            if self.check_num(self.num_input()):
                print("정답입니다!!\ncount =",self.count)
                return
            else:
                continue

    def num_input(self):
        while(True):
            ans = int(input("1 -100 사이 숫자를 입력하세요\n$ "))
            
            if ans >= 1 and ans <= 100:
                return ans

    def check_num(self, ans):
        if ans == self.random_number:
            return True
        elif ans < self.random_number:
            print("Up")
            self.count += 1
            return False
        else:
            print("Down")
            self.count += 1
            return False

start = NumberGame()