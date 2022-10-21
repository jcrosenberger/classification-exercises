'''
Warmup:
A ball is dropped and bounces. At each rebound, the ball reaches half the height of the previous rebound.
9:09
How high must the ball be dropped so it reaches exactly the height h after the sixth rebound?
9:11
Write a .py script that accepts an integer from the user that represents the height of the ball after the sixth rebound.
Your .py script should print out the height that the ball must be drop at to reach the given height at the sixth rebound.
'''

def detective_ball_bounce(n):
    i = 0
    for i in range(6):
        n = n*2
    print('The ball began bouncing at', round(n,3),'height')


print('After 6 bounces, how high did the ball bounce?')
n= input()
detective_ball_bounce(float(n))
