def detective_ball_bounce(n):
    i = 0
    for i in range(6):
        n = n*2
    print('The ball began bouncing at', round(n,3),' height')



n= input()
detective_ball_bounce(int(n))
