# N个人，N-1个间隔，假设关系只有=和>，那么N个人的关系有N种
# 如N=3时的关系只有如下种：
# _ = _ = _
# _ = _ > _
# _ > _ = _
# _ > _ > _
# 得
# 那么该问题转化为求排列的问题，全等于的情况是没有排列的
# 1个>号产生2个位置的排列，2个>号产生3个位置的排列
# 该题答案为A3,3*C2,2+A2,3*C1,2+C0,2
# 推到N的情况下，有：
# C(0,N-1)+C(1,N-1)*A(2,N)+C(2,N-1)*A(3,N)+...+C(N-1,N-1)*A(N,N)

import sys

N = int(sys.stdin.readline().strip())


def A(n, m):
    up = 1
    for i in range(1, n + 1):
        up *= i
    down = 1
    for i in range(1, n - m + 1):
        down *= i

    return up / down


def C(n, m):
    if m == 0:
        return 1
    up = 1
    for i in range(1, n + 1):
        up *= i
    down = 1
    for i in range(1, n - m + 1):
        down *= i
    for i in range(1, m + 1):
        down *= i
    return up / down


# def func(N):
#     if N == 1:
#         print(1)
#         return

if __name__=='__main__':
    print('hello')
    # print(A(3,3))
