import numpy as np
import math

def _get_percent_index(fg_inds, fg_iou, fg_num):
    """
    fg_iou 排序，得到最大点max_iou和最小点min_iou，得到中间0.2, 0.4, 0.6, 0.8大小的点对应的index；从相邻两个index中均匀抽取index.
    fg_iou 是不连续的.
    """

    sorted_index = np.argsort(fg_iou)
    sorted_iou = np.sort(fg_iou)

    min_iou = min(sorted_iou)
    max_iou = max(sorted_iou)

    a2 = min_iou + (max_iou - min_iou) * 0.2
    a4 = min_iou + (max_iou - min_iou) * 0.4
    a6 = min_iou + (max_iou - min_iou) * 0.6
    a8 = min_iou + (max_iou - min_iou) * 0.8

    print('a2:', a2)
    print('a4:', a4)
    print('a6:', a6)
    print('a8:', a8)


    def get_index(a):
        for i in range(len(sorted_iou)):
            if sorted_iou[i] > a:
                return i

    b2 = get_index(a2)
    b4 = get_index(a4)
    b6 = get_index(a6)
    b8 = get_index(a8)

    avg_num = math.floor(fg_num * 0.2)

    def get_random_index(b0, b1):
        if b1-b0 >= avg_num:
            c = np.random.choice(range(b0, b1), avg_num, False)
        elif b1-b0 > 0:
            c = np.array(range(b0, b1))
        else:
            c = []
        return c

    c0 = get_random_index(0, b2)
    c2 = get_random_index(b2, b4)
    c4 = get_random_index(b4, b6)
    c6 = get_random_index(b6, b8)
    c8 = get_random_index(b8, len(sorted_iou))
        
    
    # c2 = np.random.choice(range(b2, b4), avg_num, False) 
    # c4 = np.random.choice(range(b4, b6), avg_num, False)
    # c6 = np.random.choice(range(b6, b8), avg_num, False)
    # c8 = np.random.choice(range(b8, len(sorted_iou)), fg_num-4*avg_num, False)

    print('b2:', b2)
    print('b4:', b4)
    print('b6:', b6)
    print('b8:', b8)

    d0 = np.array(fg_inds)[sorted_index[c0]]
    d2 = np.array(fg_inds)[sorted_index[c2]]
    d4 = np.array(fg_inds)[sorted_index[c4]]
    d6 = np.array(fg_inds)[sorted_index[c6]]
    d8 = np.array(fg_inds)[sorted_index[c8]]
    
    return d0, d2, d4, d6, d8



if __name__=='__main__':
    # inds = [3, 5, 9, 10, 4, 13, 16, 23, 19, 21]
    # iou = [1.0, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55]

    inds = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    iou = [10, 9, 8, 7, 6, 5, 4, 4, 4, 4]
    fg_num = 5

    a1, a2, a3, a4, a5 = _get_percent_index(inds, iou, fg_num)

    print('a1:', a1)
    print('a2:', a2)
    print('a3:', a3)
    print('a4:', a4)
    print('a5:', a5)