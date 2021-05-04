import threading

def test_one(res):
    res.append(3)
    res.append(4)
    res.append(5)

def test_two(res):
    res.append(1)
    res.append(2)
    res.append(3)

if __name__ == "__main__":
    res1 = []
    res2 = []
    t1 = threading.Thread(target=test_one, args=(res1,))
    t2 = threading.Thread(target=test_two, args=(res2, ))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(res1)
    print(res2)