def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs= [i for i in range(len(ys))]
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def kNN(data,predict, k=3):
    distances= []
    for group in data:
        for i in data[group]:
            euclidean_distance = np.linalg.norm(np.array(i) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result