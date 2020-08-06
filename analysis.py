import json
import numpy as np
import matplotlib.pyplot as plt

dataset = open("Apps_for_Android_5.json", "r")


def compute_analysis():
    # init
    classes = []
    for i in range(0, 5):
        classes.append({
            "score": i+1,
            "num_reviews": 0,
            "avg_length_review": 0
        })
    # compute
    for i, line in enumerate(dataset):
        review = json.loads(line.replace("\n", ""))
        idx = int(review["overall"])-1
        classes[idx]["num_reviews"] += 1
        if classes[idx]["num_reviews"] > 1:
            classes[idx]["avg_length_review"] += len(review["reviewText"].split(" "))
            classes[idx]["avg_length_review"] /= 2
        else:
            classes[idx]["avg_length_review"] = len(review["reviewText"].split(" "))
    return classes


if __name__ == '__main__':
    stats = compute_analysis()
    num_reviews = []
    avg_length = []
    for stat in stats:
        num_reviews.append(stat["num_reviews"])
        avg_length.append(stat["avg_length_review"])
        print(json.dumps(stat, indent=4))
    # create small plot
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    index = np.arange(len(stats))
    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, num_reviews, bar_width,
                     alpha=opacity,
                     color='b')

    rects2 = ax2.bar(index + bar_width, avg_length, bar_width,
                     alpha=opacity,
                     color='g')

    plt.xlabel('Scores')
    ax.set_ylabel('[# Reviews]', color="b")
    ax.set_xlabel("score class")
    ax2.set_ylabel('Avg length of review [# words]', color="g")
    plt.title('Statistic for each class of review')
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
    plt.tight_layout()
    plt.show()
