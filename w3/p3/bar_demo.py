import matplotlib.pyplot as plt


def plot_stacked_bar(width=0.5):
    weeks = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    emotion_freq = {
    'Joy' : [2,3,4,5,1,8,12],
    'Anger' : [11,2,1,13,1,1,1],
    'Sadness' : [2,2,3,4,2,1,1],
    'Disgust' : [1,1,1,1,1,1,1],
    'Fear' : [1,4,2,7,2,3,7]}
    fig, ax = plt.subplots()
    bottom = [0 for i in range(len(weeks))]
    for e, ef in emotion_freq.items():
        p = ax.bar(weeks, ef, width, label=e, bottom=bottom)
        for i in range(len(bottom)):
            bottom[i]+=ef[i]
    ax.legend()
    plt.show()

plot_stacked_bar()