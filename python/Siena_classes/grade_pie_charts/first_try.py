import numpy as np
import matplotlib.pylab as plt


def grade_pie_chart(weights,scores,labels):
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    if type(weights)==list:
        weights = np.array(weights)
    if type(scores)==list:
        print(scores)
        temp_scores = np.array(scores)
        temp_scores[temp_scores == None] = 0.0
        temp_scores /= 100.0
    if type(labels)==list:
        labels = np.array(labels)

    size = 0.3

    vals = []
    for weight in weights:
        vals.append([weight,weight])

    vals = np.array(vals)
    #vals = np.array([[60., 60.], [40., 40.], [29., 29.]])
    #normalize vals to 2 pi
    valsnorm = vals/np.sum(vals)*2*np.pi
    #obtain the ordinates of the bar edges
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

    print(valsleft)
    print(valsleft[:,0])
    print(valsnorm.flatten())

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1, 2, 4, 6, 9, 8]))

    ax.bar(x=valsleft[:,0],
                   width=valsnorm.sum(axis=1), bottom=0, height=1,
                          color='lightgrey', edgecolor='w', linewidth=1, align="edge")

    print(temp_scores)
    print(labels)
    ax.bar(x=valsleft[:, 0],
                   width=valsnorm.sum(axis=1), bottom=0, height=temp_scores,
                          color=inner_colors, edgecolor='w', linewidth=1, align="edge")
    '''
    ax.bar(x=valsleft.flatten(),
                   width=valsnorm.flatten(), bottom=1-2*size, height=size,
                          color=inner_colors, edgecolor='w', linewidth=1, align="edge")
    '''


    ax.set(title="Name goes here")
    ax.set_axis_off()
    
    # Radians, r, text
    print("HERE")
    midpoints = []
    for i in range(len(valsleft[:,0])-1):
        midpoints.append((valsleft[:,0][i+1] - valsleft[:,0][i])/2 + valsleft[:,0][i])
    print(midpoints)
    midpoints.append((np.pi*2  - valsleft[:,0][-1])/2 + valsleft[:,0][-1])
    for radian, label in zip(midpoints,labels):
        print(radian)
        print(label)
        ax.text(radian, 1.3, label)

    plt.show()



grade_pie_chart([25, 30, 15, 10, 30], [95, 67, 88, 90, None], ['HW','Midterms','Lab','Quizzes','Final'])
