# CV2026

[Homework1-1](https://youtu.be/jQBcBZ859xE?si=d2F2WXn2HzMVACrh)  MediaPipe Selfie Segmentation

[Homework1-2](https://youtu.be/KaF4AkbqQrc?si=Xq0XGuHCcy_kVV27) Yolo

[Homework2] # scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


