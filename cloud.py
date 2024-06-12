import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

df = pd.read_excel("./lyrics.xlsx")
lyrics = df["lyrics"].tolist()

all_in_one = " ".join(lyrics)


wc = WordCloud(
    # these two make background transparent
    mode="RGBA",
    background_color=None,
    color_func=lambda *args, **kwargs: "red",
    collocations=False,
    max_words=100,
).generate(all_in_one)


plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


wc.to_file("tswift.png")
