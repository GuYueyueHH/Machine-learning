from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
x,y = news.data[:10],news.target[:10]
