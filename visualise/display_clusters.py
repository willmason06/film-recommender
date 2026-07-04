import numpy as np
import pandas as pd

# import embedding and coressponding labels

movie_ids = np.load("data/movie_ids.npy")
embeddings = np.load("data/movie_embeddings_umap20.npy")
labels = np.load("data/movie_clusters500.npy")

df = pd.read_csv("data/tmdb_movie_data.csv")
id_to_title = dict(zip(df["id"], df["title"]))





def check_cluster_titles(label_num, max_num=None):
    titles = []
    cluster_ids = movie_ids[labels == label_num]

    if max_num is None:
        for cluster_id in cluster_ids:
            title = id_to_title.get(cluster_id, "Unknown")
            titles.append(title)

    else:
        for cluster_id in cluster_ids[:max_num]:
            title = id_to_title.get(cluster_id, "Unknown")
            titles.append(title)

    return titles

hello = check_cluster_titles(4)

print(hello)

'''
-1 138775
0 3812
1 13778
2 13325
3 688
4 20079
5 5795
6 22246
7 900
8 2172
9 2677
10 6296
11 6270
12 7719
13 3456
14 5775
15 19285
16 1548
17 7768
18 2878
19 6391
20 955
21 1745
22 2587

'''

'''
-1 149387
0 3813
1 36831
2 12500
3 13000
4 91
5 842
6 76
7 71
8 95
9 120
10 59
11 18193
12 415
13 77
14 106
15 60
16 76
17 69
18 805
19 118
20 100
21 79
22 127
23 276
24 96
25 5718
26 67
27 132
28 2127
29 59
30 71
31 224
32 75
33 189
34 397
35 185
36 89
37 107
38 756
39 485
40 111
41 2213
42 9720
43 61
44 50
45 154
46 2456
47 2064
48 1884
49 81
50 53
51 51
52 2781
53 136
54 150
55 81
56 1553
57 310
58 337
59 1122
60 3756
61 60
62 6290
63 132
64 2880
65 69
66 346
67 1307
68 311
69 79
70 635
71 1427
72 210
73 1042
74 112
75 267
76 4466
'''
