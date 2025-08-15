Climate data analysis presents unique challenges due to the massive scale, temporal complexity, and geographic distribution of meteorological datasets. Tradi-
tional database approaches struggle with natural lan-
guage queries that require contextual understanding of
location, time, and weather phenomena. This paper
presents a novel hybrid Retrieval-Augmented Gener-
ation (RAG) system specifically designed for climate
data analysis, combining dense vector retrieval with
sparse keyword matching and cross-encoder reranking.
Our system integrates historical NOAA weather sta-
tion data spanning 2019-2024 across 50 major US cities
with real-time weather APIs, creating a comprehensive
knowledge base of over 198,123 structured climate doc-
uments. The hybrid retrieval architecture achieves supe-
rior performance in location-specific temporal queries,
with experimental results showing 86.11% accuracy in
multi-location comparisons in date-filtered queries. The
system successfully handles complex analytical ques-
tions such as ”Compare the difference between maxi-
mum temperatures in Phoenix in 2022 and New Orleans
in 2023,” demonstrating practical applications for cli-
mate research, urban planning, and environmental mon-
itoring.