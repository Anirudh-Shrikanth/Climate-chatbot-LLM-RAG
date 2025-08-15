# Weather and Climate Analysis Bot using Retrieval-Augmented Generation (RAG)

## Overview
Climate data analysis presents unique challenges due to the massive scale, temporal complexity, and geographic distribution of meteorological datasets. Traditional database approaches struggle with natural language queries that require contextual understanding of location, time, and weather phenomena. This project presents a novel hybrid Retrieval-Augmented Generation (RAG) system specifically designed for climate data analysis, combining dense vector retrieval with sparse keyword matching and cross-encoder reranking. Our system integrates historical NOAA weather station data spanning 2019-2024 across 50 major US cities with real-time weather APIs, creating a comprehensive knowledge base of over 198,123 structured climate documents. The hybrid retrieval architecture achieves superior performance in location-specific temporal queries, with experimental results showing 86.11% accuracy in multi-location comparisons in date-filtered queries. The system successfully handles complex analytical questions such as ”Compare the difference between maximum temperatures in Phoenix in 2022 and New Orleans in 2023,” demonstrating practical applications for climate research, urban planning, and environmental monitoring.

## To run the program
Download all the files into one single directory. On your terminal, please run the command "streamlit run GUI.py".
This should spin up the streamlit UI interface where you can ask climate related questions. Once you type in a question, please click on the "Get Answer" button to get a response.

Some suggested questions are:
1. What was the temperature in Phoenix in July 2023?
2. What's the weather forecast for New York?
3. Show me the coldest day in Chicago in January 2023
4. What was the hottest day in Houston during August 2024?
5. What is the maximum temparature in New York in July 2022?
6. Compare the difference between Maximum Temparatures in New Orleans in 2023 and Phoenix in 2022
7. Seattle rainfall in March 2022. How much rainfall was there on March 2nd?
8. What was the maximum temperature in Phoenix on July 2023? On which date did this occur?
9. How hot was it in Phoenix on July 2023? What is the value on July 1?

NOTE 1:
Remember to add your "Gemini flash 2.0" API key in: (https://aistudio.google.com/apikey)
1. Line 1982 in ClimateRAGChatbot.py and
2. Line 11 of GUI.py
before you run the program.

NOTE 2:
The .ipynb file the same code as ClimateRAGChatbot.py, but provided just in case you want to take a look at the outputs cell by cell that has already been provided for your convenience.
It is **not** recommended to run the notebook as this will finish the given Gemini API token limits and would also take a lot of time.

Kindly create a key if you don't have one already. I cannot provide this as it is a security risk.