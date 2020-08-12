# Nursing Home Performance

This is a data analysis project focusing on finding a correlation between Nursing Home's Star Rating (Given by Government), and it's actual performance based on the data they reported.

# Dataset

Download the dataset from here -

[COVID-19 Nursing Home Dataset](https://data.cms.gov/Special-Programs-Initiatives-COVID-19-Nursing-Home/COVID-19-Nursing-Home-Dataset/s2uc-8wxp)

[Star Rating](https://data.medicare.gov/Nursing-Home-Compare/Star-Ratings/ax9d-vq6k) - Child of 
[Provider Info](https://data.medicare.gov/d/4pq5-n9py) Dataset. 

# Intuition  
Here's the detailed intuition of the code

## Star_Rating.ipynb

- Feature 'Processing Date' is the same for all the rows, therefore, can be dropped
- Provider Name can have similar names; For example, 'MILLER'S MERRY MANOR' is being repeated 30 times, but all have different Federal Provider Number (FPN), and also all have separate addresses too [Verified]. All the providers in this dataset have a unique FPN. 
- A lot of the ratings are missing.
- The given dataset consists of Rating Footnote instead of Rating; these footnote values are usually 1,12,18, which generally is given to the facility, which has low/erroneous/irrelevant data. Therefore, to fetch the initial rating, I chose to use its parent dataset.
Check this [link](https://www.medicare.gov/hospitalcompare/data/Footnotes.html) for more details on footnotes. 
- In the second part, I focused on getting the Provider Info dataset, i.e., the original data source of "Star Rating Dataset."
I matched both the datasets to check for inconsistency.

## COVID-19_Nursing_Home_Dataset.pynb
### Part 1 
- Focused on studying the second dataset and making some Assumptions
- There are multiple instances of the same Federal Provider Number (FPN)
- FPN is a six-digit number

- Assumptions
```diff
- It’s being mentioned on the CMS website that the data might be inaccurate in 
- some instances, but later it will be taken care of; therefore, it is 
- assumed that all the given data is accurate at this moment.
+ For some of the analyses, only those data will be considered 
+ if they have passed a quality assurance check.
- First-week data can be ignored as the facility may not be 
- familiar with data inputting; therefore, it is subject to error.
```


### Part 2 
- Extracting the dataset of each facility (Pre-Processing Phase)
- The task at hand is to extract the dataset based on FPN 
- The next task is to implement assumptions and cleaning the dataset
- Remove data that does not have passed quality assurance check
- Remove the first entry of data from a facility, or it can be modified to remove date containing 05/24/2020 as it might contain data from 01/01/2020. 
- The Data might have different data types, therefore, convert all to numeric

### Part 3 Feature Selection
- As there are multiple features and for dimensionality reduction, I will use Feature selection.
- LassoCV from the sklearn library is used to filter top ‘n’ features based on thresholding.

## Analysis.pynb
- All the code is merged into a single file – [common.py](common.py)
- To study the correlation between features and facility rating, Lasso CV Regression is used to find top ‘n’ features that affect the individual ratings. (n=10)
- The given data is studied and can be visualized with the Heatmap 

![Heatmap](https://github.com/adityavyasbme/Nursing_Home_Performance/blob/master/heatmap.png)

### Insights
- The text in red, blue, and brown are features that I believe might be correlated to each other in some way. Therefore, it can be used as a primary general classification to validate our results. We can make more groups for validation based on real-world experience.
- If we check red ones, only a few of them are selected, as these are features that might be correlated. Therefore, it is dropped by lasso. "Resident COVID-19 Deaths" and "Resident access to testing in the facility" directly affect the star rating of a facility.
- Another popular feature is the "shortage of Aides," which only affects staffing, QM, and RN Staffing rating. In parallel, "Shortage of Nursing staff" affects most of the evaluation.
- I expected "Total Resident COVID-19 Deaths Per 1,000 Residents" to be the most critical vector; however, it has been selected just once. This indicates that deaths per 1000 residents do not affect the star rating of a facility by a significant amount, but cases per 1000 residents might affect Staffing and RN Staffing rating.
- "One-week supply of gown" is not an essential feature compared to the "One week supply of gloves."
- "Staff Total Confirmed COVID-19" and "Shortage of Nursing Staff" as expected directly affects staffing rating.
 - "Current supply of ventilator" affects overall rating and, in general, affects health inspection and QM rating. 
- Key Feature
  1. "Total Number of Occupied Beds" and "Number of All Beds" are selected most, which means the availability of beds is an important feature. 
   2. Also, there is a positive correlation between "No. of occupied beds" to "No. of Beds." Comparing graphs of Rating 1 and Rating 5, it can be said that there is a significant difference in their slopes; that is, we can use this as one of the critical characteristic features to look upon for comparative studies.

## Future Works
- Also, the data is divided into five parts based on the grouping of "Overall Rating," and individual feature selection is applied. There is room for more analysis, as we can generate a characteristics map of each rating group, for example, Facilities with five stars "Overall rating" will also have high "Health Inspection Rating" then as compared to facilities with 1 star "Overall rating." Therefore, this transformed data can be used for future comparative study.
- The results can be reproduced and optimized for better performance with Cython and dynamic memory management. 
- Future work can be more focused on applying machine learning models to study the relation between two (or more) features and Its contribution to Facility’s Star rating.
- The feature "Geolocation" was dropped because it is a separate analysis.


## License
[MIT](LICENSE)
