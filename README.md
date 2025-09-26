# Simple-Data-Cleaning-and-Visualization-with-Python
This is a final project for Python course in City University of Hong Kong. The course project only use Jupyter Notebook. The codes contain manipulating data files, processing data, conducting exploratory data analysis, and making predictions based on data.
File Name and explanation:
(1)SpeedDating.csv       a file containing data collected from participants in experimental speed dating events, which doesn't have titles and may contain empty cells.
(2)variables.txt         a file containing the titles for SpeedDating.csv. What we need to do is to combine the two files together for future analysis.
(3)dataforanalysis.csv   the file combined by SpeedDating.csv and variables.txt.
(4)main_program.py       the main program processing files.
(5)outputsByProjectCode.html   the outputs by running the codes.

Course Requirements:
(1) Data cleaning (25%)
	(10%) Read data from txt file.
	    Please write code to read the file named variables.txt.
	    Please write code to delete the brief introduction content at the first two lines of the file, delete the column of Definition and Types, only keep the variable names as a list.
	(15%) Pre-processing data and store new data into CSV file 
	    Please read SpeedDating.csv as a dataframe. add column names to the dataframe with the above variable names.
	    There are observations with missing values in this data. Write code to remove these observations and then display the number of observations remaining after removal.
	    Convert the categorical variable 'gender' into binary form, where 1 represents 'male' and 0 represents 'female'
	    Variables 'samerace' and 'match' contain string values that need to be converted. Convert these categorical variables into binary form, where 1 represents 'same race' or 'matched', and 0 represents 'different race' or 'not matched', respectively. For example, b'0' should be converted into 0, while b'1' should be converted into 1.
	    Please store the combined dataset into a new CSV file named dataforanalysis.csv.
(2) Exploratory data analysis (40%)
    To overview the distribution of data in the dataset, you need to conduct the descriptive statistics:
	 (10%) Please find all numerical variables, conduct descriptive statistics and draw histograms of the variables. 
	 (15%) Please calculate the age difference of variables “Age” and “age_o” for each observation. 
	    Add the age differences as variable “age_d” in a new column in the dataframe.
	    Draw a bar chart to show the age difference distribution. The x-axis should be age difference groups, and the y-axis should be the number of observations having this age difference. Below is an example of the expected bar chart.
  	(15%) In the dataset, each participant assigns a total of 100 points across six qualities they would like to see in a partner: “attractive_important”, “sincere_important”, “intelligence_important”, “funny_important”, “ambition_important”, and “shared_interests_important”. The points they assign represent the importance of each quality to them. After participants rate their partner, the scores are stored in “attractive_partner”, “sincere_partner”, “intelligence_partner”, funny_partner”, “ambition_partner”, and “shared_interests_partner”.
	    For the first participant, draw a pie chart to display the importance of each quality, you can choose any color combination for the chart. 
	    Please calculate relative scores for each quality based on the importance of each quality and the score that participant give to their partner. Add these relative scores as new variables in the dataframe. For example, the relative score of attractiveness can be calculated by: relative_attractive=attractive_partner×  (attractive_important )/100

(3) Partitioning data and predicting attrition (35%)
	(5%) Partitioning the dataframe outputted from 3.2 into train data set and test data set. The train data set should be about 80% of all data points and the test data set should be 20% of them.
	(15%) Write a program to predict matching results in the social event based on other variables in the file, using the 'match' attribute as the target. Please construct the prediction model as a function/functions and call the function(s) when making prediction. The prediction models should be chosen from those covered in class (such as decision trees or other models). 
	(15%) Evaluating the prediction results. The classification_report and plot_confusion_matrix functions can be used to check the model performance.
