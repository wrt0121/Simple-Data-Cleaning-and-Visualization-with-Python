import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
#please put the "variables.txt" and "SpeedDating.csv" into the same folder
#Data Cleaning--Read and delete msgs from txt file
def InputOutputData():
    with open("variables.txt",'r') as infile:
        lines=infile.readlines()
        variable_name=[x.split()[0] for x in lines if x.strip()]  #以空格切割每行的每部分
    with open("variables.txt",'w') as infile:
        for number,line in enumerate(variable_name):
            if number not in [0,1] and line.strip():
                infile.write(line+'\n')
    print("The first two lines of txt and empty lines have been deleted.\nDefinition and Types have been deleted.")
                
#Data cleaning--Pre-processing data and store new data into CSV file


def CsvAddTitle():
    with open("variables.txt",'r') as variables:   #read existing variables
        variable_name=variables.readlines()
        variable_name=[x.strip() for x in variable_name]
    df=pd.read_csv("SpeedDating.csv",header=None)#加标题
    df.columns=variable_name[1::]#除去variable_name这个标题
    df.to_csv("dataforanalysis.csv", index=False)
    print("Variable names have been added to the top of CSV file.")

def DropNaN():  #remove columns with empty cells
    df=pd.read_csv("dataforanalysis.csv")
    print("Before dropping columns with empty cells:",len(df)-1)
    df.replace(" ",np.nan, inplace=True)
    df.dropna(inplace=True)
    print("After dropping columns with empty cells:", len(df)-1)#-1 for column titles
    df.to_csv("dataforanalysis.csv",index=False)

def ConvertGender():
    df=pd.read_csv("dataforanalysis.csv")
    df['gender']=df['gender'].replace("b'female'","0")
    df['gender']=df['gender'].replace("b'male'","1")
    df.to_csv("dataforanalysis.csv",index=False)
    print("Gender column has been converted into binary form")
    
def ConvertSameraceAndMatch():
    df=pd.read_csv("dataforanalysis.csv")
    df['samerace']=df['samerace'].replace("b'1'","1")
    df['samerace']=df['samerace'].replace("b'0'","0")
    df['match']   =df['match'].   replace("b'1'","1")
    df['match']   =df['match'].   replace("b'0'","0")
    df.to_csv("dataforanalysis.csv",index=False)
    print("Samerace and match columns have been converted into binary form")
    

    
#Exploratory data analysis-find all numerical variables, conduct descriptive statistics and draw histograms of the variables
#Draw Histograms
def ReadCsvColumnName():
    df=pd.read_csv('dataforanalysis.csv')
    global variable_name
    variable_name=list(df.columns)  #获取变量
    
def DrawOneHistogram(variable):
    df=pd.read_csv('dataforanalysis.csv')
    data=df[variable]   #获取第x列的数据
    size_data=int(max(data)-min(data))
    plt.figure(figsize=(6,2),dpi=80)
    if size_data<=1:
        plt.hist(data,bins=2,edgecolor='black')
        plt.title(variable.capitalize()+" Histogram")
        plt.xlabel(variable.capitalize())
        plt.ylabel('Number of observations')
        plt.show()
    if size_data>1 and size_data<=20:
        plt.hist(data,bins=size_data,edgecolor='black')
        plt.title(variable.capitalize()+" Histogram")
        plt.xlabel(variable.capitalize())
        plt.ylabel('Number of observations')
        plt.show()
    if size_data>20:
        size_data=int(size_data/5)
        plt.hist(data,bins=size_data,edgecolor='black')
        plt.title(variable.capitalize()+" Histogram")
        plt.xlabel(variable.capitalize())
        plt.ylabel('Number of observations')
        plt.show()
        
def DrawHistograms():
    df=pd.read_csv('dataforanalysis.csv')
    ReadCsvColumnName()
    for variable in variable_name:
        DrawOneHistogram(variable)  #对于每个variable都进行一次画图

def AddAgeDifferenceToCsv():
    ReadCsvColumnName()
    df=pd.read_csv('dataforanalysis.csv')
    Age_self_list      =list(df['age'])
    Age_partner_list   =list(df['age_o'])
    Age_difference_list=[]
    len_excel          =len(Age_self_list)
    for x in range(len_excel):  #每个值都相减，取绝对值
        Age_difference=Age_self_list[x]-Age_partner_list[x]
        if Age_difference < 0:
            Age_difference=0-Age_difference
        Age_difference_list.append(Age_difference)
    df['age_d']=Age_difference_list #添加新列值和新列名
    df.to_csv('dataforanalysis.csv',index=False)
    print("New column age_d has been added to the csv")

def DrawAgeDifferenceHistogram():
    df=pd.read_csv('dataforanalysis.csv')
    data=df['age_d']   #获取第x列的数据
    size_data=int(max(data)-min(data)) #防止直方图不正确
    plt.figure(figsize=(6,2),dpi=80)
    plt.hist(data,bins=int(size_data),edgecolor='black')
    plt.title ('Age Difference Distribution')
    plt.xlabel('Age Difference Groups')
    plt.ylabel('Number of observations')
    ax=plt.gca()
    x_major_locator=MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()

#variable explainings:

#Gender          0:female 1:male
#Age             Age of self
#age_o           Age of partner
#samerace        Whether the two persons have the same race or not 0:no 1:yes

#无用
#attractive_o       Rating by partner (about me) on attractiveness
#sincere_o          Rating by partner (about me) on sincerity
#intelligence_o     Rating by partner (about me) on intelligence
#funny_o            Rating by partner (about me) on being funny
#ambition_o         Rating by partner (about me) on being ambitious
#shared_interests_o Rating by partner (about me) on shared interest

#无用
#attractive        Rate yourself-attractiveness
#sincere           Rate yourself-sincerity
#intelligence      Rate yourself-intelligence
#funny             Rate yourself-being funny
#ambition          Rate yourself-ambition

#100 in total，有用
#attractive_important       What do you look for in a partner-attractiveness
#sincere_important          What do you look for in a partner-sincerity
#intelligence_important     What do you look for in a partner-intelligence
#funny_important            What do you look for in a partner-being funny
#ambition_important         What do you look for in a partner-ambition
#shared_interests_important What do you look for in a partner-shared interests

#有用
#attractive_partner           Rate your partner-attractiveness
#sincere_partner              Rate your partner-sincerity
#intelligence_partner         Rate your partner-intelligence
#funny_partner                Rate your partner-being funny
#ambition_partner             Rate your partner-ambition
#shared_interests_partner     Rate your partner-shared interests

#interests_correlate               Correlation between participant's and partner's ratings of interests
#expected_num_interested_in_me     Out of the 20 people you will meet, how many do you expect will be interested in
#expected_num_matches              How many matches do you expect to get?
#like                              Did you like your partner? 
#guess_prob_liked                  How likely do you think it is that your partner likes you?
#match                             Match 1:yes 0:no

#Draw Pie Chart
def DrawOnePieChart(x):
    df=pd.read_csv('dataforanalysis.csv')
    ReadCsvColumnName()
    attractive_important      =df['attractive_important'][x]
    sincere_important         =df['sincere_important'][x]
    intelligence_important    =df['intelligence_important'][x]
    funny_important           =df['funny_important'][x]
    ambition_important        =df['ambition_important'][x]
    shared_interests_important=df['shared_interests_important'][x]
    pie_chart=np.array([attractive_important,sincere_important,intelligence_important,funny_important,ambition_important,shared_interests_important])
    pie_label=[f"attractive_important {attractive_important}%",
               f"sincere_important {sincere_important}%",
               f"intelligence_important {intelligence_important}%",
               f"funny_important {funny_important}%",
               f"ambition_important {ambition_important}%",
               f"shared_interests_important {shared_interests_important}%"]
    plt.pie(pie_chart, labels=pie_label)
    #plt.legend(title=f"six qualities assigned by {x}")
    plt.show()

def DrawTheFirstOnePieChart():
    DrawOnePieChart(1)
    
def AddARelativeAttributeToCsv(x,y):
    ReadCsvColumnName()
    df=pd.read_csv('dataforanalysis.csv')
    My_list       =list(df[x])
    Partner_list  =list(df[y])
    AttributeSort =str(x).split("_")[0]  #取第一个单词
    Total_list    =[]
    len_excel     =len(My_list)
    for number in range(len_excel):  
        Total=My_list[number]*Partner_list[number]/100
        Total_list.append(Total)
    
    df[f'relative_{AttributeSort}']=Total_list #添加新列值和新列名
    df.to_csv('dataforanalysis.csv',index=False)
    print(f"New column relative_{AttributeSort} has been added to the csv.")
    
def AddAllRelativeAttributesToCsv():
    Attribute_list={"attractive_important":"attractive_partner",
                    "sincere_important":"sincere_partner",
                    "intelligence_important":"intelligence_partner",
                    "funny_important":"funny_partner",
                    "ambition_important":"ambition_partner",
                    "shared_interests_important":"shared_interests_partner"}
    for x in Attribute_list:
        y=Attribute_list[x]
        AddARelativeAttributeToCsv(x,y)
    
#Partitioning data and predicting attrition

def DefineVariablesAndAttributeForMatching(): #设置目标和因素
    global features,target
    features =[ 'relative_attractive',
                'relative_sincere',
                'relative_intelligence',
                'relative_funny',
                'relative_ambition',
                'relative_shared']
    target  =[  'match']

def DataPartition():
    df=pd.read_csv('dataforanalysis.csv')
    DefineVariablesAndAttributeForMatching()
    global x_train, x_test, y_train, y_test
    x=df[features]
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=42) #分割数据成Train and test

def DecisionTree():
    #gain x_train,x_test,y_train,y_test
    DataPartition()  
    
    #train the decision tree
    global dtree
    dtree=DecisionTreeClassifier(max_leaf_nodes=15).fit(x_train,y_train)
    
    #show the decision tree
    plt.figure(figsize=(24, 10))
    plot_tree(dtree, feature_names=features, filled=True, fontsize=10)
    plt.show()
    
#make predictions and evaluate the model
def EvaluatingPredictionDecisionTreeResults():
    DataPartition() #gain x_train,x_test,y_train,y_test
    DecisionTree()  #gain decision tree model
    y_prediction=dtree.predict(x_test)
    
    accuracy    =accuracy_score  (y_test,y_prediction)
    conf_matrix =confusion_matrix(y_test,y_prediction)
    class_report=classification_report(y_test,y_prediction,target_names=['False','True'])
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n",conf_matrix)
    print("Classification Report:\n",class_report)

#full program run
def main_program():
    InputOutputData()
    CsvAddTitle()
    DropNaN()
    ConvertGender()
    ConvertSameraceAndMatch()
    DrawHistograms()
    AddAgeDifferenceToCsv()
    DrawAgeDifferenceHistogram()
    DrawTheFirstOnePieChart()
    AddAllRelativeAttributesToCsv()
    EvaluatingPredictionDecisionTreeResults()

main_program()