"""
Google Play store dataset
"""
"""
A feature to boost visibility for the most promising apps. 
Now to get the most promising apps we would need to understand the features that define a well-performing app.
This analysis helps in finding those promising apps.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") #for ignoring unnecessary warnings due to seaborn
import seaborn as sns

#reading the dataset and checking the rows
path= r'C:\\Users\\AIDEN SAMUEL\\Desktop\\googleplaystore_v2.csv'
data0 = pd.read_csv(path)
print(data0.head())
print(data0.shape)


## Data handling and cleaning process
#Checking the datatype of all the columns of the dataframe
print(data0.info())

## Missing Value Treatment

print(data0.isnull().sum())


data1 = data0[~data0.Rating.isnull()] #complement of the null function which will drop the null values
print(data1.shape)    #shape has reduced


print(data1.isnull().sum()) #cross-checking

#Inspecting the nulls in the Android Version column
print(data1[data1['Android Ver'].isnull()])
pd.set_option('display.max_columns', None)     # just for showing all the rows which got truncated
print(data1[data1['Android Ver'].isnull()])


print(data1.loc[10472,:])      #Dropping the rows having shifted values
data1=data1[~(data1['Android Ver'].isnull() & (data1.Category=="1.9"))]

print(data1[data1['Android Ver'].isnull()]) #cross-checking

print(data1['Android Ver'].mode()[0])   #in string form the value to be inserted in missing places

#Filling up the nulls in the Android Version column with the above value
pd.set_option('mode.chained_assignment',None)   #according to pandas documentation settingwithcopy warning
                                                # may be a false positive sometimes hence ignored error
data1['Android Ver']=data1['Android Ver'].fillna(data1['Android Ver'].mode()[0])
# print(data1.isnull().sum())

#checking nulls in df
print(data1['Android Ver'].isnull().sum())

print(data1['Current Ver'].mode()[0])
print(data1['Current Ver'].isnull())
#its a string value hence

#Replacing with the string above
data1['Current Ver']=data1['Current Ver'].fillna(data1['Current Ver'].mode()[0])
print(data1['Current Ver'].isnull().sum())

# print(data1['Android Ver'].describe())      solving error ignore line

print(data1.dtypes)  # checking if dtypes are proper

#value counts of all variables in price column
print(data1['Price'].value_counts())

#replacing values
data1['Price'] = data1['Price'].apply(lambda x:0 if x=='0' else float(x[1:]))

#checking dtypes after replacement
print(data1['Price'].dtype)

#Reviews column
print(data1['Reviews'].value_counts())
#it is all in int, dtype is not int only because earlier there was a row with a string that row got dropped

#Changing the dtype of this column
data1['Reviews']=data1['Reviews'].astype('int32')

# shows the quantitative values of df
print(data1['Reviews'].describe())

print(data1['Installs'].head())  # comma and plus symbol is seen because of which dtype is wrong

#Installs columnn
#Number of apps at the 50th percentile.
data1['Installs']=data1['Installs'].str.replace('+','', regex=True)
data1['Installs']=data1['Installs'].str.replace(',','', regex=True)
data1['Installs']=data1['Installs'].astype('int32')
print(data1['Installs'].head())

#to ensure data is factually right

#Ratings and Reviews column
print(data1['Rating'].describe())  #using describe for looking at max value
print(data1[data1['Reviews']>data1['Installs']].shape) # for finding out the number of apps that meet the condition we use shape
data1=data1[data1['Reviews']<=data1['Installs']]
#print(data1[data1['Reviews']>data1['Installs']].shape)

#performing checks on prices of free apps
print(data1[(data1['Price']>0) & (data1['Type']=='Free')].shape)

#For outliers in the dataset using boxplots

#Starting with price creating boxplots

print(data1['Price'].describe())
plt.boxplot(data1['Price'])
plt.show()

#Checking the apps with price more than 200
#print(data1[data1['Price']>200])
print(data1['Price'].shape)

#Cleaning the Price column
data1=data1[data1['Price']<200]
#print(data1['Price'].describe())

#Box plot but now for paid apps only
# data1[data1['Price']>0].Price.plot.box()

print(data1[data1['Price']>30])

#Removing enteries
data1 =data1[data1['Price']<=30]
#print(data1['Price'].describe())

#histograms
#histogram of the Reviews
plt.hist(data1['Reviews'])
plt.show()

# plt.boxplot(data1['Reviews'])
# plt.show()

#wont help us as these apps are already popular
print(data1[data1['Reviews']>1000000])

#Dropping the above records as they won't really help in the main aim of finding promising apps
data1=data1[data1['Reviews']<1000000]

# print(data1['Reviews'].value_counts())
#print(data1['Reviews'].describe())

# plt.hist(data1['Reviews'])
# plt.show()

plt.boxplot(data1['Installs'])
plt.show()
# print(data1['Installs'].describe())
# 1 * 10^6-  1*10^4 = 1e+06

#same thing as above removing apps that wont help
data1=data1[data1['Installs']<=100000000]
print(data1['Installs'].describe())

# plt.hist(data1['Size'])
# plt.show()

# plt.boxplot(data1['Size'])
# plt.show()

"""using seaborn"""
#import the necessary libraries
# import warnings
# warnings.filterwarnings("ignore") #for ignoring unnecessary warnings due to seaborn
# import seaborn as sns

#Distribution plot for rating
plt.style.use("dark_background")         #matplot and seaborn aesthetic together
sns.set_style("darkgrid")                 #sets basic things like grid
plt.style.use("dark_background")         #matplot and seaborn aesthetic together
#
sns.distplot(data1['Rating'],bins=20,kde=True,color='b')    #kde
# plt.title("Distribution of app ratings" ,fontsize=14)
# sns.countplot(data1['Rating'])

#all possible styles in sns
# print(plt.style.available)

plt.show()

#pie charts here
# print(data1['Content Rating'].value_counts())

#Removing the rows with values which are less represented hardly any
data1=data1[~data1['Content Rating'].isin(["Adults only 18+","Unrated"])]

#resetting
data1.reset_index(inplace=True,drop=True)

print(data1['Content Rating'].value_counts())
print(data1.shape)

#pie
# data1['Content Rating'].value_counts().plot.pie()
# plt.show()

#some things cannot be seen clearly so bar instead
data1['Content Rating'].value_counts().plot.bar()
plt.show()

print(data1['Android Ver'].value_counts())
# data1['Android Ver'].value_counts().plot.bar()
# plt.show()


###Size vs Rating
#using scatter plots
sns.scatterplot(data1.Size,data1.Rating)
plt.show()

sns.jointplot(data1.Size,data1.Rating,kind='kde',color='g')        #kind=kde can be removed later
plt.show()

sns.jointplot(data1.Price,data1.Rating,kind='kde',color='g')
plt.show()

#reg plot for Price and Rating
# sns.jointplot(data1.Price,data1.Rating,kind='kde',color='g')
# plt.show()

#paid apps only now
sns.jointplot(data1.Price>1,data1.Rating,kind='reg',color='g')
plt.show()

#pair plots here

#Pair plots for Reviews, Size, Price and Rating
sns.pairplot(data1[['Size','Rating','Price','Reviews']])
plt.show()
#shows the trends all together , several inferences can be made

#bar plots with estimator using seaborn and percentiles to make it better

sns.barplot(data=data1,x='Content Rating',y='Rating',estimator=lambda x: np.quantile(x,0.05))
plt.show()

sns.barplot(data=data1,x='Content Rating',y='Rating',estimator=np.min)
plt.show()

#rating vs content rating box plots as much better to look at all the ratings together
# # plt.figure(figsize=[7,9])        # for changing proportions
sns.boxplot(data1['Rating'],data1['Content Rating'])
plt.show()

# sns.boxplot(data1['Rating'])
# plt.show()

#across most popular genres
print(data1.Genres.value_counts())
temp=['Tools','Entertainment','Education','Medical']
data_temp=data1[data1['Genres'].isin(temp)]
# sns.boxplot(data_temp['Genres'],data1['Rating'])
# plt.show()

#heatmaps here

#Ratings vs Size vs Content Rating

#qcut now for binning
data1['Size_bins']=pd.qcut(data1['Size'],[0,0.2,0.4,0.6,0.8,1],['VL','L','M','H','VH'])
print(data1.head())

#grid as pivot table for heatmap
#content rating is age groups
#aggregation to 20th percentile
heatmap_grid=pd.pivot_table(data=data1,index='Content Rating',columns='Size_bins',values='Rating',aggfunc=lambda x: np.quantile(x,0.2))

# cmap stands for colormaps using greens here as easy to understand for viewers
sns.heatmap(heatmap_grid,cmap='Greens',annot=True,linecolor=['White'],linewidths=1)
plt.show()

# comparing reviews and size and ratings
#creating reviews bins
data1['Reviews_bins']=pd.qcut(data1['Reviews'],[0,0.2,0.4,0.6,0.8,1],['VL','L','M','H','VH'])
heatmap_grid2=pd.pivot_table(data=data1,index='Reviews_bins',columns='Size_bins',values='Rating',aggfunc=lambda x: np.min(x))
# sns.heatmap(heatmap_grid2,cmap='Greens')
# plt.show()

#to get months from the series and using it to compare acc to months
#rating across months

data1['Updated_months']=pd.to_datetime(data1['Last Updated']).dt.month
# data1.groupby(data1['Updated_months'])['Rating'].mean().plot()      #applying mean on ratings
# plt.show()

#stacked bar charts

#pivott table for Content Rating and updated Month with the values set to Installs
month_table=pd.pivot_table(data=data1,index='Updated_months',columns='Content Rating',values='Installs',aggfunc=sum)

#acc to proportions for that
months_prop=month_table[['Everyone','Everyone 10+','Mature 17+','Teen']].apply(lambda x: x/x.sum(),axis=1)

# dividing by the total to get all the values in proportion all the bars at the same line to be compared better
months_prop.plot(kind='bar',stacked='True') #for stacked bars
plt.show()


# end of visualizations
