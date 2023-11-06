from pandas.testing import assert_frame_equal, assert_series_equal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from IPython.display import display
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


############################################################
#                         Data                             #
############################################################
# A DataFrame with the original data.
DATA_0 = pd.read_csv("/mnt/work/workbench/dipendrp/data/TilNTNU.csv", sep=";")


def get_renamed_data() -> pd.DataFrame:
    data = DATA_0.copy(deep=True)
    renamed = data.rename(
        columns={
            "Løpenummer": "id",
            "Kjønn": "gender",
            "Diagnosekode": "diagnosis_code",
            "Diagnosetekst": "diagnosis_text",
            "Alder_førstekontakt_dager": "age_first_contact_days",
            "Alder_sistekontakt_dager": "age_last_contact_days",
            "Alder_diagnose_dager": "age_diagnosis_days",
            "Alder_død_dager": "age_dead_days",
            "Alder_diagnose_aar": "age_diagnosis_year",
            "Alder_aar_førstekontakt": "age_first_contact_year",
            "Alder_aar_sistekontakt": "age_last_contact_year",
            "Alder_aar_død": "age_dead_year",
        })
    return renamed


# A DataFrame with the original data where:
# 1. the column names are renamed
DATA_1 = get_renamed_data()


def is_empty_str(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return text.strip() == ""


def get_na_converted_data() -> pd.DataFrame:
    na_converted_df = DATA_1.copy(deep=True)
    for col in DATA_1.columns:
        rows_with_na = DATA_1.loc[:, col].apply(is_empty_str)
        if sum(rows_with_na):
            na_converted_df.loc[rows_with_na, col] = np.nan
    return na_converted_df


# A DataFrame with the original data where:
# 1. the column names are renamed.
# 2. nans' are properly formated.
DATA_2 = get_na_converted_data()


def get_formatted_data() -> pd.DataFrame:
    formatted_df = DATA_2.copy(deep=True)
    # Gender
    formatted_df["gender"] = formatted_df["gender"].replace(
        "M", "male")
    formatted_df["gender"] = formatted_df["gender"].replace(
        "K", "female")
    formatted_df["gender"] = formatted_df["gender"].astype(
        "category")
    # Diagnosis Code and Diagnosis Text
    for col in ("diagnosis_code", "diagnosis_text"):
        formatted_df[col] = formatted_df[col].astype("string")

    # From column four to the end, which all contain integers
    for col in formatted_df.iloc[:, 4:].columns:
        formatted_df[col] = formatted_df[col].astype("Float32").astype("Int32")

    return formatted_df


# A DataFrame with the original data where:
# 1. the column names are renamed.
# 2. nans' are properly formated.
# 3. each column is converted to the correct data type
DATA_3 = get_formatted_data()

# A DataFrame with the original data where:
# 1. the column names are renamed.
# 2. NaNs' are properly formated.
# 3. each column is converted to the correct data type
# 4. All rows with NaNs are removed
DATA_4 = DATA_3.dropna(axis=0)

def gender_get_formatted_data() -> pd.DataFrame:
    DATA_5 = DATA_4.copy(deep=True)
      # Gender
    DATA_5["gender"] = DATA_5["gender"].replace(
        "male", 0)
    DATA_5["gender"] = DATA_5["gender"].replace(
        "female", 1)
    
    return DATA_5

# A DataFrame with the original data where:
# 1. gender are codes as "male" as 0 and "female" as 1
# 2. other remains same as DATA_4
DATA_5 = gender_get_formatted_data()


def diagnosis_code_get_formatted_data() -> pd.DataFrame:
    DATA_66 = DATA_5.copy(deep=True)
    
    def map_codes(code):
        if code.startswith('A'):
            return 'A'
        elif code.startswith('B'):
            return 'B'
        elif code.startswith('C'):
            return 'C'
        elif code.startswith('D'):
            return 'D'
        elif code.startswith('E'):
            return 'E'
        elif code.startswith('F'):
            return 'F'
        elif code.startswith('G'):
            return 'G'
        elif code.startswith('H'):
            return 'H'
        elif code.startswith('I'):
            return 'I'
        elif code.startswith('J'):
            return 'J'
        elif code.startswith('K'):
            return 'K'
        elif code.startswith('L'):
            return 'L'
        elif code.startswith('M'):
            return 'M'
        elif code.startswith('N'):
            return 'N'
        elif code.startswith('O'):
            return 'O'
        elif code.startswith('P'):
            return 'P'
        elif code.startswith('Q'):
            return 'Q'
        elif code.startswith('R'):
            return 'R'
        elif code.startswith('S'):
            return 'S'
        elif code.startswith('T'):
            return 'T'
        elif code.startswith('U'):
            return 'U'
        elif code.startswith('V'):
            return 'V'
        elif code.startswith('X'):
            return 'X'
        elif code.startswith('Y'):
            return 'Y'
        elif code.startswith('Z'):
            return 'Z'
        else:
            return 'Others'

        # Apply the function to the diagnosis_code column
    DATA_66['diagnosis_code'] = DATA_66['diagnosis_code'].apply(map_codes)

    return DATA_66

# A DataFrame with the original data where:
# 1. gender are codes as "male" as 0 and "female" as 1
# 2. other remains same as DATA_4
DATA_66 = diagnosis_code_get_formatted_data()

# A DataFrame with the original data where:
# 1. For NLP tasks
DATA_20 = DATA_66.copy(deep=True)

############################################################
#                  Printing Functions                      #
############################################################


def prRed(string: str) -> None:
    print(f"\033[91m {string}\033[00m")


def prGreen(string: str) -> None:
    print(f"\033[92m {string}\033[00m")


############################################################
#                        Question 1                         #
############################################################
def q1_check(answer1: pd.DataFrame, answer2: pd.DataFrame) -> None:
    solution1 = DATA_4[['age_first_contact_year', 'age_diagnosis_year', 'age_last_contact_year']]
    solution2 = DATA_4[['age_dead_year']]
    try:
        assert_frame_equal(solution1, answer1)
        assert_frame_equal(solution2, answer2)
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q1_solution():
    print("independent_variable = DATA_4[['age_first_contact_year', 'age_diagnosis_year', 'age_last_contact_year']]")
    print("dependent_variable = DATA_4[['age_dead_year']]")
    
    
############################################################
#                        Question 2                          #
############################################################
def q2_check(answer1: str, answer2: str, answer3: str, answer4: str, answer5: str, answer6: str) -> None:
    solution1 = "(80640, 3)"
    solution2 = "(23155, 3)"
    solution3 = "(11405, 3)"
    solution4 = "(80640, 1)"
    solution5 = "(23155, 1)"
    solution6 = "(11405, 1)"
    
    try:
        if (solution1 == answer1) & (solution2 == answer2) & (solution3 == answer3) & (solution4 == answer4) & (solution5 == answer5) & (solution6 == answer6):
            prGreen("Correct")
        else:
            prRed("Incorrect")
            
    except Exception:
        prRed("Incorrect")
        

def q2_solution():
    print("independent_variable = DATA_4[['age_first_contact_year', 'age_diagnosis_year', 'age_last_contact_year']]")
    print("dependent_variable = DATA_4[['age_dead_year']]")
    print("X_train, X_temp, y_train, y_temp = train_test_split(independent_variable,dependent_variable,test_size=0.3)")
    print("X_test,X_eval,y_test,y_eval = train_test_split(X_temp,y_temp,test_size=0.33)")
    
    
############################################################
#                        Question 3                          #
############################################################
def q3_check(answer1: str) -> None:
    solution1 = "0.9"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q3_solution():
    print("independent_variable = DATA_4[['age_first_contact_year', 'age_diagnosis_year','age_last_contact_year']]")
    print("dependent_variable = DATA_4[['age_dead_year']]")
    print("X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)")
    print("X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)")
    print("model = LinearRegression()")
    print("model.fit(X_train, y_train)")
    
############################################################
#                        Question 4                          #
############################################################
def q4_check(age_first_contact_year: int,age_last_contact_year: int,age_diagnosis_year: int, answer1:int) -> None:
    independent_variable = DATA_4[['age_first_contact_year', 'age_diagnosis_year','age_last_contact_year']]
    dependent_variable = DATA_4[['age_dead_year']]
    X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)
    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict([[age_first_contact_year, age_last_contact_year, age_diagnosis_year]])
    prediction_str = str(prediction[0][0])
    solution1 = int(prediction_str[0:1])   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q4_solution() -> None:
    print("prediction = model.predict([[age_first_contact_year, age_last_contact_year, age_diagnosis_year]])")
    
############################################################
#                        Question 5.5                          #
############################################################
def q55_check(age_first_contact_year: int,age_last_contact_year: int,age_diagnosis_year: int, answer1:int) -> None:
    independent_variable = DATA_4[['age_first_contact_year', 'age_diagnosis_year', 'age_last_contact_year']]
    dependent_variable = DATA_4[['age_dead_year']]
    X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)
    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict([[age_first_contact_year, age_last_contact_year, age_diagnosis_year]])
    prediction_str = str(prediction[0][0])
    solution1 = int(prediction_str[0:1])   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q55_solution() -> None:
    print("independent_variable = DATA_4[['age_first_contact_year', 'age_diagnosis_year', 'age_last_contact_year']]")
    print("dependent_variable = DATA_4[['age_dead_year']]")
    print("X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)")
    print("X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)")
    print("model = LinearRegression()")
    print("model.fit(X_train, y_train)")
    print("prediction = model.predict([[age_first_contact_year, age_last_contact_year, age_diagnosis_year]])")

    
############################################################
#                        Question 6                          #
############################################################
def q6_check(gender:int,age_first_contact_year:int,age_last_contact_year:int,age_diagnosis_year:int,answer1:int) -> None:
    independent_variable = DATA_5[['gender', 'age_first_contact_year', 'age_last_contact_year', 'age_diagnosis_year']]
    dependent_variable = DATA_5[['age_dead_year']]
    X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)
    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)
    model_lin = LinearRegression()
    model_lin.fit(X_train, y_train)
    prediction = model_lin.predict([[gender,age_first_contact_year, age_last_contact_year, age_diagnosis_year]])
    prediction_str = str(prediction[0][0])
    solution1 = int(prediction_str[0:1])
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q6_solution() -> None:
    print("independent_variable = DATA_5[['gender', 'age_first_contact_year', 'age_last_contact_year', 'age_diagnosis_year']]")
    print("dependent_variable = DATA_5[['age_dead_year']]")
    print("X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)")
    print("X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)")
    print("model_lin = LinearRegression()")
    print("model_lin.fit(X_train, y_train)")
    print("prediction = model_lin.predict([[gender,age_first_contact_year, age_last_contact_year, age_diagnosis_year]])")  
    print("prediction_str = str(prediction[0][0])")
    print("print('Linear regression predicted dead year age:', prediction[0][0])")
    
############################################################
#                        Question 7                          #
############################################################
def q7_check(gender:int,age_first_contact_year:int,age_last_contact_year:int,age_diagnosis_year:int,answer1:int) -> None:
    independent_variable = DATA_5[['gender', 'age_first_contact_year', 'age_last_contact_year', 'age_diagnosis_year']]
    dependent_variable = DATA_5[['age_dead_year']]
    X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)
    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train,y_train)
    prediction = model_logistic.predict([[gender,age_first_contact_year,age_last_contact_year,age_diagnosis_year]])
    prediction_str = str(prediction[0])
    solution1 = int(prediction_str[0:1])
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q7_solution() -> None:
    print("independent_variable = DATA_5[['gender', 'age_first_contact_year', 'age_last_contact_year', 'age_diagnosis_year']]")
    print("dependent_variable = DATA_5[['age_dead_year']]")
    print("X_train, X_temp, y_train, y_temp = train_test_split(independent_variable, dependent_variable, test_size=0.3)")
    print("X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.33)")
    print("model_logistic = LogisticRegression()")
    print("model_logistic.fit(X_train,y_train)")
    print("prediction = model_logistic.predict([[gender,age_first_contact_year,age_last_contact_year,age_diagnosis_year]])")
    print("print('Logistic Regression prediction for age death year:', int(prediction[0]))")
            
            
############################################################
#                        Question 8                         #
############################################################
def q8view_kmean_suitable_number_clusters(data):
    k_range = range(1, 10)
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.plot(k_range, wcss, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of the variance in each cluster (WCSS)')
    plt.title('Elbow for K-Means Cluster')
    plt.show()

############################################################
#                        Question 9                         #
############################################################
def q9_check():
    selected_column_DATA_5 = DATA_5[['age_first_contact_year', 'age_last_contact_year', 'age_diagnosis_year','age_dead_year']].copy()
    selected_column_DATA_5 = selected_column_DATA_5.iloc[:500]
    kmeans = KMeans(n_clusters=3)
    cluster_assignments = kmeans.fit_predict(selected_column_DATA_5)
    selected_column_DATA_5['Cluster'] = cluster_assignments
    print(selected_column_DATA_5['Cluster'].value_counts())
    plt.scatter(selected_column_DATA_5["age_first_contact_year"], selected_column_DATA_5["age_dead_year"], c=selected_column_DATA_5["Cluster"], cmap="rainbow")
    plt.xlabel("age_first_contact_year")
    plt.ylabel("age_dead_year")
    plt.title("Kmeans Clustering")
    plt.show() 

        

def q9_solution() -> None:
    print("kmeans = KMeans(n_clusters=3)")
    print("cluster_assignments = kmeans.fit_predict(selected_column_DATA_5)")
    print("selected_column_DATA_5['Cluster'] = cluster_assignments")
    print("print(selected_column_DATA_5['Cluster'].value_counts())")
    print("plt.scatter(selected_column_DATA_5['age_first_contact_year'], selected_column_DATA_5['age_dead_year'], c=selected_column_DATA_5['Cluster'], cmap='rainbow')")
    print("plt.xlabel('age_first_contact_year')")
    print("plt.ylabel('age_dead_year')")
    print("plt.title('Kmeans Clustering')")
    print("plt.show()")

############################################################
#                        Question 10                         #
############################################################
def q10view_kmean_suitable_number_clusters(data):
    k_range = range (1, 10)
    wcss = []
    for k in k_range:
        kproto = KPrototypes (n_clusters=k, init='Cao')
        clusters = kproto.fit_predict (data, categorical=[0])
        wcss.append (kproto.cost_)

    plt.plot (k_range, wcss, marker='o')
    plt.xlabel ('Number of clusters')
    plt.ylabel ('Sum of the variance in each cluster (WCSS)')
    plt.title ('Elbow plot for k-prototypes')
    plt.show ()

############################################################
#                        Question 11                         #
############################################################
def q11_check():
    selected_column_DATA_5 = DATA_5[['age_first_contact_year', 'age_last_contact_year', 'age_diagnosis_year','age_dead_year']].copy()
    selected_column_DATA_5 = selected_column_DATA_5.iloc[:500]
    kproto = KPrototypes(n_clusters=3,init='Cao')
    cluster_assignments = kproto.fit_predict(selected_column_DATA_5,categorical=[0])
    selected_column_DATA_5['Cluster'] = cluster_assignments
    print(selected_column_DATA_5['Cluster'].value_counts())
    plt.scatter(selected_column_DATA_5["age_first_contact_year"], selected_column_DATA_5["age_dead_year"], c=selected_column_DATA_5["Cluster"], cmap="rainbow")
    plt.xlabel("age_first_contact_year")
    plt.ylabel("age_dead_year")
    plt.title("KPrototypes Clustering")
    plt.show() 

        

def q11_solution() -> None:
    print("kproto = KPrototypes(n_clusters=3,init='Cao')")
    print("cluster_assignments = kproto.fit_predict(selected_column_DATA_5,categorical=[0])")
    print("selected_column_DATA_5['Cluster'] = cluster_assignments")
    print("print(selected_column_DATA_5['Cluster'].value_counts())")
    print("plt.scatter(selected_column_DATA_5['age_first_contact_year'], selected_column_DATA_5['age_dead_year'], c=selected_column_DATA_5['Cluster'], cmap='rainbow')")
    print("plt.xlabel('age_first_contact_year')")
    print("plt.ylabel('age_dead_year')")
    print("plt.title('KPrototypes Clustering')")
    print("plt.show()")
    

############################################################
#                        Question 14                          #
############################################################
def q12_check(answer1:str) -> None:
    solution1 = "1"
    
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q12_solution() -> None:
    print("1")
    
    
############################################################
#                        Question 15                          #
############################################################
def q13_check(answer1: str) -> None:
    solution1 = "0.8"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q13_solution() -> None:
    print("X = DATA_20['diagnosis_text']")
    print("y = DATA_20['diagnosis_code']")
    print("vectorizer = CountVectorizer()")
    print("X = vectorizer.fit_transform(X)")
    print("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)")
    print("classifier = MultinomialNB()")
    print("classifier.fit(X_train, y_train)")
    print("accuracy = classifier.score(X_test, y_test)")
    print("print(f'Accuracy: {accuracy}')")
    print("acc = str(accuracy)")
    
############################################################
#                        Question 16                          #
############################################################
def q14_check(answer1: str) -> None:
    solution1 = "D"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q14_solution() -> None:
    print("diagnosis_text = input('Enter the diagnosis text (in Norwegian) i.e. Uspesifisert aplastisk anemi: ')")
    print("X_user = vectorizer.transform([diagnosis_text])")
    print("diagnosis_code = classifier.predict(X_user)")
    print("print('The predicted diagnosis code for is:', diagnosis_code[0])")
    
    
############################################################
#                        Question 17                          #
############################################################
def q15_check(answer1: str) -> None:
    solution1 = "I"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q15_solution() -> None:
    print("diagnosis_text = input('Enter the diagnosis text (in Norwegian) i.e, ikke-reumatisk aortainsuffisi: ')")
    print("X_user = vectorizer.transform([diagnosis_text])")
    print("diagnosis_code = classifier.predict(X_user)")
    print("print('The predicted diagnosis code', diagnosis_code[0])")
    