from pandas.testing import assert_frame_equal, assert_series_equal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

############################################################
#                         Data                             #
############################################################
# A DataFrame with the original data.
DATA_0 = pd.read_csv("../data/TilNTNU.csv", sep=";")


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
#                        Question 1.1                          #
############################################################
def q1_1_check(answer: pd.DataFrame) -> None:
    solution = DATA_4.loc[:,['age_first_contact_days', 'age_last_contact_days','age_dead_days']]
    try:
        assert_frame_equal(solution, answer)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")


def q1_1_solution() -> None:
    print("days_only_df = DATA_4.loc[:,['age_first_contact_days', 'age_last_contact_days','age_dead_days']]")
    print("q1_1_check(days_only_df)")
    print('days_only_df.head(4).style.hide(axis="index")')

    

############################################################
#                        Question 1.2                          #
############################################################
def q1_2_check(answer: pd.DataFrame) -> None:
    solution = DATA_4[['age_first_contact_days', 'age_last_contact_days','age_dead_days']]
    try:
        assert_frame_equal(solution, answer)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")


def q1_2_solution() -> None:
    print("days_only_df = DATA_4[['age_first_contact_days', 'age_last_contact_days','age_dead_days']]")
    print("q1_2_check(days_only_df)")
    print('days_only_df.head(4).style.hide(axis="index")')
    
    

############################################################
#                        Question 2                          #
############################################################
def q2_check(answer1: pd.DataFrame, answer2: pd.DataFrame) -> None:
    solution1 = DATA_4[['age_first_contact_days', 'age_last_contact_days']]
    solution2 = DATA_4[['age_dead_year']]
    try:
        assert_frame_equal(solution1, answer1)
        assert_frame_equal(solution2, answer2)
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q2_solution() -> None:
    print("independent_variable = DATA_4[['age_first_contact_days', 'age_last_contact_days']]")
    print("dependent_variable = DATA_4[['age_dead_year']]")
    print("q2_check(independent_variable,dependent_variable)")
    print("display(independent_variable.head(4).style.hide(axis='index'))")
    print("display(dependent_variable.head(4).style.hide(axis='index'))")
    
    
############################################################
#                        Question 3                          #
############################################################
def q3_check(answer1: str, answer2: str, answer3: str, answer4: str) -> None:
    solution1 = "(92160, 2)"
    solution2 = "(23040, 2)"
    solution3 = "(92160, 1)"
    solution4 = "(23040, 1)"
    
    try:
        if (solution1 == answer1) & (solution2 == answer2) & (solution3 == answer3) & (solution4 == answer4):
            prGreen("Correct")
        else:
            prRed("Incorrect")
            
    except Exception:
        prRed("Incorrect")
        

def q3_solution() -> None:
    print("X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)")
    print("print(f'Size of X_train:{X_train.shape},X_test:{X_test.shape},y_train:{y_train.shape},y_test:{y_test.shape}')")
    
############################################################
#                        Question 4.1                          #
############################################################
def q4_1_check(answer1: str) -> None:
    solution1 = "model.fit(X_train, y_train)"
    solution2 = "model.fit(X_train,y_train)"
    solution3 = "model.fit( X_train, y_train)"
    try:
        if solution1 == answer1 or solution2 == answer1 or solution2 == answer1:
            prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q4_1_solution() -> None:
    print("code_to_fit_model = 'model.fit(X_train, y_train)'")
    print("print(code_to_fit_model)")
    
############################################################
#                        Question 4.2.                          #
############################################################
def q4_2_check(answer1:pd.DataFrame) -> None:
    DATA_4_2 = DATA_4.copy()
    gender_df = DATA_4_2[['gender']]
    encoder = OneHotEncoder(sparse_output=False)
    gender_encoded = encoder.fit_transform(gender_df)
    gender_encoded_df = pd.DataFrame(gender_encoded.astype("int32"), columns=['gender_male', 'gender_female'])
    DATA_4_2 = pd.concat([DATA_4_2, gender_encoded_df], axis=1)
    DATA_4_2.drop('gender', axis=1, inplace=True)
    
    try:
        assert_frame_equal(DATA_4_2, answer1)
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q4_2_solution() -> None:
    print("DATA_4_2 = DATA_4.copy(deep=True)") 
    print("gender_df = DATA_4_2[['gender']]")
    print("encoder = OneHotEncoder(sparse_output=False)")
    print("gender_encoded = encoder.fit_transform(gender_df)")
    print("gender_encoded_df = pd.DataFrame(gender_encoded.astype('int32'), columns=['gender_male', 'gender_female'])")
    print("DATA_4_2 = pd.concat([DATA_4_2, gender_encoded_df], axis=1)")
    print("DATA_4_2.drop('gender', axis=1, inplace=True)")   
    
    
############################################################
#                        Question 5                          #
############################################################
def q5_check(answer1: str) -> None:
    solution1 = "0.8"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q5_solution() -> None:
    print("independent_variable = DATA_4[['age_first_contact_days', 'age_last_contact_days', 'age_dead_days', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]")
    print("dependent_variable = DATA_4[['age_diagnosis_year']]")
    print("X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)")
    print("model = LinearRegression()")
    print("model.fit(X_train, y_train)")
    print("score = model.score(X_test, y_test)")
    print("print('R^2 score:', score)")
    
############################################################
#                        Question 6                          #
############################################################
def q6_check(answer1: str) -> None:
    solution1 = "1"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q6_solution() -> None:
    print("1") 

    
    
############################################################
#                        Question 7                          #
############################################################
def q7_check(answer1: str) -> None:
    solution1 = "61"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q7_solution() -> None:
    print("age_first_contact_days = int(input('Enter age at first contact in days as: 23215 = '))") 
    print("age_last_contact_days = int(input('Enter age at last contact in days as: 23373 = '))") 
    print("age_dead_days = int(input('Enter age at death in days as: 24073 = '))") 
    print("age_first_contact_year = int(input('Enter year of first contact as: 63 = '))") 
    print("age_last_contact_year = int(input('Enter year of last contact as: 63 = '))") 
    print("age_dead_year = int(input('Enter year of death as: 65 = '))") 
    print("prediction = model.predict([[age_first_contact_days, age_last_contact_days, age_dead_days, age_first_contact_year, age_last_contact_year, age_dead_year]])") 
    print("print('Linear regression model prediction for age_diagnosis_year is:', prediction[0][0])")

    
############################################################
#                        Question 8                          #
############################################################
def q8_check(answer1:str) -> None:
    solution1 = "61"
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q8_solution() -> None:
    print("independent_variable = DATA_5[['gender','age_first_contact_days', 'age_last_contact_days', 'age_dead_days', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]") 
    print("dependent_variable = DATA_5[['age_diagnosis_year']]")
    print("X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)")
    print("model_lin = LinearRegression()")
    print("model_lin.fit(X_train, y_train)")
    print("score = model_lin.score(X_test, y_test)")
    print("print('R^2 score:', score)\n")
    
    print("gender = int(input('Enter gender as 1 = '))")
    print("age_first_contact_days = int(input('Enter age at first contact in days as 23215 = '))")
    print("age_last_contact_days = int(input('Enter age at last contact in days as 23373 =  '))")
    print("age_dead_days = int(input('Enter age at death in days as 24073 = '))")
    print("age_first_contact_year = int(input('Enter year of first contact as 63 = '))")
    print("age_last_contact_year = int(input('Enter year of last contact as 63 = '))")
    print("age_dead_year = int(input('Enter year of death as 65 = '))")
    
    
    print("prediction = model_lin.predict([[gender,age_first_contact_days, age_last_contact_days, age_dead_days, age_first_contact_year, age_last_contact_year, age_dead_year]])")
    print("print('Linear regression model prediction for age_diagnosis_year is:', prediction[0][0])")       
            
            
############################################################
#                        Question 9                          #
############################################################
def q9_check(answer1: str) -> None:
    solution1 = "model_lin"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q9_solution() -> None:
    print("model_lin : because it has higher R square value")

    
############################################################
#                        Question 10                          #
############################################################
def q10_check(gender: int,age_first_contact_days: int, age_last_contact_days: int, age_dead_days: int, age_first_contact_year: int, 
                   age_last_contact_year: int, age_dead_year: int,answer1:str) -> None:
    independent_variable = DATA_5[['gender','age_first_contact_days', 'age_last_contact_days', 'age_dead_days', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]
    dependent_variable = DATA_5[['age_diagnosis_year']]
    X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)

    model_ran = RandomForestClassifier()
    model_ran.fit(X_train, y_train) 
    prediction = model_ran.predict([[gender,age_first_contact_days, age_last_contact_days, age_dead_days, age_first_contact_year, 
                   age_last_contact_year, age_dead_year]])
    solution1 = str(prediction[0])[:1]
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q10_solution() -> None:
    print("independent_variable = DATA_5[['gender','age_first_contact_days', 'age_last_contact_days', 'age_dead_days', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]")
    print("dependent_variable = DATA_5[['age_diagnosis_year']]")
    print("X_train, X_test, y_train, y_test = train_test_split(independent_variable,dependent_variable,test_size=0.2)")
    print("model_ran = RandomForestClassifier()")
    print("model_ran.fit(X_train, y_train)")
    print("score = model_ran.score(X_test, y_test)")
    print("print('R^2 score:', score)\n")
    
    print("gender = int(input('Enter gender as: 1 '))")
    print("age_first_contact_days = int(input('Enter age at first contact in days as: 23215 = '))")
    print("age_last_contact_days = int(input('Enter age at last contact in days as: 23373 = '))")
    print("age_dead_days = int(input('Enter age at death in days as: 24073 = '))")
    print("age_first_contact_year = int(input('Enter year of first contact as: 63 = '))")
    print("age_last_contact_year = int(input('Enter year of last contact as: 63 = '))")
    print("age_dead_year = int(input('Enter year of death as: 65  = '))")
    
    
    print("prediction = model_ran.predict([[gender,age_first_contact_days, age_last_contact_days, age_dead_days, age_first_contact_year, age_last_contact_year, age_dead_year]])")
    print("print('Random Forest model prediction for age_diagnosis_year is:', prediction[0])")
    
############################################################
#                        Question 11                          #
############################################################
def q11_check(gender,age_first_contact_days, age_last_contact_days, age_dead_days,answer1: str) -> None:
    independent_variable = DATA_5[['gender','age_first_contact_days', 'age_last_contact_days', 'age_dead_days']]
    dependent_variable = DATA_5[['age_diagnosis_year']]
    X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)

    model_log = LogisticRegression()
    model_log.fit(X_train, y_train)
    prediction = model_log.predict([[gender,age_first_contact_days, age_last_contact_days, age_dead_days]])
    solution1 = str(prediction[0])[:2]
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q11_solution() -> None:
    print("independent_variable = DATA_5[['gender','age_first_contact_days', 'age_last_contact_days', 'age_dead_days']]") 
    print("dependent_variable = DATA_5[['age_diagnosis_year']]")
    print("X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)")
    print("model_logistic = LogisticRegression()")
    print("model_logistic.fit(X_train, y_train)")
    print("score = model_logistic.score(X_test, y_test)")
    print("print('R^2 score:', score)\n")
    
    print("gender = int(input('Enter gender: (0 for male and 1 for female) = '))")
    print("age_first_contact_days = int(input('Enter age at first contact in days (In dataset max is 35159 and min is 0) = '))")
    print("age_last_contact_days = int(input('Enter age at last contact in days (In dataset max is 36956 and min is 0) = '))")
    print("age_dead_days = int(input('Enter age at death in days (In dataset max is 37692 and min is 0) = '))")
    
    
    print("prediction = model_logistic.predict([[gender,age_first_contact_days, age_last_contact_days, age_dead_days]])")
    print("print('Random Forest model prediction for age_diagnosis_year is:', prediction[0])")
    
    
############################################################
#                        Question 12                          #
############################################################
def q12_check(answer1: list) -> None:
    solution1 = ['age_first_contact_days', 'age_last_contact_days', 'age_dead_days']
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q12_solution() -> None:
    print("independent_variable = DATA_5[['gender','age_first_contact_days', 'age_last_contact_days', 'age_dead_days', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]")
    print("dependent_variable = DATA_5[['age_diagnosis_year']]")
    print("X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.2)")
    print("rfe_method = RFE(RandomForestClassifier( n_estimators=10,random_state=10),n_features_to_select=3,step=2,)")
    print("rfe_method.fit(X_train, y_train)")
    print("X_train.columns[(rfe_method.get_support())].tolist()")
          
          
############################################################
#                        Question 13                          #
############################################################
def q13_check(answer1: str) -> None:
    solution1 = "{'copy_X': True, 'fit_intercept': True, 'n_jobs': None}"
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q13_solution() -> None:
    print("param_grid = { 'fit_intercept': [True, False], 'copy_X': [True, False], 'n_jobs': [None, -1]}")
    print("model_lin = LinearRegression()")
    print("model_lin.fit(X_train, y_train)")
    print("grid = GridSearchCV(model_lin, param_grid=param_grid, scoring='r2', cv=5)")
    print("grid.fit(X_train, y_train)")
    print("print('Best parameters:', grid.best_params_)")
    print("print('Best score:', grid.best_score_)")

    print("score = model_lin.score(X_test, y_test)")
    print("print('R^2 score:', score)")
    
    
############################################################
#                        Question 14                          #
############################################################
def q14_check(answer1:str) -> None:
    solution1 = "1"
    
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q14_solution() -> None:
    print("1")
    
    
############################################################
#                        Question 15                          #
############################################################
def q15_check(answer1: str) -> None:
    solution1 = "0.8"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q15_solution() -> None:
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
def q16_check(answer1: str) -> None:
    solution1 = "D"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q16_solution() -> None:
    print("diagnosis_text = input('Enter the diagnosis text (in Norwegian) i.e. Uspesifisert aplastisk anemi: ')")
    print("X_user = vectorizer.transform([diagnosis_text])")
    print("diagnosis_code = classifier.predict(X_user)")
    print("print('The predicted diagnosis code for is:', diagnosis_code[0])")
    
    
############################################################
#                        Question 17                          #
############################################################
def q17_check(answer1: str) -> None:
    solution1 = "I"   
    try:
        assert solution1 == answer1
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q17_solution() -> None:
    print("diagnosis_text = input('Enter the diagnosis text (in Norwegian) i.e, ikke-reumatisk aortainsuffisi: ')")
    print("X_user = vectorizer.transform([diagnosis_text])")
    print("diagnosis_code = classifier.predict(X_user)")
    print("print('The predicted diagnosis code', diagnosis_code[0])")
    
    
############################################################
#                        Question 18                          #
############################################################
def q18_check(answer1: pd.DataFrame) -> None:
    DATA_88 = DATA_66.copy(deep=True)
    features = ['gender', 'age_first_contact_days', 'age_last_contact_days', 'age_diagnosis_days', 'age_dead_days', 'age_diagnosis_year', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']
    scaler = StandardScaler()
    DATA_88[features] = scaler.fit_transform(DATA_88[features])
    pca = PCA(n_components=2)
    pca.fit(DATA_88[features])
    transformed_data = pca.transform(DATA_88[features])
    new_data = pd.DataFrame({'PC1': transformed_data[:, 0], 'PC2': transformed_data[:, 1]})

    try:
        assert_frame_equal(new_data,answer1)
        prGreen("Correct")
            
    except Exception:
        prRed("Incorrect")
        

def q18_solution() -> None:
    print("DATA_88 = DATA_66.copy(deep=True)")
    print("features = ['gender', 'age_first_contact_days', 'age_last_contact_days', 'age_diagnosis_days', 'age_dead_days', 'age_diagnosis_year', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']")
    print("scaler = StandardScaler()")
    print("DATA_88[features] = scaler.fit_transform(DATA_88[features])")
    print("pca = PCA(n_components=2)")
    print("pca.fit(DATA_88[features])")
    print("transformed_data = pca.transform(DATA_88[features])")
    print("new_data = pd.DataFrame({'PC1': transformed_data[:, 0], 'PC2': transformed_data[:, 1]})")
    
    