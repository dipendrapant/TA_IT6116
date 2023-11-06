from pandas.testing import assert_frame_equal, assert_series_equal
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from IPython.display import display


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

############################################################
#                  Printing Functions                      #
############################################################


def prRed(string: str) -> None:
    print(f"\033[91m {string}\033[00m")


def prGreen(string: str) -> None:
    print(f"\033[92m {string}\033[00m")

    
    
def datasetOverview()-> pd.DataFrame:
    display(DATA_4.head(10))
    print("\n\n")
    print("================================================================================================================================")
    df_columnname_norsk = {"In Norsk":["Løpenummer","Kjønn","Diagnosekode","Diagnosetekst","Alder_førstekontakt_dager","Alder_sistekontakt_dager","Alder_diagnose_dager","Alder_død_dager",
                       "Alder_diagnose_aar","Alder_aar_førstekontakt","Alder_aar_sistekontakt","Alder_aar_død"],
                      "In English":["ID","Gender","ICD_Code-Diagnostic_code","ICD_Description_Diagnostic_text",
                                    "Age_In_Days_On_FirstContact","Age_In_Days_On_LastContact","Age_In_Days_Taken_To_Know_Diagnosis_Reasult","Age_In_Days_During_Death",
                       "Age_In_Year_Taken_To_Know_Diagnosis_Reasult","Age_In_Years_On_FirstContact","Age_In_Years_On_LastContact","Age_In_Years_During_Death"]}

    column_name_norsk_english = pd.DataFrame(df_columnname_norsk)
    print(column_name_norsk_english.to_string(index=False))
    print("================================================================================================================================")
    print(str("\n\n Rows:{} and Columns:{}").format(len(DATA_4),len(DATA_4.columns)))
    print("\n\n Null Count:{} ".format(DATA_4.isnull().sum().sum()))
    print("================================================================================================================================")
    print(DATA_4.info())

    
    

    

############################################################
#                        Warmup 1                          #
############################################################
def w1_check(answer: pd.DataFrame) -> None:
    solution = pd.DataFrame([["C000", "External upper lip"], ["D104", "Tonsil"]], columns=["ICD Code", "Name"])
    try:
        assert_frame_equal(solution, answer)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")


def w1_solution() -> None:
    print('icd_codes = pd.DataFrame([["C000", "External upper lip"], ["D104", "Tonsil"]], columns=["ICD Code", "Name"])')

############################################################
#                        Warmup 2                          #
############################################################


def w2_check() -> None:
    icd_codes = pd.DataFrame([["C000", 4], ["D104", 6]], columns=["ICD Code", "Count"])
    icd_codes.plot.bar(x="ICD Code", y="Count")
    plt.show()


def w2_solution() -> None:
    print('icd_codes = pd.DataFrame([["C000", 4], ["D104", 6]], columns=["ICD Code", "Count"])')
    print('icd_codes.plot.bar(x="ICD Code", y="Count")')
    print('plt.show()')

############################################################
#                        Warmup 3                          #
############################################################


def w3_check(answer: str) -> None:
    solution = "A151"
    try:
        assert answer == solution
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def w3_solution() -> None:
    print('A151')

############################################################
#                      Question 1                            #
############################################################
def t1_check(answer: str) -> None:
    solution = "D50"
    try:
        assert answer == solution
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")


def t1_solution() -> None:
    print('D50')
    
############################################################
#                      Question 2                             #
############################################################
def t2_check(answer: str) -> None:
    solution = "L02"
    try:
        assert answer == solution
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")


def t2_solution() -> None:
    print('L02')

############################################################
#                      Question 3                          #
############################################################
def t3_check(answer: str) -> None:
    solution = "F00"
    try:
        assert answer == solution
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t3_solution() -> None:
    print('F00')

############################################################
#                      Question 4                          #
############################################################
def t4_check(answer: pd.DataFrame) -> None:
    solution = DATA_4[DATA_4["diagnosis_code"].str.startswith("F00")]
    try:
        assert_frame_equal(answer, solution)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t4_solution() -> None:
    print('icd_code = "F00"')
    print('diagnosis_codes_XXX_df = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_code)]')
    
############################################################
#                      Question 5                          #
############################################################
def t5_check(answer: pd.DataFrame) -> None:
    DATA_4_Gender_DiagnosisCode = DATA_4[['gender','diagnosis_code']].copy()
    le = LabelEncoder()
    for col in DATA_4_Gender_DiagnosisCode.columns:
        DATA_4_Gender_DiagnosisCode[col] = le.fit_transform(DATA_4_Gender_DiagnosisCode[col])
    solution = DATA_4_Gender_DiagnosisCode.head(20)
    try:
        assert_frame_equal(answer, solution)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t5_solution() -> None:
    print("DATA_4_Gender_DiagnosisCode = DATA_4[['gender','diagnosis_code']].copy()")
    print("display(DATA_4_Gender_DiagnosisCode.head(20))")
    print("le = LabelEncoder()")
    print("for col in DATA_4_Gender_DiagnosisCode.columns:")
    print("\tDATA_4_Gender_DiagnosisCode[col] = le.fit_transform(DATA_4_Gender_DiagnosisCode[col])")
    print("display(DATA_4_Gender_DiagnosisCode.head(20))")
    
############################################################
#                      Question 6                          #
############################################################
def t6_check(answer: pd.DataFrame) -> None:
    DATA_4_Gender_DiagnosisCode = DATA_4[['gender','diagnosis_code']].copy()
    onehotencoded_DATA_4 = pd.get_dummies(DATA_4_Gender_DiagnosisCode,columns = ['gender','diagnosis_code'],dtype=int)
    solution = onehotencoded_DATA_4.head(20)
    try:
        assert_frame_equal(answer,solution)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t6_solution() -> None:
    print("DATA_4_Gender_DiagnosisCode = DATA_4[['gender','diagnosis_code']].copy()")
    print("display(DATA_4_Gender_DiagnosisCode.head(20))")
    print("onehotencoded_DATA_4 = pd.get_dummies(DATA_4_Gender_DiagnosisCode,columns = ['gender','diagnosis_code'],dtype=int)")
    print("display(onehotencoded_DATA_4.head(20))")
    
############################################################
#                      Question 7                          #
############################################################
def t7_check(answer: str) -> None:
    solution = 13.490737563232042
    try:
        assert answer == solution
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")


def t7_solution() -> None:
    print('selected_df = DATA_4[DATA_4["diagnosis_code"].isin(["F102", "F011"])]')
    print('x = selected_df["age_dead_year"].values')
    print('y = selected_df["age_diagnosis_year"].values')
    print('distance = np.sqrt(np.sum((x - y) ** 2))')
    print('print(distance)')
    
############################################################
#                      Question 8                          #
############################################################
def t8_check(answer: pd.DataFrame) -> None:
    TASKXX_DATA_4 = DATA_4[['diagnosis_text']].iloc[:5].copy()
    count_vectorizor = CountVectorizer()
    numeric_diagnosis_text = count_vectorizor.fit_transform(TASKXX_DATA_4['diagnosis_text'])
    numeric_diagnosis_text_df = pd.DataFrame(numeric_diagnosis_text.toarray(), columns=count_vectorizor.get_feature_names_out())
    try:
        assert_frame_equal(answer, numeric_diagnosis_text_df)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t8_solution() -> None:
    print("TASKXX_DATA_4 = DATA_4[['diagnosis_text']].iloc[:5].copy()")
    print("display(TASKXX_DATA_4)")
    print("count_vectorizor = CountVectorizer()")
    print("numeric_diagnosis_text = count_vectorizor.fit_transform(TASKXX_DATA_4['diagnosis_text'])")
    print("numeric_diagnosis_text_df = pd.DataFrame(numeric_diagnosis_text.toarray(), columns=count_vectorizor.get_feature_names_out())")
    print("display(numeric_diagnosis_text_df)")


############################################################
#                      Question 9                          #
############################################################
def t9_check(answer:tuple):
    min_val = DATA_4['age_dead_year'].min()
    max_val = DATA_4['age_dead_year'].max()
    mean_val = DATA_4['age_dead_year'].mean()
    median_val = DATA_4['age_dead_year'].median()
    solution = min_val,max_val,mean_val,median_val
    try:
        assert answer == solution
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t9_solution() -> None:    
    print("min_val = DATA_4['age_dead_year'].min()")
    print("max_val = DATA_4['age_dead_year'].max()")
    print("mean_val = DATA_4['age_dead_year'].mean()")
    print("median_val = DATA_4['age_dead_year'].median()")
    print("print(f'Minimum:{min_val}, Maximum:{max_val}, Mean:{mean_val}, Median:{median_val}')")
    
############################################################
#                      Question 10                         #
############################################################
def t10_check(answer: pd.DataFrame) -> None:
    solution = DATA_4[DATA_4["diagnosis_code"].str.startswith("J449")]
    try:
        assert_frame_equal(answer, solution)
        prGreen("Correct")
    except Exception:
        prRed("Incorrect")

def t10_solution() -> None:
    print('diagnosis_codes_starting_with_J449_df = DATA_4[DATA_4["diagnosis_code"].str.startswith("J449")]')
    
############################################################
#                      Question 11                         #
############################################################
def t11_check() -> None:
    DATA_4_ICD = DATA_4[DATA_4['diagnosis_code'].str.startswith('C11')]
    DATA_4_ICD_C11 = DATA_4_ICD[['id', 'age_diagnosis_year', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]
    DATA_4_ICD_C11.set_index('id', inplace=True)
    ax = DATA_4_ICD_C11.plot(kind='line', marker='o')
    ax.grid(True)
    plt.xlabel('Patient id')
    plt.ylabel('Year')
    plt.title('Line plot of patient with diagnose, first, last, dead year')
    plt.show()
    
def t11_solution() -> None:
    print("DATA_4_ICD = DATA_4[DATA_4['diagnosis_code'].str.startswith('C11')]")
    print("DATA_4_ICD_C11 = DATA_4_ICD[['id', 'age_diagnosis_year', 'age_first_contact_year', 'age_last_contact_year', 'age_dead_year']]")
    print("DATA_4_ICD_C11.set_index('id', inplace=True)")
    print("ax = DATA_4_ICD_C11.plot(kind='line', marker='o')")
    print("ax.grid(True)")
    print("plt.xlabel('Patient id')")
    print("plt.ylabel('Year')")
    print("plt.title('Line plot of patient with diagnose, first, last, dead year')")
    print("plt.show()")

############################################################
#                      Question 12                         #
############################################################
def t12_check() -> None:
    DATA_4_ICD = DATA_4[DATA_4["diagnosis_code"].str.startswith("F001")]
    DATA_4_ICD.plot.scatter(x="age_last_contact_year", y="age_dead_year")
    plt.scatter(DATA_4_ICD["age_last_contact_year"], DATA_4_ICD["age_dead_year"])
    plt.xlabel("age_last_contact_year")
    plt.ylabel("age_dead_year")
    plt.show()
    
def t12_solution() -> None:
    print('DATA_4_ICD = DATA_4[DATA_4["diagnosis_code"].str.startswith("F001")]')
    print('DATA_4_ICD.plot.scatter(x="age_last_contact_year", y="age_dead_year")')
    print('plt.scatter(DATA_4_ICD["age_last_contact_year"], DATA_4_ICD["age_dead_year"])')
    print('plt.xlabel("age_last_contact_year")')
    print('plt.ylabel("age_dead_year")')
    print('plt.show()')
    
############################################################
#                      Question 13                         #
############################################################
def t13_check() -> None:
    DATA_4_ICD_I2 = DATA_4[DATA_4['diagnosis_code'].str.startswith('I2')]
    DATA_4_ICD_C = DATA_4[DATA_4['diagnosis_code'].str.startswith('C')]
    DATA_4_ICD_J4 = DATA_4[DATA_4['diagnosis_code'].str.startswith('J4')]
    DATA_4_ICD_E66 = DATA_4[DATA_4['diagnosis_code'].str.startswith('E66')]
    DATA_4_ICD_F00 = DATA_4[DATA_4['diagnosis_code'].str.startswith('F00')]
    DATA_4_ICD_E1 = DATA_4[DATA_4['diagnosis_code'].str.startswith('E1')]
    number_of_patient = [len(DATA_4_ICD_I2), len(DATA_4_ICD_C), len(DATA_4_ICD_J4), len(DATA_4_ICD_E66), len(DATA_4_ICD_F00), len(DATA_4_ICD_E1)]
    diagnose_code_labels = ['I2', 'C', 'J4', 'E66', 'F00', 'E1']
    plt.bar(diagnose_code_labels,number_of_patient, color='blue', edgecolor='black')
    plt.title('Histogram showing the number of patients with some common diseases ')
    plt.xlabel('Diagnosis Code')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.show()
    
def t13_solution() -> None:
    print("DATA_4_ICD_I2 = DATA_4[DATA_4['diagnosis_code'].str.startswith('I2')]")
    print("DATA_4_ICD_C = DATA_4[DATA_4['diagnosis_code'].str.startswith('C')]")
    print("DATA_4_ICD_J4 = DATA_4[DATA_4['diagnosis_code'].str.startswith('J4')]")
    print("DATA_4_ICD_E66 = DATA_4[DATA_4['diagnosis_code'].str.startswith('E66')]")
    print("DATA_4_ICD_F00 = DATA_4[DATA_4['diagnosis_code'].str.startswith('F00')]")
    print("DATA_4_ICD_E1 = DATA_4[DATA_4['diagnosis_code'].str.startswith('E1')]")
    print("number_of_patient = [len(DATA_4_ICD_I2), len(DATA_4_ICD_C), len(DATA_4_ICD_J4), len(DATA_4_ICD_E66), len(DATA_4_ICD_F00), len(DATA_4_ICD_E1)]")
    print("diagnose_code_labels = ['I2', 'C', 'J4', 'E66', 'F00', 'E1']")
    print("plt.bar(diagnose_code_labels,number_of_patient, color='blue', edgecolor='black')")
    print("plt.title('Histogram showing the number of patients with some common diseases ')")
    print("plt.xlabel('Diagnosis Code')")
    print("plt.ylabel('Count')")
    print("plt.grid(axis='y')")
    print("plt.show()")


############################################################
#                      Question 14                          #
############################################################
def t14_check() -> None:
    icd_codes = ("C41", "I42", "J42", "Z42")
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]
    plt.scatter(df_selected["age_diagnosis_year"], df_selected["diagnosis_code"])
    plt.xlabel("Diagnosis Age in Year")
    plt.ylabel("Diagnosis Code")
    plt.title("age_diagnosis_year vs diagnosis_code")
    plt.show()
    
def t14_solution() -> None:
    print('icd_codes = ("C41", "I42", "J42", "Z42")')
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]')
    print('plt.scatter(df_selected["age_diagnosis_year"], df_selected["diagnosis_code"])')
    print('plt.xlabel("Diagnosis Age in Year")')
    print('plt.ylabel("Diagnosis Code")')
    print('plt.title("age_diagnosis_year vs diagnosis_code")')
    print('plt.show()')
    
############################################################
#                      Question 15                         #
############################################################
def t15_check() -> None:
    icd_codes =("A415", "C348", "J440")
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]
    plt.scatter(df_selected["age_diagnosis_year"], df_selected["diagnosis_code"])
    plt.xlabel("age_diagnosis_year")
    plt.ylabel("diagnosis_code")
    plt.title("age_diagnosis_year vs. diagnosis_code")
    plt.show()
    
def t15_solution() -> None:
    print('icd_codes =("A415", "C348", "J440")')
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]')
    print('plt.scatter(df_selected["age_diagnosis_year"], df_selected["diagnosis_code"])')
    print('plt.xlabel("age_diagnosis_year")')
    print('plt.ylabel("diagnosis_code")')
    print('plt.title("age_diagnosis_year vs. diagnosis_code")')
    print('plt.show()')
    
############################################################
#                      Question 16                         #
############################################################
def t16_check() -> None:
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(("A415", "J440"))]
    grouped_count_df = df_selected.groupby(["age_diagnosis_year", "diagnosis_code"]).size().unstack()
    grouped_count_df.plot(kind="bar")
    plt.xlabel("Diagnosis Year")
    plt.ylabel("Count")
    plt.legend(title="ICD Code")
    plt.show()
    
def t16_solution() -> None:
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(("A415", "J440"))]')
    print('grouped_count_df = df_selected.groupby(["age_diagnosis_year", "diagnosis_code"]).size().unstack()')
    print('grouped_count_df.plot(kind="bar")')
    print('plt.xlabel("Diagnosis Year")')
    print('plt.ylabel("Count")')
    print('plt.legend(title="ICD Code")')   
    print('plt.show()')  
    
    
############################################################
#                      Question 17                         #
############################################################
def t17_check() -> None:
    icd_codes = ("A415", "J440", "C348")
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]
    age_counts = df_selected["diagnosis_code"].value_counts()
    plt.pie(age_counts, labels=age_counts.index, autopct="%1.1f%%")
    plt.title("Pie chart of selected ICD codes with their count values")
    plt.show()
    
def t17_solution() -> None:
    print('icd_codes = ("A415", "J440", "C348")')
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]')
    print('age_counts = df_selected["diagnosis_code"].value_counts()')
    print('plt.pie(age_counts, labels=age_counts.index, autopct="%1.1f%%")')
    print('plt.title("Pie chart of selected ICD codes with their count values")')
    print('plt.show()')  
    
    
############################################################
#                      Question 18                         #
############################################################
def t18_check() -> None:
    icd_codes = ("A415", "J440", "C348")
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]
    icd_counts = df_selected["diagnosis_code"].value_counts()
    plt.bar(icd_counts.index, icd_counts)
    plt.xlabel("Diagnosis Code")
    plt.ylabel("Count")
    plt.title("Bar plot of selected ICD codes")
    plt.grid(True)
    plt.show()
    
def t18_solution() -> None:
    print('icd_codes = ("A415", "J440", "C348")')
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]')
    print('icd_counts = df_selected["diagnosis_code"].value_counts()')
    print('plt.bar(icd_counts.index, icd_counts)')
    print('plt.xlabel("Diagnosis Code")')
    print('plt.ylabel("Count")')
    print('plt.title("Bar plot of selected ICD codes(A415, J440, C348)")')
    print('plt.grid(True)')
    print('plt.show()')  
    
    
    
############################################################
#                      Question 19                         #
############################################################
def t19_check() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    icd_codes = ("A415", "C348")
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]
    ax1.scatter(df_selected["age_first_contact_year"], df_selected["age_diagnosis_year"])
    ax1.set_xlabel("Age at first contact") 
    ax1.set_ylabel("Age diagnosis year")
    ax1.set_title("Relationship between age at first and last contact")
    ax2.scatter(df_selected["age_last_contact_year"], df_selected["age_dead_year"])
    ax2.set_xlabel("age_last_contact_year") 
    ax2.set_ylabel("age_dead_year")
    ax2.set_title("Relationship between age_last_contact_year and age_dead_year")
    plt.show()
    
def t19_solution() -> None:
    print('fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))')
    print('icd_codes = ("A415", "C348")')
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(icd_codes)]')
    print('ax1.scatter(df_selected["age_first_contact_year"], df_selected["age_diagnosis_year"])')
    print('ax1.set_xlabel("Age at first contact")') 
    print('ax1.set_ylabel("Age diagnosis year")')
    print('ax1.set_title("Relationship between age at first and last contact")')
    print('ax2.scatter(df_selected["age_last_contact_year"], df_selected["age_dead_year"])')
    print('ax2.set_xlabel("age_last_contact_year")') 
    print('ax2.set_ylabel("age_dead_year")')
    print('ax2.set_title("Relationship between age_last_contact_year and age_dead_year")')
    print('plt.show()')

    
############################################################
#                      Question 20                          #
############################################################
def t20_check() -> None:
    CLUSTER_DATA_4 = DATA_4.replace(to_replace=["female","male"],value=[0,1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(CLUSTER_DATA_4["gender"], CLUSTER_DATA_4["age_diagnosis_year"], CLUSTER_DATA_4["age_dead_year"])
    ax.set_xlabel("Gender")
    ax.set_ylabel("Age at diagnosis")
    ax.set_zlabel("Age at death")
    ax.set_title("3D plot of gender, age at diagnosis, and age at death")
    plt.show()
    
    
def t20_solution() -> None:
    print('CLUSTER_DATA_4 = DATA_4.replace(to_replace=["female","male"],value=[0,1])')
    print('fig = plt.figure()')
    print('ax = fig.add_subplot(111, projection="3d")')
    print('ax.scatter(CLUSTER_DATA_4["gender"], CLUSTER_DATA_4["age_diagnosis_year"], CLUSTER_DATA_4["age_dead_year"])')
    print('ax.set_xlabel("Gender")')
    print('ax.set_ylabel("Age at diagnosis")')
    print('ax.set_zlabel("Age at death")')
    print('ax.set_title("3D plot of gender, age at diagnosis, and age at death")')
    print('plt.show()')
    

############################################################
#                      Question 21                         #
############################################################
def t21_check() -> None:
    df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(("H"))]
    kmeans = KMeans(n_clusters =3)
    y_predicted = kmeans.fit_predict(df_selected[["age_diagnosis_year","age_dead_year"]])
    df_selected["cluster"] = y_predicted
    cluster0 = df_selected[df_selected["cluster"]==0]
    cluster1 = df_selected[df_selected["cluster"]==1]
    cluster2 = df_selected[df_selected["cluster"]==2]
    plt.scatter(cluster0["age_diagnosis_year"],cluster0["age_dead_year"],color="green", label = "Cluster 1")
    plt.scatter(cluster1["age_diagnosis_year"],cluster1["age_dead_year"],color="red", label = "Cluster 2")
    plt.scatter(cluster2["age_diagnosis_year"],cluster2["age_dead_year"],color="blue", label = "Cluster 3")
    plt.xlabel("age_diagnosis_year")
    plt.ylabel("age_dead_year")
    plt.legend()
    plt.show()
    
    
    
def t21_solution() -> None:
    print('df_selected = DATA_4[DATA_4["diagnosis_code"].str.startswith(("H"))]')
    print('kmeans = KMeans(n_clusters =3)')
    print('y_predicted = kmeans.fit_predict(df_selected[["age_diagnosis_year","age_dead_year"]])')
    print('df_selected["cluster"] = y_predicted')
    print('cluster0 = df_selected[df_selected["cluster"]==0]')
    print('cluster1 = df_selected[df_selected["cluster"]==1]')
    print('cluster2 = df_selected[df_selected["cluster"]==2]')
    print('plt.scatter(cluster0["age_diagnosis_year"],cluster0["age_dead_year"],color="green", label = "Cluster 1")')
    print('plt.scatter(cluster1["age_diagnosis_year"],cluster1["age_dead_year"],color="red", label = "Cluster 2")')
    print('plt.scatter(cluster2["age_diagnosis_year"],cluster2["age_dead_year"],color="blue", label = "Cluster 3")')
    print('plt.xlabel("age_diagnosis_year")')
    print('plt.ylabel("age_dead_year")')
    print('plt.legend()')
    print('plt.show()')
    
    


