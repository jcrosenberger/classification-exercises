'''
██╗███╗░░░███╗██████╗░░█████╗░██████╗░████████╗ ███╗░░░███╗░█████╗░██████╗░██╗░░░██╗██╗░░░░░███████╗░██████╗
██║████╗░████║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝ ████╗░████║██╔══██╗██╔══██╗██║░░░██║██║░░░░░██╔════╝██╔════╝
██║██╔████╔██║██████╔╝██║░░██║██████╔╝░░░██║░░░ ██╔████╔██║██║░░██║██║░░██║██║░░░██║██║░░░░░█████╗░░╚█████╗░
██║██║╚██╔╝██║██╔═══╝░██║░░██║██╔══██╗░░░██║░░░ ██║╚██╔╝██║██║░░██║██║░░██║██║░░░██║██║░░░░░██╔══╝░░░╚═══██╗
██║██║░╚═╝░██║██║░░░░░╚█████╔╝██║░░██║░░░██║░░░ ██║░╚═╝░██║╚█████╔╝██████╔╝╚██████╔╝███████╗███████╗██████╔╝
╚═╝╚═╝░░░░░╚═╝╚═╝░░░░░░╚════╝░╚═╝░░╚═╝░░░╚═╝░░░ ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░░╚═════╝░╚══════╝╚══════╝╚═════╝░
'''

datasets_available= ['titanic','iris','telco']
print('The following datasets are available:',*datasets_available, sep='\n')

#generally required for working with datasets
import seaborn as sns
import pandas as pd
import numpy as np
import os

#required for acquire
from env import get_db_url

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


'''
██████╗░░█████╗░████████╗░█████╗░░██████╗███████╗████████╗░██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔════╝
██║░░██║███████║░░░██║░░░███████║╚█████╗░█████╗░░░░░██║░░░╚█████╗░
██║░░██║██╔══██║░░░██║░░░██╔══██║░╚═══██╗██╔══╝░░░░░██║░░░░╚═══██╗
██████╔╝██║░░██║░░░██║░░░██║░░██║██████╔╝███████╗░░░██║░░░██████╔╝
╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝╚═════╝░╚══════╝░░░╚═╝░░░╚═════╝░
'''


'''
████████╗██╗████████╗░█████╗░███╗░░██╗██╗░█████╗░ 
╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗░██║██║██╔══██╗ 
░░░██║░░░██║░░░██║░░░███████║██╔██╗██║██║██║░░╚═╝ 
░░░██║░░░██║░░░██║░░░██╔══██║██║╚████║██║██║░░██╗ 
░░░██║░░░██║░░░██║░░░██║░░██║██║░╚███║██║╚█████╔╝ 
░░░╚═╝░░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░╚════╝░ 
'''


def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('titanic_db'))
    
    return df

def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('py/data/titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('py/data/titanic_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('py/data/titanic_df.csv')
        
    return df


'''
██╗██████╗░██╗░██████╗ 
██║██╔══██╗██║██╔════╝ 
██║██████╔╝██║╚█████╗░ 
██║██╔══██╗██║░╚═══██╗ 
██║██║░░██║██║██████╔╝ 
╚═╝╚═╝░░╚═╝╚═╝╚═════╝░ 
'''




def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('iris_db'))
    
    return df

def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('py/data/iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('py/data/iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('py/data/iris_df.csv')
        
    return df


'''
████████╗███████╗██╗░░░░░░█████╗░░█████╗░ 
╚══██╔══╝██╔════╝██║░░░░░██╔══██╗██╔══██╗ 
░░░██║░░░█████╗░░██║░░░░░██║░░╚═╝██║░░██║ 
░░░██║░░░██╔══╝░░██║░░░░░██║░░██╗██║░░██║ 
░░░██║░░░███████╗███████╗╚█████╔╝╚█████╔╝ 
░░░╚═╝░░░╚══════╝╚══════╝░╚════╝░░╚════╝░ 
'''

def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('py/data/telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('py/data/telco_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('py/data/telco_df.csv')
        
    return df




