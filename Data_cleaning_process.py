import pandas as pd
from datetime import datetime

# Function for cleaning other files
def filter_files_by_date(df,filter_col: str,start_date: str,end_date: str):
    new_df = df[(df[filter_col] >= start_date) &
                (df[filter_col] < end_date)]
    return new_df


def main():
    # Building one data file for customers
    cases_df = pd.read_csv('raw_data/customer_cases.csv')
    customerInfo_df = pd.read_csv('raw_data/customer_info.csv')
    customerProd_df = pd.read_csv('raw_data/customer_product.csv')
    prodInfo_df = pd.read_csv('raw_data/product_info.csv')
    # Need to update col date time in cases df
    cases_df['date_time'] = pd.to_datetime(cases_df['date_time'])
    # No data types in customer info need to be adjusted
    # need to update sinup_date_time and canel_date_time
    customerProd_df['signup_date_time'] = pd.to_datetime(customerProd_df['signup_date_time'])
    customerProd_df['cancel_date_time'] = pd.to_datetime(customerProd_df['cancel_date_time'])
    # No data types need to be adjusted in prodinfo_df
    # Need to find the start and end of the data set to filter out
    # non-need information
    # The time frame will be form jan-1-2017 to dec-31-2021
    # this is good becasue there is a lot of econmic use in this
    print(customerProd_df.signup_date_time.min(),",",customerProd_df.signup_date_time.max())
    print(customerProd_df.cancel_date_time.min(),",",customerProd_df.cancel_date_time.max())
    minDate = str(datetime.date(customerProd_df['signup_date_time'].min()))
    maxDate = str(datetime.date(customerProd_df['cancel_date_time'].max()))

    Fed_df = pd.read_csv('raw_data/FEDFUNDS.csv')
    Fed_df['DATE'] = pd.to_datetime(Fed_df['DATE'], format='%Y-%m-%d')
    Fed_df.head()
    FedClean_df = filter_files_by_date(Fed_df,'DATE',minDate,maxDate).reset_index(drop=True)
    CPI_df = pd.read_csv('raw_data/CPI_Inflation_2012_2022.csv')
    CPI_df = CPI_df.drop(['Annual','HALF1','HALF2'],axis=1)
    CPI_df = CPI_df.melt(id_vars=['Year'], var_name='month', value_name='inflation')
    CPI_df['Date'] = pd.to_datetime(CPI_df['Year'].astype(str) +"-" + CPI_df['month'] + '-01',format='%Y-%b-%d')
    CPIClean_df = filter_files_by_date(CPI_df,'Date',minDate,maxDate).reset_index(drop=True)
    ICS_df = pd.read_csv('raw_data/Index_of_Consumer_Sentiment.csv')
    ICS_df['Date'] = pd.to_datetime(ICS_df['YYYY'].astype(str) +"-" + ICS_df['Month'] + '-01',format='%Y-%B-%d')
    ICSClean_df = filter_files_by_date(ICS_df,'Date',minDate,maxDate).reset_index(drop=True)

    ICSClean_df.to_csv('prepared_data/ICS.csv')
    CPI_df.to_csv('prepared_data/CPI.csv')
    FedClean_df.to_csv('prepared_data/FedData.csv')
    cases_df.to_csv('prepared_data/Cases_df.csv')
    customerInfo_df.to_csv('prepared_data/CustomerInfo.csv')
    customerProd_df.to_csv('prepared_data/CustomerProd.csv')
    prodInfo_df.to_csv('prepared_data/Prod_info.csv')


if __name__ == '__main__':
    main()