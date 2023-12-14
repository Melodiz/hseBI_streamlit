import streamlit as st
import pandas as pd
import numpy as np
import os 
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go


st.title("HSE BI PROJECT")
st.header("Prepared by Novosad Ivan")

@st.cache_data
def load_data_transactions():
    #uploading all data
    PATH_DATA = './dataset'
    tr_1 = pd.read_csv(os.path.join(PATH_DATA, 'transaction_1.csv'), index_col='client_id').sort_index()
    tr_2 = pd.read_csv(os.path.join(PATH_DATA, 'transaction_2.csv'), index_col='client_id').sort_index()
    tr_3 = pd.read_csv(os.path.join(PATH_DATA, 'transaction_3.csv'), index_col='client_id').sort_index()
    tr_4 = pd.read_csv(os.path.join(PATH_DATA, 'transaction_4.csv'), index_col='client_id').sort_index()
    tr_5 = pd.read_csv(os.path.join(PATH_DATA, 'transaction_5.csv'), index_col='client_id').sort_index()
    tr_6 = pd.read_csv(os.path.join(PATH_DATA, 'transaction_6.csv'), index_col='client_id').sort_index()
    transactions = pd.concat(
        [tr_1, tr_2, tr_3, tr_4, tr_5, tr_6]
    )

    mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
    gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
    trans_types = pd.read_csv('dataset/trans_types.csv')
    return  transactions


@st.cache_data
def load_data_mcc():
    PATH_DATA = './dataset'
    mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
    return mcc_codes

@st.cache_data
def load_data_gender():
    PATH_DATA = './dataset'
    gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
    return gender_train


@st.cache_data
def load_data_trans_types():
    trans_types = pd.read_csv('dataset/trans_types.csv')
    return trans_types

transactions = load_data_transactions()
gender_train = load_data_gender()
mcc_codes = load_data_mcc()
trans_types = load_data_trans_types()

# main settings about the style: 
main_template = 'plotly_dark'

st.subheader("Description & links")
st.link_button("Telegram bot", "https://t.me/hse_project_data_analys_bot")
st.link_button("Telegram bot source", "https://github.com/Melodiz/hseBI_bot")
st.link_button("Streamlit app & jupyter notebook & dataframes sourses", 
               "https://github.com/Melodiz/hseBI_streamlit")
st.text(
    "It's a demo project prepared for the Higher School of Economics\n"\
    "by DSBA first-year student."
)

st.text(
    "It presents some business analytics of the data obtained\n"\
    "at the Sber Hackathon.This is the closest possible imitation of\n"\
    "real-life transactions(or that’s real transactions, \n"\
    "I have no confirmed information) as they are proposed to be used to train the models.\n"
    "The total weight of the files in CSV format is 300 megabytes."
)
st.text(
    "For more information about the files, see the overview section.\n"\
    "Special attention should be paid to the parameter trans_time in \n"\
    "the transactions file. These are data in the format day / time, \n"\
    "where day is the day past from a certain point of reference \n"\
    "(the first transaction in the file) and time is local time.")

def change_bool_to_names(gender_value:int) -> str:
    if gender_value:
        return 'male'
    return 'female'

# combine them into one dataframe with 2 columns
# client_id & gender 
@st.cache_data
def build_client_id_gender():
    genders_id_group = pd.DataFrame(transactions.index.copy()).set_index('client_id')
    genders_id_group = genders_id_group.join(gender_train)
    genders_id_group = genders_id_group.loc[:, ~genders_id_group.columns.str.contains('^Unnamed')]
    genders_id_group = genders_id_group[~genders_id_group.index.duplicated(keep='first')]


    genders_id_group = genders_id_group.reset_index().groupby('gender')['client_id'].count().reset_index()
    genders_id_group['gender'][0] = 'female'
    genders_id_group['gender'][1] = 'male'

    #make simple bar chart using this data

    fig_genders_id_group = px.bar(genders_id_group,
        x = 'gender',
        y = 'client_id', 
        template=main_template, 
        labels={'client_id': 'quantity'}, 
        title='how many men and women in dataset', 
        color='gender'
    )
    st.subheader("Genders")
    st.plotly_chart(fig_genders_id_group)
    del fig_genders_id_group
    del genders_id_group
    return

build_client_id_gender()

@st.cache_data
def build_extended_transactions():
    extended_transactions = pd.merge(transactions, gender_train, left_index=True, right_index=True)
    extended_transactions = extended_transactions.sort_index()
    return extended_transactions

extended_transactions = build_extended_transactions()


@st.cache_data
def build_show_row_dataframe():

    st.header("Data overview")
    st.subheader("row form")

    st.text("transactions & genders(first 100 rows, of 3'563'529)")
    st.dataframe(extended_transactions.head(100))

    st.text('mcc codes')
    st.dataframe(mcc_codes)

    st.text('transaction types')
    st.dataframe(trans_types)
    return

build_show_row_dataframe()

def exclude_trans_time_ext(n):
    return (n>0)*(n)

@st.cache_data
def build_time_period_covered():
    #it takes about a minute on m2 macbook
    st.subheader('Time period covered')

    st.text(
        "transaction time data is recorded in the form \n"\
        "of how manydays have passed since the countdown date, \n"\
        "but the time is local time"
    )

    period_covered_sl = {}

    prev_index = transactions.index[0]
    temp_max = -1
    temp_min = 10**5
    for index, row in transactions.sort_index().iterrows():
        data = int(row['trans_time'].split()[0])
        if index == prev_index:
            temp_max = max(temp_max, data)
            temp_min = min(temp_min, data)
        else:
            period_covered_sl[index]=temp_max-temp_min
            prev_index = index
            temp_min = 10**5
            temp_max = -1


    df_period_covered = pd.DataFrame(list(period_covered_sl.items()), columns=['client_id', 'count'])
    df_period_covered['count'] = df_period_covered['count'].apply(exclude_trans_time_ext)



    fig_period_covered = px.histogram(df_period_covered,
                                    x = 'count', 
                                    template=main_template, 
                                    title = 'Spending rate')
    fig_period_covered.update_layout(xaxis_title = 'period of spendings in days', 
                                    yaxis_title = 'users count', 
                                    title_x = 0.45,
                                    bargap = 0.04)

    st.plotly_chart(fig_period_covered)
    del fig_period_covered

    fig_cor_period_covered = px.histogram(df_period_covered.loc[df_period_covered['count']>430],
                                    x = 'count', 
                                    template=main_template, 
                                    title = 'Adjustment Spending rate')
    fig_cor_period_covered.update_layout(xaxis_title = 'period of spendings in days', 
                                    yaxis_title = 'users count', 
                                    title_x = 0.45,
                                    bargap = 0.2)

    st.plotly_chart(fig_cor_period_covered)
    del fig_cor_period_covered
    st.text("conclusion: \n"\
        "our data mainly contain informationon people's \n"\
        "spending over a period of 400-450 days")
    return 

build_time_period_covered()


st.subheader(
    "mean, distribition, std & e.t.c"
)
# lets find how many women and men spend for given period:

#slit men's and women's transactions
@st.cache_data
def build_genders_spendings():
    women_spendings = extended_transactions.loc[(gender_train['gender'] == 0) & (extended_transactions['amount'] < 0)]
    men_spendings = extended_transactions.loc[(gender_train['gender'] == 1) & (extended_transactions['amount'] < 0)]
    return (women_spendings,men_spendings)

women_spendings, men_spendings = build_genders_spendings()

@st.cache_data
def build_gr_gender_spendings():

    gr_women_spendings = women_spendings.groupby(level=0)['amount'].sum().reset_index().set_index('client_id')
    gr_men_spendings = men_spendings.groupby(level=0)['amount'].sum().reset_index().set_index('client_id')
    return (gr_women_spendings, gr_men_spendings)

gr_women_spendings, gr_men_spendings = build_gr_gender_spendings()

# histogram about their spendings
def find_closest_recursive(arr, left, right, target):
	# base case: when there is only one element in the array
	if left == right:
		return arr[left]

	# calculate the middle index
	mid = (left + right) // 2

	# recursively search the left half of the array
	left_closest = find_closest_recursive(arr, left, mid, target)

	# recursively search the right half of the array
	right_closest = find_closest_recursive(arr, mid + 1, right, target)

	# compare the absolute differences of the closest elements in the left and right halves
	if abs(left_closest - target) <= abs(right_closest - target):
		return left_closest
	else:
		return right_closest
     

@st.cache_data
def build_avg_spendings(gr_men_spendings, gr_women_spendings):
    avg_men_spendings = round(abs(gr_men_spendings['amount'].mean()), 2)
    avg_women_spendings = round(abs(gr_women_spendings['amount'].mean()), 2)

    bar_c_of_avg_spendings = px.bar(
            x = [avg_men_spendings, avg_women_spendings],
            y = ['men', 'women'],
            orientation='h',
            title = "AVG men's and women's spendings: (sum over a period)", 
            labels={ "x": "quantity in millions of rubles",
                    "y": "sex"},
                    template=main_template,)
    
    del avg_women_spendings
    del avg_men_spendings

    st.plotly_chart(bar_c_of_avg_spendings)
    del bar_c_of_avg_spendings

    del gr_women_spendings

    return

build_avg_spendings(
    gr_men_spendings = gr_men_spendings,
    gr_women_spendings = gr_women_spendings
)




@st.cache_data
def build_bar_income_distr(gr_men_spendings):
    fig_box_gr_men_spendigns = px.box(gr_men_spendings['amount'].abs(), 
                template=main_template, 
                title = "Box plot, men's spendings", 
                y = 'amount', 
                points='all')

    del gr_men_spendings

    st.plotly_chart(fig_box_gr_men_spendigns)
    del fig_box_gr_men_spendigns

    return 

build_bar_income_distr(gr_men_spendings=gr_men_spendings)


@st.cache_data
def build_bar_hist_income_distr(n,gr_men_spendings, gr_women_spendings ):
    array_of_closest = [x*(1_500_000/n) for x in range(n)]
    gr_men_spendings_cor = gr_men_spendings.reset_index()['amount'].abs().tolist()
    gr_men_spendings_cor = [find_closest_recursive(array_of_closest, 0, n-1, x) for x in gr_men_spendings_cor]

    gr_men_spendings_cor = list(Counter(gr_men_spendings_cor).items())
    gr_men_spendings_cor = pd.DataFrame(gr_men_spendings_cor).rename(columns={0:'amount', 1:'count'})

    gr_men_spendings_cor = gr_men_spendings_cor.sort_values('amount')
    gr_men_spendings_cor = gr_men_spendings_cor.head(len(gr_men_spendings_cor)-1)
    gr_men_spendings_cor['gender'] = ['male']*len(gr_men_spendings_cor)

    # same for women
    gr_women_spendings_cor = gr_women_spendings.reset_index()['amount'].abs().tolist()
    gr_women_spendings_cor = [find_closest_recursive(array_of_closest, 0, n-1, x) for x in gr_women_spendings_cor]

    gr_women_spendings_cor = list(Counter(gr_women_spendings_cor).items())
    gr_women_spendings_cor = pd.DataFrame(gr_women_spendings_cor).rename(columns={0:'amount', 1:'count'})

    gr_women_spendings_cor = gr_women_spendings_cor.sort_values('amount')
    gr_women_spendings_cor = gr_women_spendings_cor.head(len(gr_women_spendings_cor)-1)
    gr_women_spendings_cor['gender'] = ['female']*len(gr_women_spendings_cor)
    return gr_men_spendings_cor,gr_women_spendings_cor

st.title("Details")
st.header("Distribution charts")
st.subheader("income distribution")
st.text("the sum of all expenses for the whole period, for each user (row)")

n = st.slider('choose the step', 2, 400, 60)

gr_men_spendings_cor,gr_women_spendings_cor = build_bar_hist_income_distr(
    n,
    gr_men_spendings=gr_men_spendings, 
    gr_women_spendings=gr_women_spendings
)



gr_women_spendings_cor['gender'] = ['female']*len(gr_women_spendings_cor)

fig_men_spendings_bar = px.bar(
		gr_men_spendings_cor,
		x = 'amount', 
		y = 'count', 
		template=main_template, 
		color='amount', 
		title = "distribution chart of men's spendings",
		labels={'count':'amount of people'}
						)
fig_women_spendings_bar = px.bar(
		gr_women_spendings_cor,
		x = 'amount', 
		y = 'count', 
		template=main_template, 
		color='amount',
		title = "distribution chart of women's spendings",
		labels={'count':"amount of people"}
						)

st.plotly_chart(fig_men_spendings_bar)
del fig_men_spendings_bar
st.plotly_chart(fig_women_spendings_bar)
del fig_women_spendings_bar

merge_spendings_cor = pd.concat([gr_men_spendings_cor, gr_women_spendings_cor])

del gr_women_spendings_cor
del gr_men_spendings_cor

fig_spendings_merge = px.line(merge_spendings_cor, 
							  color='gender', 
							  x = 'amount', 
							  y = 'count', 
							  template=main_template, 
							  title="distribution chart of men's and women's spendings",
							  labels={
									'1':'men',
				 					'amount':'spends in rubles',
									'count':'amount of people'
										})
del merge_spendings_cor
st.plotly_chart(fig_spendings_merge)
del fig_spendings_merge

st.subheader("time distribution")
st.text(
	"what time of day people are more likely to make transactions"
)

# lets visualise all men's and women's spendings time information:

partical_men_sp_data = men_spendings[['trans_time', 'amount']].copy()
partical_women_sp_data = women_spendings[['trans_time', 'amount']].copy()

def sep_time_and_data(trans_time):
    return trans_time[-8:-6]

# def mcc_value_s(mcc_code):
#     return mcc_means[mcc_code]['mcc_description']

def cor_amount(amount):
    return (abs(amount)<130_000) * abs(amount)


partical_men_sp_data['trans_time'] = (partical_men_sp_data['trans_time'].apply(sep_time_and_data))
# partical_men_sp_data['mcc_code'] = partical_men_sp_data['mcc_code'].apply(mcc_value_s)
partical_men_sp_data['amount'] = partical_men_sp_data['amount'].apply(cor_amount)

partical_men_sp_data = partical_men_sp_data.sort_values('trans_time')


partical_women_sp_data['trans_time'] = partical_women_sp_data['trans_time'].apply(sep_time_and_data)
# partical_women_sp_data['mcc_code'] = partical_women_sp_data['mcc_code'].apply(mcc_value_s)
partical_women_sp_data['amount'] = partical_women_sp_data['amount'].apply(cor_amount)
partical_women_sp_data = partical_women_sp_data.sort_values('trans_time')

time_spendings_men = px.histogram(partical_men_sp_data, x = 'trans_time', y = 'amount',
                                    template=main_template,
                                    labels={'trans_time':'transactions time'},
                                    title="sum of men's transactions on each hour",
                                    color_discrete_sequence=px.colors.qualitative.Set1)
time_spendings_men.update_layout(yaxis_title = 'sum of transaction on eeach hour')

st.plotly_chart(time_spendings_men)
del time_spendings_men

time_spendings_women = px.histogram(partical_women_sp_data, x = 'trans_time', y = 'amount',
                                    template=main_template,
                                    labels={'trans_time':'transactions time'},
                                    title="sum of women's transactions on each hour", 
                                    color_discrete_sequence=px.colors.qualitative.Set3)
time_spendings_women.update_layout(yaxis_title = 'sum of transaction on eeach hour')

st.plotly_chart(time_spendings_women)
del time_spendings_women

# show how much operation they perform on each hour
plank = st.slider('Choose plank', 0, 100_000, 10_000)

def amount_transofrm(value):
    return int(value>=plank)


count_trans_men = partical_men_sp_data.copy()
del partical_men_sp_data
count_trans_men['amount'] = count_trans_men['amount'].apply(amount_transofrm)

count_trans_women = partical_women_sp_data.copy()
del partical_women_sp_data
count_trans_women['amount'] = count_trans_women['amount'].apply(amount_transofrm)

fig_count_trans_per_hounr_men = px.histogram(
            count_trans_men, 
            x = 'trans_time', 
            y = 'amount', 
            title=f'how many operations greater then {plank} rub men perform in an hour', 
            template=main_template,
            color_discrete_sequence=px.colors.qualitative.Set1, 
            labels={'trans_time':'transactions time'})

del count_trans_men

fig_count_trans_per_hounr_men.update_layout(yaxis_title="how many operations men perform in an hour")

st.plotly_chart(fig_count_trans_per_hounr_men)
del fig_count_trans_per_hounr_men

fig_count_trans_per_hounr_women = px.histogram(
            count_trans_women, 
            x = 'trans_time', 
            y = 'amount', 
            title=f'how many operations greater then {plank} rub women perform in an hour ', 
            template=main_template,
            color_discrete_sequence=px.colors.qualitative.Set3, 
            labels={'trans_time':'transactions time'})
del count_trans_women

fig_count_trans_per_hounr_women.update_layout(yaxis_title="how many operations men perform in an hour")

st.plotly_chart(fig_count_trans_per_hounr_women)
del fig_count_trans_per_hounr_women


st.subheader("Categorization")
st.text("get statistics on the 10 most popular expenses of men and women")

categories_number = st.slider('select the number of categories', 0, 50, 10)

#collecting men's and women's mean spendings by category 
mean_mcc_men = men_spendings.groupby('mcc_code')['amount'].sum().reset_index().set_index('mcc_code')
mean_mcc_women = women_spendings.groupby('mcc_code')['amount'].sum().reset_index().set_index('mcc_code')

del women_spendings
del men_spendings

mcc_means = mcc_codes.to_dict('index')
# get statistics on the 15 most popular expenses of men and women
top10_men_sum = abs(mean_mcc_men.sort_values('amount').head(categories_number)).reset_index()
top10_women_sum = abs(mean_mcc_women.sort_values('amount').head(categories_number)).reset_index()

del mean_mcc_men
del mean_mcc_women
# visualise it:

top10_men_mcc_names = [mcc_means[x]['mcc_description'] for x in top10_men_sum['mcc_code'].tolist()]

fig_top10_men = px.bar(top10_men_sum, y = 'amount', x = [x for x in range(categories_number)], 
            title="Top 10 men's spending categories", 
            labels={'y':'spending categories'}, 
            template=main_template, 
            color=top10_men_mcc_names)

del top10_men_mcc_names

fig_top10_men.update_layout(legend_orientation="h",    
    legend_y=-0.25,
    legend_x = -0.1)

st.plotly_chart(fig_top10_men)
del fig_top10_men

top10_women_mcc_names = [mcc_means[x]['mcc_description'] for x in top10_women_sum['mcc_code'].tolist()]

fig_top10_women = px.bar(top10_women_sum, y = 'amount', x = [x for x in range(categories_number)], 
            title="Top 10 women's spending categories", 
            labels={'y':'spending categories'}, 
            template=main_template, 
            color=top10_women_mcc_names)

del top10_women_mcc_names

fig_top10_women.update_layout(legend_orientation="h",    
    legend_y=-0.25,
    legend_x = -0.1)

st.plotly_chart(fig_top10_women)
del fig_top10_women

st.subheader("City distribution")

st.text(
      "box plot by transaction. Distribution by city. \n"\
    "Extremes outside the picture"
)

city_distr_men = extended_transactions.loc[(extended_transactions['gender']==1) & (extended_transactions['amount']<0)].reset_index(
).groupby(['client_id', 'trans_city'])['amount'].sum().reset_index()

city_distr_men['amount'] = city_distr_men['amount'].abs()

city_distr_women = extended_transactions.loc[(extended_transactions['gender']==0) & (extended_transactions['amount']<0)].reset_index(
).groupby(['client_id', 'trans_city'])['amount'].sum().reset_index()

city_distr_women['amount'] = city_distr_men['amount'].abs()

fig_distr_city_men = px.box(city_distr_men, x = 'trans_city', y = 'amount', 
             template=main_template, 
             title = 'distribution of spending by city among men (relevant part)', 
             labels={'amount':'spending data', 'trans_city':'city name'})
del city_distr_men

fig_distr_city_men.update_layout( 
    yaxis=dict(
        range=[0,2*10**6]))

st.plotly_chart(fig_distr_city_men)
del fig_distr_city_men
fig_distr_city_women = px.box(city_distr_women, x = 'trans_city', y = 'amount', 
             template=main_template, 
             title = 'distribution of spending by city among women (relevant part)', 
             labels={'amount':'spending data', 'trans_city':'city name'})
del city_distr_women

fig_distr_city_women.update_layout( 
    yaxis=dict(
        range=[0,2*10**6]))
st.plotly_chart(fig_distr_city_women)
del fig_distr_city_women


st.text(
    "While researching the date set, discovered a woman\n"\
    "who was able to spend 262 million rubles in a year in Penza. \n"\
    "HOW? BOUGHT PENZA?"
)


st.header("PENZA ANOMALY")
st.subheader("Overview")

st.text(
     "For comparison, I decided to display the median \n"\
    "expenditure of a Penza resident, \n"\
    "but it just doesn't show up (152,000 rub)"
)

penza_anomaly_data = transactions.loc[
    transactions.index=='a288a664b343fb3b2ce8fc48ccfa328b']

penza_anomaly_spend = penza_anomaly_data.loc[penza_anomaly_data['amount']<0]
penza_anomaly_income = penza_anomaly_data.loc[penza_anomaly_data['amount']>0]

penzas_median = transactions.loc[transactions['trans_city'] == 'Penza']
penzas_median = penzas_median.groupby(level=0)['amount'].sum().reset_index()['amount'].median()


fig_penza_overview_income_spend = px.bar(
    x = ['Expenses', 'Income', 'Median value for Penza'], 
    y = [abs(penza_anomaly_spend['amount'].sum()), 
         penza_anomaly_income['amount'].sum(), 
         abs(penzas_median)],
    template=main_template,
    title = 'Scale of the anomaly', 
    labels= {'x':'', 'y': 'amount in rubles'}
)
del penzas_median

st.plotly_chart(fig_penza_overview_income_spend)
del fig_penza_overview_income_spend
st.text("Just in case: all transactions of this user are registered in Penza.")
st.text("Let's see for what period and in what amount transactions were made")

def month_determ(data: str) -> int:
    return int(data.split()[0])//30 - 2

penza_data_month_spend = penza_anomaly_spend.copy()
penza_data_month_spend['trans_time'] = penza_data_month_spend['trans_time'].apply(month_determ)
penza_data_month_spend_gr = penza_data_month_spend.groupby('trans_time')['amount'].sum().reset_index()

del penza_data_month_spend

penza_data_month_spend_gr['amount'] = penza_data_month_spend_gr['amount'].abs()
penza_data_month_spend_gr['type'] = 'expenses'

penza_data_month_income = penza_anomaly_income.copy()
penza_data_month_income['trans_time'] = penza_data_month_income['trans_time'].apply(month_determ)
penza_data_month_income_gr = penza_data_month_income.groupby('trans_time')['amount'].sum().reset_index()
penza_data_month_income_gr['type'] = 'income'

gr_penza_month = pd.concat([penza_data_month_income_gr, penza_data_month_spend_gr])
del penza_data_month_spend_gr
del penza_data_month_income_gr

gr_penza_month = gr_penza_month.sort_values('trans_time')
fig_penza_bar_month = px.bar(
    gr_penza_month, 
    y = 'amount', 
    x = 'trans_time',
    color='type',
    barmode='group', 
    template = main_template, 
    title = 'Duration of the anomaly observation',
    labels={'trans_time': 'observed month'}
)
del gr_penza_month

st.plotly_chart(fig_penza_bar_month)
del fig_penza_bar_month

st.text(
     "Hmmm, maybe most of the transactions were completely on any one day?\n"\
    "It could have been the day she bought Penza, for example. \n"\
    "Let's build a histogram with the sum of transactions by dayFor \n"\
    "the readability of the histogram, let's consider only month 4"
)

ext_penza = penza_anomaly_data.copy()
del penza_anomaly_data

ext_penza = ext_penza.reset_index().drop(columns=['client_id'])
ext_penza['type'] = np.where(ext_penza['amount'] > 0, 'income', 'expenses')
ext_penza['amount'] = ext_penza['amount'].abs()

def tranform_to_onlyDays(data:str) -> int:
    return int(data.split()[0])

four_month_penza = ext_penza.copy()
four_month_penza['trans_time'] = four_month_penza['trans_time'].apply(tranform_to_onlyDays)
four_month_penza = four_month_penza.loc[four_month_penza['trans_time']<=210]
four_month_penza = four_month_penza.loc[four_month_penza['trans_time']>=180]

four_month_penza_sped = four_month_penza.loc[four_month_penza['type']=='expenses']
four_month_penza_income = four_month_penza.loc[four_month_penza['type']=='income']

four_month_penza_sped = four_month_penza_sped.groupby('trans_time')['amount'].sum().reset_index()
four_month_penza_income = four_month_penza_income.groupby('trans_time')['amount'].sum().reset_index()

four_month_penza_sped['type'] = 'expenses'
four_month_penza_income['type'] = 'income'

four_month_penza = pd.concat([four_month_penza_sped, 
                              four_month_penza_income])
del four_month_penza_sped
del four_month_penza_income

def four_month_clear_days(data: str) -> int:
    return data-180

four_month_penza['trans_time'] = four_month_penza['trans_time'].apply(four_month_clear_days)

fig_penza_hist_days = px.bar(
    four_month_penza, 
    x = 'trans_time', 
    y = 'amount', 
    color='type', 
    template=main_template, 
    barmode='group', 
    title = "Distribution of expenditure and income by days of the fourth month", 
    labels={'trans_time':'day'}
)
del four_month_penza

st.plotly_chart(fig_penza_hist_days)
del fig_penza_hist_days
st.text(
     "Conclusion: no, it wasn't one big transaction on \n"\
    "any given day, the distribution is roughly uniform"
)


st.subheader('Categories')
st.text(
     "Let's find out what she spent her money on and where she got it from."
)

penza_times = ext_penza.copy()
def mcc_code_means(code: str) -> str:
    return mcc_codes.loc[code]['mcc_description']

ext_penza['mcc_code'] = ext_penza['mcc_code'].apply(mcc_code_means)

df = ext_penza.groupby('mcc_code')['amount'].sum().reset_index()
df
fig_top3_penza_spend = px.bar(
    ext_penza.loc[ext_penza['type'] == 'expenses'].groupby(
        'mcc_code')['amount'].sum().reset_index().sort_values('amount', ascending=False),
        y = 'amount', x = [x+1 for x in range(3)], 
        template=main_template,
        color='mcc_code', 
        labels={'x':'', 'mcc_code':'category'}, 
        title = 'Income categories'
        )

st.plotly_chart(fig_top3_penza_spend)
del fig_top3_penza_spend

fig_top3_penza_income = px.bar(
    ext_penza.loc[ext_penza['type'] == 'income'].groupby(
        'mcc_code')['amount'].sum().reset_index().sort_values('amount', ascending=False),
        y = 'amount', x = ['1', '2'], 
        template=main_template,
        color='mcc_code', 
        labels={'x':'', 'mcc_code':'category'}, 
        title = 'Spending categories'
        )

del ext_penza
st.plotly_chart(fig_top3_penza_income)
del fig_top3_penza_income

st.text(
     "So, we won't get anything interesting out of the categories, \n"\
    "let's look at which ATMs and what time she withdrew the money from"
)


st.subheader("ATMs")
st.text(
     "Let's find out which terminals she used to withdraw millions in Penza. \n"\
    "Might be something interesting"
)
penza_term = penza_anomaly_spend.groupby('term_id')['amount'].sum().reset_index()
penza_term['amount'] = penza_term['amount'].abs()
fig_penza_term = px.bar(penza_term, 
             x ='term_id', 
             y = 'amount', 
             template=main_template, 
             labels={'term_id':'terminal id'}, 
             title = 'What terminals she used to withdraw money from')

del penza_term
st.plotly_chart(fig_penza_term)
del fig_penza_term

st.text(
     "Conclusion: \n"\
    "nothing particularly interesting, but there are ATMs \n"\
    "in Penza with 20 million roubles in them."
)

st.subheader("Anomaly time")
st.text(
     "Let's see at what time the money is received and debited from the account. \n"\
    "Maybe she withdraws the money deep into the night, \n"\
    "in the silence of the night, away from the eyes of witnesses."
)

def sep_to_hours_int(data: str) -> int:
    return int(data[-8:-6])

penza_anomaly_spend['trans_time'] = penza_anomaly_spend['trans_time'].apply(sep_to_hours_int)
penza_anomaly_spend = pd.concat(
    [penza_anomaly_spend, 
     pd.DataFrame([[x, 0] for x in range(24)],
                  columns=['trans_time', 'amount'])])

penza_anomaly_spend = penza_anomaly_spend.groupby('trans_time')['amount'].sum().reset_index().sort_values('trans_time')
penza_anomaly_spend['amount'] = penza_anomaly_spend['amount'].abs()
penza_anomaly_spend['type'] = 'expenses'


penza_anomaly_income['trans_time'] = penza_anomaly_income['trans_time'].apply(sep_to_hours_int)
penza_anomaly_income = pd.concat(
    [penza_anomaly_spend, 
     pd.DataFrame([[x, 0] for x in range(24)],
                  columns=['trans_time', 'amount'])])

penza_anomaly_income = penza_anomaly_income.groupby('trans_time')['amount'].sum().reset_index().sort_values('trans_time')
penza_anomaly_income['type'] = 'income'

penza_times = pd.concat([penza_anomaly_income, penza_anomaly_spend])
del penza_anomaly_spend
del penza_anomaly_income

fig_penza_times = px.bar(
    penza_times, 
    x = 'trans_time', 
    y = 'amount', 
    color='type', 
    barmode='group', 
    title = 'At what time she spent and received the money',
    labels={'trans_time':'hour'}, 
    template=main_template
)
del penza_times

fig_penza_times.update_xaxes(minor=dict(ticklen=6, tickcolor="white"))

st.plotly_chart(fig_penza_times)
del fig_penza_times

st.text(
     "No, nothing noteworthy. \n"\
    "Except that money is debited and received at the same time of day. \n"\
    "In the end, it turned out to be a rather boring anomaly. \n"\
    "But our job here is just to draw graphs, isn't it?"
)






st.header("Correlations")
st.subheader("Restaurants")
st.text(
     "Since the average or median spend is irrelevant,\n"\
    "let's display a graph of the sum of transactions"
)

restaurant_codes = [5812,5813,5814]
restaurant_data = extended_transactions.loc[
    extended_transactions['mcc_code'].isin(restaurant_codes)]
del restaurant_codes
def sep_time_and_data(trans_time):
    return (trans_time[-8:-6])

restaurant_data['trans_time'] = (restaurant_data['trans_time'].apply(sep_time_and_data))
restaurant_data['amount'] = restaurant_data['amount'].abs()

rest_woman_temp = restaurant_data.loc[restaurant_data['gender']==0]
rest_man_temp = restaurant_data.loc[restaurant_data['gender']==1]

rest_woman_temp = rest_woman_temp.groupby('trans_time')['amount'].sum().reset_index()
rest_man_temp = rest_man_temp.groupby('trans_time')['amount'].sum().reset_index()

rest_woman_temp['gender']='female'
rest_man_temp['gender']='male'

count_rest_data = pd.concat([rest_woman_temp, 
                                rest_man_temp])
del rest_woman_temp
del rest_man_temp

fig_bar_rest_time_sum = px.bar(count_rest_data, 
             y='amount', x = 'trans_time', 
             color='gender', 
             template=main_template, 
             barmode='relative', 
             labels={'trans_time':'hour'}, 
             title = 'Amount spent on restaurants by men and women by hour (sum, barmode=relative)')

st.plotly_chart(fig_bar_rest_time_sum)
del fig_bar_rest_time_sum

st.text(
    "find out what is the ratio between transfers \n"\
    "and payment by terminal in restaurants"
)

nan_terms = restaurant_data['term_id'].isna().sum()

fig_terms_trans_pie = px.pie(
    values=[nan_terms, len(restaurant_data)-nan_terms], 
    names=['by terminal', 'by transfer'],
    template=main_template, 
    title='Ratio of the number of transactions paid via terminal and transfer', 
    hole=0.3
    )
fig_terms_trans_pie.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,
                  marker=dict(line=dict(color='#ffffff', width=0.5)))
del nan_terms

st.plotly_chart(fig_terms_trans_pie)
del fig_terms_trans_pie

def check_terminal_present(term_id: str) -> str:
    if len(str(term_id))!=3:
        return 'By terminal'
    return 'By transfer'

restaurant_data['term_id'] = restaurant_data['term_id'].apply(check_terminal_present)

restaurant_data_women = restaurant_data.loc[restaurant_data['gender']==0]
restaurant_data_men = restaurant_data.loc[restaurant_data['gender']==1]

restaurant_data_men = restaurant_data_men.groupby('term_id')['amount'].count().reset_index()
restaurant_data_women = restaurant_data_women.groupby('term_id')['amount'].count().reset_index()

restaurant_data_men['gender'] = 'male'
restaurant_data_women['gender'] = 'female'


restaurant_data_4_lines = pd.concat([restaurant_data_men, restaurant_data_women])
del restaurant_data_women
del restaurant_data_men

fig = px.bar(
    restaurant_data_4_lines, 
    x = 'gender', 
    y = 'amount', 
    color='term_id', 
    barmode='group', 
    template=main_template,
    labels={'term_id':'type'},
    title='Who and how to make payments.' 
)
del restaurant_data_4_lines

st.plotly_chart(fig)
del fig

fig_box_city_rest = px.box(
    restaurant_data, 
    x = 'trans_city', 
    y = 'amount', 
    color='gender', 
    template=main_template,
    labels={'trans_city':'city',
            '1':'male', 1:'male', 
            '0':'female', 0:'female'},
    title = 'Spending on restaurants in different Russian cities (1-male, 0-female)' 

)
del restaurant_data

fig_box_city_rest.update_layout( 
    yaxis=dict(
        range=[0,1000]))

st.plotly_chart(fig_box_city_rest)
del fig_box_city_rest



st.subheader("Car owners")
st.text(
     "Let's compare the costs of car owners and those who don't have a car. \n"\
    "We will guess who has a car and who does not by the presence of \n"\
    "transactions characteristic for car owners, \n"\
    "such as payment for car service or petrol stations."
)

st.text(
     "As a result, out of 7559 users, 3924 thousand are car owners \n"\
    "(well, or they paid someone for a service station). \n"\
    "Car owners made 2_115_721 out of 3_563_529 transactions"
)

codes_car = [5532,5533,5541,5542,7523,7531,7538]

car_owners = extended_transactions.loc[extended_transactions['mcc_code'].isin(codes_car)]
del codes_car
car_owners = set(car_owners.index.values.tolist())
tr_cars = extended_transactions.loc[extended_transactions.index.isin(car_owners)]

car_genders = gender_train.loc[gender_train.index.isin(car_owners)].drop(['Unnamed: 0'], axis = 1).reset_index()
car_genders = car_genders.groupby('gender')['client_id'].count().reset_index()
car_genders['gender'] = car_genders['gender'].apply(change_bool_to_names)

st.dataframe(car_genders.head(100))

fig_car_genders_pie = px.pie(
    car_genders, 
    values='client_id',
    names='gender',
    template=main_template, 
    title = 'Gender distribution among car owners',
    hole=0.6
)
del car_genders
st.plotly_chart(fig_car_genders_pie)
del fig_car_genders_pie

st.text("Let's look at the ratio of male to female motorists")

st.text(
     "Conclusion, there is no clear correlation, \n"\
    "this difference can be attributed to a slight advantage \n"\
    "of men over women in the number of represented users"
)
st.text(
     "Let's see if there is a correlation between income and having a car"
)

no_cars_income = extended_transactions.loc[
    (extended_transactions.index.isin(car_owners)==False) & (extended_transactions['amount']>0)]
del car_owners

no_cars_gr_inc = no_cars_income.groupby(level=0)['amount'].sum().reset_index()
del no_cars_income
no_cars_gr_inc['car'] = 'carless'


cars_income = tr_cars.loc[tr_cars['amount']>0]
del tr_cars
cars_gr_inc = cars_income.groupby(level=0)['amount'].sum().reset_index()
del cars_income
cars_gr_inc['car'] = 'have car'

merge_cars_inc = pd.concat([cars_gr_inc,no_cars_gr_inc]).set_index('client_id')
del no_cars_gr_inc

fig_car_income_box = px.box(
    merge_cars_inc, 
    x = 'car', 
    y = 'amount', 
    color='car', 
    title = 'Comparison of incomes of people with and without a car', 
    template=main_template, 
    labels={'car':'type'}
)
del merge_cars_inc

fig_car_income_box.update_layout(yaxis=dict(range=[0,1.5*10**6]))

st.plotly_chart(fig_car_income_box)
del fig_car_income_box

st.text(
     "Thus we can see a noticeable correlation between earnings and having a car"
)

st.text(
     "I could make a graph of the dependence between the amount of spendings on\n"\
    "cars and total spending for example, it would make a nice scatter plot, \n"\
    "but I spent too much time on this project, \n"\
    "and I hope the project turned out well without it, right?"
)


st.header("Medicine")

st.text(
    "Let's take a look at when people commit to medical spending. \n"\
    "Maybe there's a seasonality there"
)

st.text("We got the following date set: \n"\
        "(first 100 rows of 69'691)")

# get all medicine spendings transactions
codes_medicine = [5912,8071,8099,8062,8011,5122]
medicine_spendings = extended_transactions.loc[
    (extended_transactions['mcc_code'].isin(codes_medicine)) &
    (extended_transactions['amount']<0)]
del codes_medicine

# remove time data and group them by week for clarity. Believe me, a 450-day chart is overnoisy
def keep_only_weeks_int(trans_time: str) -> int:
    return (int(trans_time.split()[0])//7)

medicine_spendings['trans_time'].replace('', np.nan, inplace=True)
medicine_spendings.dropna(subset=['trans_time'], inplace=True)
    
medicine_spendings['trans_time'] = medicine_spendings['trans_time'].apply(keep_only_weeks_int)

# remove the oversized transactions so they don't spoil the picture:
def remove_ext_medicine(amount: float) -> float:
    return (amount>-10_000)*amount

medicine_spendings['amount'] = medicine_spendings['amount'].apply(remove_ext_medicine)

#group by day number:
medicine_spendings_gr = medicine_spendings.groupby('trans_time')['amount'].sum().abs().reset_index().sort_values('trans_time')
# let's delete the last incomplete week so that the graph doesn't fall at the end 
medicine_spendings_gr = medicine_spendings_gr.head(len(medicine_spendings_gr)-1)

#abs for non-grooped dataframe (used for scatter and box plots)
medicine_spendings['amount'] = medicine_spendings['amount'].abs()    

st.dataframe(medicine_spendings.head(100))

st.text(
     "Let's display the data split by week, \n"\
    "since a displaying day is not representative."
)

fig_medicine_line = px.line(
    medicine_spendings_gr,
    x = 'trans_time', #day number
    y = 'amount', 
    template=main_template, 
    title='Spending on medicine in 450 days',
    labels={'trans_time':'week number'}
)   
del medicine_spendings_gr

st.plotly_chart(fig_medicine_line)
del fig_medicine_line

st.text("Hmm, total spending across all medical categories looks uninformative, \n"\
    "let's break this graph down into a few")


fig_medicine_scatter = px.scatter(
    medicine_spendings, 
    x = 'trans_time', 
    y = 'amount', 
    template=main_template, 
    title='Spending on medicine in 450 days',
    labels={'trans_time':'week number'}, 
    color='mcc_code'
)

st.plotly_chart(fig_medicine_scatter)
del fig_medicine_scatter

#preparing dataframe for ulta_mega_super_combined_line_graph

medicine_mcc_gr = medicine_spendings.groupby(['trans_time', 'mcc_code'])['amount'].sum().reset_index().sort_values('trans_time')
medicine_mcc_gr = medicine_mcc_gr.head(len(medicine_mcc_gr)-4)

medicine_mcc_gr['mcc_code'] = medicine_mcc_gr['mcc_code'].apply(mcc_code_means)
medicine_mcc_gr=medicine_mcc_gr.sort_values(['trans_time', 'mcc_code'])

fig_medicine_line_pro = px.line(
    medicine_mcc_gr,
    x = 'trans_time',
    y = 'amount', 
    color='mcc_code',
    template=main_template, 
    title='Spending on medicine in 450 days', 
    labels={'trans_time':'week number', 
            'mcc_code':'category:'},
)
del medicine_mcc_gr

fig_medicine_line_pro.update_layout(
    legend_orientation='h')
fig_medicine_line_pro.update_layout(
    legend_x = 0, 
    legend_y = -0.25,
    xaxis_type = "category",
    margin_r = 40,
    legend_title_text = "Category:",
)

st.plotly_chart(fig_medicine_line_pro)
del fig_medicine_line_pro


st.text(
     "Based on this, it is possible to make an assumption about the approximate time \n"\
    "of occurrence of autumn and spring - periods when many people get sick with SARS."
)
st.text("Let's look at the correlation between income and spending on medicine:")

users_total_spend = extended_transactions.loc[extended_transactions['amount']<0].reset_index().groupby(
    ['client_id', 'gender'])['amount'].sum().abs().reset_index().set_index('client_id').sort_values('amount')


med_users_total_spent = medicine_spendings.reset_index().groupby(
    ['client_id', 'gender', 'mcc_code']
)['amount'].sum().reset_index().set_index('client_id').sort_index()
del medicine_spendings

med_income_spend_correl = pd.merge(
    med_users_total_spent,
    users_total_spend, 
    right_index=True, 
    left_index=True
).drop(columns=['gender_y']).rename(
    columns={
        'gender_x':'gender', 
        'amount_x':'spent',
        'amount_y':'income'
        })
del med_users_total_spent

# rename genders: from 0-1 to mele-female
med_income_spend_correl['gender'] = np.where(
    med_income_spend_correl['gender'], 'male', 'female')

# Let’s remove the extremes, as they prevent us from understanding the general trend
med_income_spend_correl = med_income_spend_correl.loc[
    (med_income_spend_correl['income']<5_000_000) &
    (med_income_spend_correl['spent']<100_000)
]
fig_med_income_scatter = px.scatter(
    med_income_spend_correl,
    x = 'income', 
    y = 'spent', 
    color='gender', 
    title='The dependence of spending on medicine on total income', 
    template=main_template,
    log_x=False,
)

st.plotly_chart(fig_med_income_scatter)
del fig_med_income_scatter

st.text(
     "The correlation is weak due to the high density near the origin. \n"\
    "Let’s try to make a bar chart out of this. \n"\
    "To do this, we will round the income, group by it, \n"\
    "and take the median value of spending on medicine."
)

# grouping by income (rounding), take spent median

def round_income(income_value: float) -> int:
    return (income_value//15_000) * 15_000

# round users income:
med_income_spend_correl['income'] = med_income_spend_correl['income'].apply(round_income)

# grouping by income:
med_income_spend_correl = med_income_spend_correl.reset_index().groupby(
    ['income', 'gender', 'mcc_code']
)['spent'].median().reset_index().sort_values('income')

# change mcc codes to names:

med_income_spend_correl['mcc_code'] = med_income_spend_correl['mcc_code'].apply(mcc_code_means)

# cut the income range:
med_income_spend_correl = med_income_spend_correl.loc[med_income_spend_correl['income']<10**6]

fig_medicine_income_bar_pro = px.bar(
    med_income_spend_correl,
    x = 'income', 
    y = 'spent', 
    color='mcc_code', 
    title='The dependence of spending on medicine on total income', 
    template=main_template, 
)
fig_medicine_income_bar_pro.update_layout(
    legend_orientation='h', 
    legend_y = -0.25, 
    legend_title = 'Category:'
)
fig_medicine_income_bar_pro.add_trace(
    go.Line(
        x = med_income_spend_correl.groupby(['income'])['spent'].sum().reset_index()['income'],
        y = med_income_spend_correl.groupby(['income'])['spent'].sum().reset_index()['spent'], 
        name = 'category sum'
    )
)
del med_income_spend_correl

st.plotly_chart(fig_medicine_income_bar_pro)
del fig_medicine_income_bar_pro
st.text(
    "Yes, there is a correlation between a person's income and their spending on medicine. \n"\
    "The more a person earns, the more likely they spend on medicine. \n"\
    "Another interesting observation is that women spend more on medicine than men on average"
)

st.header("Trevelling")
st.text(
     "Let's look at the transactions related to leisure.\n"\
    "Such as: rental of cars, railway and air transportation, hotels, etc."
)
st.text("Thus, we received about 15 thousand transactions, which, in fact, is not a lot")
# travelling codes
codes_trevel = [3000,3351,3501,4112,4411,4511,4722,7011,7512,7991]

trev_transactions = extended_transactions.loc[
    extended_transactions['mcc_code'].isin(codes_trevel)
]
del extended_transactions
del codes_trevel

trev_transactions = trev_transactions.loc[
    trev_transactions['amount']<0
]
trev_transactions['amount'] = trev_transactions['amount'].abs()

trev_transactions['trans_time'].replace('', np.nan, inplace=True)
trev_transactions.dropna(subset=['trans_time'], inplace=True)

st.dataframe(trev_transactions)

st.text(
     "We group expenses by week and then look at which week has how \n"\
    "much spent on travelling, in order to see the seasonality."
)

# separate days
trev_transactions['trans_time'] = trev_transactions['trans_time'].apply(keep_only_weeks_int)

trev_transactions_gr = trev_transactions.groupby(
    ['trans_time', 'gender', 'mcc_code'])['amount'].sum().reset_index().sort_values('trans_time')

trev_transactions_gr['amount'] = trev_transactions_gr['amount'].abs() 
# let's delete the last incomplete week so that the graph doesn't fall at the end 
trev_transactions_gr = trev_transactions.head(len(trev_transactions)-1)

# abs for non-grooped dataframe (used for scatter and box plots)
trev_transactions['amount'] = trev_transactions['amount'].abs() 

#remove extremums
def remove_extremums_trevel(amount: float) -> float:
    return (amount<100_000)*amount  

trev_transactions['amount'] = trev_transactions['amount'].apply(remove_extremums_trevel)


# change mcc codes to their russian names
trev_transactions_gr['mcc_code'] = trev_transactions_gr['mcc_code'].apply(mcc_code_means)

#group categories; drop: city, gender, term_id, trans_type
trev_transactions_gr = trev_transactions_gr.groupby(['trans_time', 'mcc_code'])['amount'].sum().reset_index()

fig_trevel_days_distr_bar = px.bar(
    # remove last not-full week
    trev_transactions_gr.head(len(trev_transactions_gr)-7), 
    x = 'trans_time', 
    y = 'amount', 
    color='mcc_code', 
    title='Distribution of travel expenses by weeks:', 
    labels={'trans_time':'week number'}, 
    template=main_template
)

fig_trevel_days_distr_bar.update_layout(
    legend_orientation='h', 
    legend_title = 'Category:', 
    legend_y=-0.25,
    legend_x = -0.1
)

st.plotly_chart(fig_trevel_days_distr_bar)
del fig_trevel_days_distr_bar

st.text(
     "We should group categories of expenses, \n"\
    "and, finally, translate them into English."
)

def group_data_trevel(mcc_code:str) -> str:
    group_data = {
        'Авиалинии, авиакомпании':'airline tickets',
        'Авиалинии, авиакомпании, нигде более не классифицированные':'airline tickets',
        'Жилье — отели, мотели, курорты': 'hotels', 
        'Отели, мотели, базы отдыха, сервисы бронирования':'hotels', 
        'Круизные линии':'cruises', 
        'Пассажирские железные перевозки':'railway tickets',
        'Прокат автомобилей':'car hire',
        'Агентства по аренде автомобилей':'car hire', 
        'Туристические агентства и организаторы экскурсий':'entertainments',
        'Туристические аттракционы и шоу':'entertainments'
    }
    return group_data[mcc_code]

trev_transactions_gr['mcc_code'] = trev_transactions_gr['mcc_code'].apply(group_data_trevel)

trev_transactions_gr = trev_transactions_gr.groupby(['trans_time', 'mcc_code'])['amount'].sum().reset_index()

trev_code = """
def group_data_trevel(mcc_code:str) -> str:
    group_data = {
        'Авиалинии, авиакомпании':'airline tickets',
        'Авиалинии, авиакомпании, нигде более не классифицированные':'airline tickets',
        'Жилье — отели, мотели, курорты': 'hotels', 
        'Отели, мотели, базы отдыха, сервисы бронирования':'hotels', 
        'Круизные линии':'cruises', 
        'Пассажирские железные перевозки':'railway tickets',
        'Прокат автомобилей':'car hire',
        'Агентства по аренде автомобилей':'car hire', 
        'Туристические агентства и организаторы экскурсий':'entertainments',
        'Туристические аттракционы и шоу':'entertainments'
    }
    return group_data[mcc_code]

trev_transactions_gr['mcc_code'] = trev_transactions_gr['mcc_code'].apply(group_data_trevel)

trev_transactions_gr = trev_transactions_gr.groupby(['trans_time', 'mcc_code'])['amount'].sum().reset_index()
"""
st.code(trev_code, language='python')

fig_adv_trevel_days_distr_bar = px.bar(
    # remove last not-full week
    trev_transactions_gr.head(len(trev_transactions_gr)-7), 
    x = 'trans_time', 
    y = 'amount', 
    color='mcc_code', 
    title='Distribution of travel expenses by weeks:', 
    labels={'trans_time':'week number'}, 
    template=main_template
)

fig_adv_trevel_days_distr_bar.update_layout(
    legend_orientation='h', 
    legend_title = 'Category:', 
    legend_y=-0.25,
    legend_x = -0.1
)
st.plotly_chart(fig_adv_trevel_days_distr_bar)
del fig_adv_trevel_days_distr_bar
st.text(
     "Let's now construct a linear graph and increase the step: \n"\
    "now we will group by 2 weeks."
)


def one_to_two_week(week_number:int) -> int:
    return (week_number//2) * 2

trev_transactions_gr_two = trev_transactions_gr.copy()
trev_transactions_gr_two['trans_time'] = trev_transactions_gr['trans_time'].apply(one_to_two_week)
trev_transactions_gr_two = trev_transactions_gr.groupby(['trans_time', 'mcc_code'])['amount'].sum().reset_index()

fig_adv_trevel_days_distr_line = px.line(
    # remove last not-full week
    trev_transactions_gr_two.head(len(trev_transactions_gr)-7), 
    x = 'trans_time', 
    y = 'amount', 
    color='mcc_code', 
    title='Distribution of travel expenses by weeks:', 
    labels={'trans_time':'week number'}, 
    template=main_template,
    line_shape='spline'
)
del trev_transactions_gr
del trev_transactions_gr_two

fig_adv_trevel_days_distr_line.update_layout(
    legend_orientation='h', 
    legend_title = 'Category:', 
    legend_y=-0.25,
    legend_x = -0.1
)

st.plotly_chart(fig_adv_trevel_days_distr_line)
del fig_adv_trevel_days_distr_line

st.text(
     "Conclusion: \n"\
    "a slight tendency towards seasonal vacation can be traced, \n"\
    "but it is not so obvious."
)
st.text("Let's try to find a correlation between income and travel expenses:")


trev_users_total_spent = trev_transactions.reset_index().groupby(
    ['client_id', 'gender', 'mcc_code']
)['amount'].sum().reset_index().set_index('client_id').sort_index()
# group categories
trev_users_total_spent['mcc_code'] = trev_users_total_spent['mcc_code'].apply(mcc_code_means)
trev_users_total_spent['mcc_code'] = trev_users_total_spent['mcc_code'].apply(group_data_trevel)

trev_income_spend_correl = pd.merge(
    trev_users_total_spent,
    users_total_spend, 
    right_index=True, 
    left_index=True
).drop(columns=['gender_y']).rename(
    columns={
        'gender_x':'gender', 
        'amount_x':'spent',
        'amount_y':'income'
        })

del users_total_spend
del trev_users_total_spent

# rename genders: from 0-1 to mele-female
trev_income_spend_correl['gender'] = np.where(
    trev_income_spend_correl['gender'], 'male', 'female')

# Let’s remove the extremes, as they prevent us from understanding the general trend
trev_income_spend_correl = trev_income_spend_correl.loc[
    (trev_income_spend_correl['income']<5_000_000) &
    (trev_income_spend_correl['spent']<50_000)
]

fig_trev_income_scatter = px.scatter(
    trev_income_spend_correl,
    x = 'income', 
    y = 'spent', 
    color='gender', 
    title='The dependence of spending on treveling on total income', 
    template=main_template,
    log_x=False,
    
    
)
st.plotly_chart(fig_trev_income_scatter)
del fig_trev_income_scatter


st.text(
     "Let's try to make the correlation more visible: \n"\
    "round up income and group users by it, \n"\
    "taking the median travel expenses."
)

# grouping by income (rounding), take spent median

# round users income:
trev_income_spend_correl['income'] = trev_income_spend_correl['income'].apply(round_income)

# grouping by income:
trev_income_spend_correl = trev_income_spend_correl.reset_index().groupby(
    ['income', 'gender', 'mcc_code']
)['spent'].median().reset_index().sort_values('income')

# change mcc codes to names:

# cut the income range:
trev_income_spend_correl = trev_income_spend_correl.loc[trev_income_spend_correl['income']<10**6]

fig_trevel_income_bar_pro = px.bar(
    trev_income_spend_correl,
    x = 'income', 
    y = 'spent', 
    color='mcc_code', 
    title='The dependence of spending on trevelling on total income', 
    template=main_template, 
)
fig_trevel_income_bar_pro.update_layout(
    legend_orientation='h', 
    legend_y = -0.25, 
    legend_title = 'Category:'
)
fig_trevel_income_bar_pro.add_trace(
    go.Line(
        x = trev_income_spend_correl.groupby(['income'])['spent'].sum().reset_index()['income'],
        y = trev_income_spend_correl.groupby(['income'])['spent'].sum().reset_index()['spent'], 
        name = 'category sum',
        line_shape = 'spline'
    )
)
del trev_income_spend_correl

st.plotly_chart(fig_trevel_income_bar_pro)
del fig_trevel_income_bar_pro

st.text(
     "Conclusion: there are no a correlation between income and travel expenses."
)

st.text(
     "Let's see what contribution different cities \n"\
    "made to the distribution of travel expenses by weeks"
)

trev_city = trev_transactions.groupby(['trans_time', 'trans_city'])['amount'].sum().reset_index()

#make smother var
trev_city_smooth = trev_city.copy()
del trev_city
trev_city_smooth = trev_city_smooth.copy()
trev_city_smooth['trans_time'] = trev_city_smooth['trans_time'].apply(one_to_two_week)
trev_city_smooth = trev_city_smooth.groupby(['trans_time', 'trans_city'])['amount'].sum().reset_index()

fig_trev_city_week_bar = px.bar(
    trev_city_smooth, 
    x = 'trans_time',
    y = 'amount',
    color='trans_city', 
    title="Travelling expenses in different cities of Russia", 
    labels={'trans_time': 'week_number'}, 
    template=main_template
)

fig_trev_city_week_bar.update_layout(
    legend_orientation='h', 
    legend_title = 'Category:', 
    legend_y=-0.25,
    legend_x = -0.1
)

st.plotly_chart(fig_trev_city_week_bar)
del fig_trev_city_week_bar

st.text(
     "Conclusion: \n"\
    "in all cities spend almost the same amount on travel, \n"\
    "but in Kazan and Moscow spend a little more."
)

st.text(
     "Let's look at the statistics for cities. \n"\
    "First we will find out who spends money on trips and when, \n"\
    "maybe the seasonality of different cities is different."
)

fig_trev_city_week_line = px.line(
    trev_city_smooth, 
    x = 'trans_time',
    y = 'amount',
    color='trans_city', 
    title="Travelling expenses in different cities of Russia", 
    labels={'trans_time': 'week_number'}, 
    template=main_template,
)
del trev_city_smooth

fig_trev_city_week_line.update_layout(
    legend_orientation='h', 
    legend_title = 'Category:', 
    legend_y=-0.25,
    legend_x = -0.1
)

st.plotly_chart(fig_trev_city_week_line)
del fig_trev_city_week_line

st.text(
     "Conclusion: \n"\
    "different cities indeed have different seasons of travel spending, \n"\
    "but there are also cities with very similar seasonality."
)

st.text("Finally, let's compare travel expenses in different cities.")

fig_trev_city_box = px.box(
    trev_transactions, 
    x = 'trans_city', 
    y = 'amount', 
    title = 'The travel expenses of inhabitants of various cities', 
    labels={'trans_city':'city name'},
    template=main_template
)
del trev_transactions

fig_trev_city_box.update_layout( 
    yaxis=dict(
        range=[0,20_000]
    )
)

st.plotly_chart(fig_trev_city_box)
del fig_trev_city_box
st.text(
    "Conclusion: residents of different cities spend different amounts of money on travel. \n"\
    "Some spend more, some less, some may spend very little, \n"\
    "some may spend a lot if they spend anything at all, and so on."
)
del transactions
del mcc_codes
del trans_types
del gender_train
