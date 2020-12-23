import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gdp  # conda
import geoplot as gplt # pip
import geoplot.crs as gcrs


sns.set_style('whitegrid')


# In[2]:


#Reading the shape file
fp = r'india-polygon.shp'
map_df = gpd.read_file(fp)

map_df.head()

from datetime import date
output_file =  str(date.today())


# In[16]:


data = pd.read_csv("13-Dec.csv",index_col=0)
data.iloc[-1:, -4:] = data.iloc[1:-1,-4:].astype(int).sum().to_list()
# data = data.iloc[-1:,-4:]
data
# type(data['10-Mar'][0])


# In[17]:


total_cases = pd.DataFrame()
total_cured = pd.DataFrame()
total_deaths = pd.DataFrame()


# In[18]:


i=1
for col in data.columns:
    if i%4==1:total_cases[col]=data[col]
    if i%4==3:total_cured[col]=data[col]
    if i%4==0:total_deaths[col]=data[col]
    i+=1


# In[20]:


total_cases.drop(index='NUMBERS', axis=0, inplace=True)
total_cured.drop(index='NUMBERS', axis=0, inplace=True)
total_deaths.drop(index='NUMBERS', axis=0, inplace=True)
total_cases = total_cases.astype(int)
total_cured = total_cured.astype(int)
total_deaths = total_deaths.astype(int)


# In[21]:


tot = total_cases.copy().transpose()
cur = total_cured.copy().transpose()
dea = total_deaths.copy().transpose()
cur.index = tot.index
dea.index = tot.index
active_cases = pd.DataFrame()
for col in tot:
    active_cases[col] = tot[col]-(cur[col]+dea[col])
    
active_cases


# In[22]:


# Generates Total Cases Map
dfm = total_cases.copy()
dfm.drop(dfm.tail(1).index,inplace=True)
for i in list(dfm):
    
    if ((i[0]=='0'and i[1]=='1') or (i[0]=='1'and i[1]=='-')):
        clmn = dfm.index
        df = pd.DataFrame(clmn,columns=['Name of State/UN'])
        df['total_cases'] = list(dfm[i])
        merged = map_df.set_index('st_nm').join(df.set_index('Name of State/UN'))
        merged.head()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.axis('off')
        ax.set_title('Total Cases', fontdict={'fontsize': '25', 'fontweight' : '10'})
        dest = 'Images/total_cases/' + i + '.png'
        # plot the figure
        merged.plot(column='total_cases',cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0', legend=True,legend_kwds={'shrink': 0.6},markersize=[39.739192, -104.990337])
        import matplotlib.pyplot as plt
        plt.savefig(dest, dpi=300, bbox_inches='tight')


# In[9]:


# Generates Total Deaths Map
dfm = total_deaths.copy()
dfm.drop(dfm.tail(1).index,inplace=True)
for i in list(dfm):
    
    if ((i[0]=='0'and i[1]=='1') or (i[0]=='1'and i[1]=='-')):
        clmn = dfm.index
        df = pd.DataFrame(clmn,columns=['Name of State/UN'])
        df['total_deaths'] = list(dfm[i])
        merged = map_df.set_index('st_nm').join(df.set_index('Name of State/UN'))
        merged.head()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.axis('off')
        ax.set_title('Total Deaths', fontdict={'fontsize': '25', 'fontweight' : '10'})
        dest = 'Images/deaths/' + i + '.png'
        # plot the figure
        merged.plot(column='total_deaths',cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0', legend=True,legend_kwds={'shrink': 0.6},markersize=[39.739192, -104.990337])
        import matplotlib.pyplot as plt
        plt.savefig(dest, dpi=300, bbox_inches='tight')


# In[10]:


# Generates Total Cured Map 
dfm = total_cured.copy()
dfm.drop(dfm.tail(1).index,inplace=True)
for i in list(dfm):
    
    if ((i[0]=='0'and i[1]=='1') or (i[0]=='1'and i[1]=='-')):
        clmn = dfm.index
        df = pd.DataFrame(clmn,columns=['Name of State/UN'])
        df['total_cured'] = list(dfm[i])
        merged = map_df.set_index('st_nm').join(df.set_index('Name of State/UN'))
        merged.head()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.axis('off')
        ax.set_title('Total Cured', fontdict={'fontsize': '25', 'fontweight' : '10'})
        dest = 'Images/cured_cases/' + i + '.png'
        # plot the figure
        merged.plot(column='total_cured',cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0', legend=True,legend_kwds={'shrink': 0.6},markersize=[39.739192, -104.990337])
        import matplotlib.pyplot as plt
        plt.savefig(dest, dpi=300, bbox_inches='tight')


# In[ ]:





# In[39]:


from datetime import timedelta, date

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

start_dt = date(2020, 4, 15)
end_dt = date(2020, 12, 1)
time_axis = []
for dt in daterange(start_dt, end_dt):
    time_axis.append(dt)
print(len(time_axis))


# In[40]:


cases = total_cases.copy().transpose()
cases = cases[36:-11]
cases


# In[43]:


# cases = total_cases.copy().transpose()
date= pd.to_datetime(time_axis)
cases['date'] = date
active = active_cases.copy()[36:-11]
active['date'] = date
active


# In[44]:


other_states = [col for col in cases.columns if col not in ["Delhi", "Maharashtra", "Tamil Nadu","Andhra Pradesh", "Totals"]]


# In[45]:


cases["Rest of India"] = cases[other_states].sum(axis=1)
# active["Rest of India"] = active_cases[other_states].sum(axis=1)


# In[46]:


plot_cases = cases[["Delhi", "Maharashtra", "Tamil Nadu","Andhra Pradesh", "Rest of India"]]
major_affected = cases[["Maharashtra", "Tamil Nadu","Andhra Pradesh", "Karnataka"]]
eastern_states = cases[["West Bengal", "Jharkhand", "Bihar", "Odisha"]]
northern_states = cases[["Uttar Pradesh", "Punjab",  "Rajasthan", "Haryana"]]
small_states = cases[["Chandigarh", "Goa", "Sikkim", "Tripura"]]

# plot_active = active[["Delhi", "Maharashtra", "Tamil Nadu", "Rest of India"]]


# In[47]:


date = pd.DataFrame(date)
date.iloc[:,0].dt.is_month_start
plt.rcParams['figure.figsize'] = 12, 6
plt.style.use("fivethirtyeight")


# In[48]:


fig = plt.plot(plot_cases.index,plot_cases)
plt.title("Total Cases Reported")
plt.legend(labels=["Delhi", "Maharashtra", "Tamil Nadu", "Andhra Pradesh", "Rest of India"])
plt.xticks(ticks=range(1,date.shape[0],7), rotation=90)
plt.ylabel("Number of Cases")
plt.savefig('Images/comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# In[49]:


from pylab import text
fig = plt.plot(cases.index,cases["Totals"])
plt.title("TOTAL CASES IN INDIA")
# plt.legend(labels=[])
plt.xticks(ticks=range(1,date.shape[0],7), rotation=90)
plt.ylabel("Number of Cases(in ten lakhs)")
plt.axvline(x=204.5, color="red", linewidth=2)
# plt.grid(b=None)
# text(0.6, 0.5,'Red Line denotes split of training dataset upto 31st September',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes)
plt.savefig('Images/all_india.png', dpi=300, bbox_inches='tight')
plt.show()


# In[12]:


confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')


# In[14]:


import matplotlib.dates as mdates

hotspots = ['Brazil','Spain','US', 'Russia','France','India']
dates = list(confirmed_df.columns[4:])
dates = list(pd.to_datetime(dates))
dates_india = dates[33:]

df1 = confirmed_df.groupby('Country/Region').sum().reset_index()
df2 = deaths_df.groupby('Country/Region').sum().reset_index()
df3 = recovered_df.groupby('Country/Region').sum().reset_index()

global_confirmed = {}
global_deaths = {}
global_recovered = {}
global_active= {}

for country in hotspots:
    k =df1[df1['Country/Region'] == country].loc[:,'1/30/20':]
    global_confirmed[country] = k.values.tolist()[0][25:]

    k =df2[df2['Country/Region'] == country].loc[:,'1/30/20':]
    global_deaths[country] = k.values.tolist()[0][25:]

    k =df3[df3['Country/Region'] == country].loc[:,'1/30/20':]
    global_recovered[country] = k.values.tolist()[0][25:]
    
for country in hotspots:
    k = list(map(int.__sub__, global_confirmed[country], global_deaths[country]))
    global_active[country] = list(map(int.__sub__, k, global_recovered[country]))
    
fig = plt.figure(figsize= (15,15))
plt.suptitle('Active Cases and Deaths in Hotspot Countries',fontsize = 20,y=1.0)
#plt.legend()
k=0
for i in range(1,7):
    ax = fig.add_subplot(6,2,i)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.bar(dates_india,global_active[hotspots[k]],color = 'green',alpha = 0.6,label = 'Active')
#     ax.bar(dates_india,global_recovered[hotspots[k]],color='grey',label = 'Recovered');
    ax.bar(dates_india,global_deaths[hotspots[k]],color='red',label = 'Death')
#     ax.set_facecolor('#000000')
    plt.title(hotspots[k])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    k=k+1

plt.tight_layout(pad=3.0)
plt.savefig('Images/global.png', dpi=300, bbox_inches='tight')
plt.show()


# In[50]:


plot_cases = pd.concat([plot_cases,cases['date']], axis=1)
major_affected = pd.concat([major_affected,cases['date']], axis=1)
eastern_states = pd.concat([eastern_states,cases['date']], axis=1)
northern_states = pd.concat([northern_states,cases['date']], axis=1)
small_states = pd.concat([small_states,cases['date']], axis=1)
# plot_active = pd.concat([plot_active,cases['date']], axis=1)



# In[52]:


import plotly.express as px
fig = px.bar(testingHistory, x="time_stamp", y="testing_no", barmode='group',height=500,color = "testing_no",
             orientation = 'v',color_discrete_sequence = px.colors.sequential.Plasma_r)
fig.update_layout(title_text='Number of COVID-19 test conducted everyday',plot_bgcolor='rgb(275, 270, 273)')
fig.update_layout(barmode='stack')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',yaxis_title='Tests',xaxis_title='Date')
fig.show()


# In[63]:


starting_of_week = []
starting_week_dates = []
i = 0
wc = 0
for date in list(plot_cases.index.values):
    if (i%7 == 0):
        wc +=1
        starting_week_dates.append(date)
    starting_of_week.append(wc)
    i+=1
    
weeks = np.floor((plot_cases.shape[0]+6)/7)
weeks = int(weeks)

plot_cases['week'] = starting_of_week
major_affected['week'] = starting_of_week
eastern_states['week'] = starting_of_week
northern_states['week'] = starting_of_week
small_states['week'] = starting_of_week

# plot_active['week'] = starting_of_week


# plot_cases['week'] = plot_cases['date'].dt.weekofyear
plot_by_week = plot_cases.groupby(['week']).sum()
plot_by_week['week'] = starting_week_dates
# plot_by_week.drop(plot_by_week.tail(1).index,inplace=True)
plot_by_week2 = pd.melt(plot_by_week, id_vars = "week")

plot_by_week_major = major_affected.groupby(['week']).sum()
plot_by_week_major['week'] = starting_week_dates
# plot_by_week_major.drop(plot_by_week_major.tail(1).index,inplace=True)
plot_by_week2_major = pd.melt(plot_by_week_major, id_vars = "week")

plot_by_week_east = eastern_states.groupby(['week']).sum()
plot_by_week_east['week'] = starting_week_dates
# plot_by_week_east.drop(plot_by_week_east.tail(1).index,inplace=True)
plot_by_week2_east = pd.melt(plot_by_week_east, id_vars = "week")

plot_by_week_north = northern_states.groupby(['week']).sum()
plot_by_week_north['week'] = starting_week_dates
# plot_by_week_north.drop(plot_by_week_north.tail(1).index,inplace=True)
plot_by_week2_north = pd.melt(plot_by_week_north, id_vars = "week")

plot_by_week_small = small_states.groupby(['week']).sum()
plot_by_week_small['week'] = starting_week_dates
# plot_by_week_small.drop(plot_by_week_small.tail(1).index,inplace=True)
plot_by_week2_small = pd.melt(plot_by_week_small, id_vars = "week")



# In[65]:


plt.figure(figsize=(10,8))
sns.barplot(x='week',y='value', hue='variable',data=plot_by_week2)
plt.ylabel("Total Number of Cases Per Week (in ten lakhs)")
plt.xticks(rotation = 45)
plt.legend(title="")
plt.savefig('Images/bar_weekly.png', dpi=300, bbox_inches='tight')
plt.show()


# In[66]:


plt.figure(figsize=(10,8))
sns.barplot(x='week',y='value', hue='variable',data=plot_by_week2_major)
plt.ylabel("Total Number of Cases Per Week (in lakhs)")
plt.xticks(rotation = 45)
plt.legend(title="")
plt.savefig('Images/major_weekly.png', dpi=300, bbox_inches='tight')
plt.show()


# In[67]:


plt.figure(figsize=(10,8))
sns.barplot(x='week',y='value', hue='variable',data=plot_by_week2_east)
plt.ylabel("Total Number of Cases Per Week (in lakhs)")
plt.xticks(rotation = 45)
plt.legend(title="")
plt.savefig('Images/eastern_weekly.png', dpi=300, bbox_inches='tight')
plt.show()


# In[68]:


plt.figure(figsize=(10,8))
sns.barplot(x='week',y='value', hue='variable',data=plot_by_week2_north)
plt.ylabel("Total Number of Cases Per Week (in lakhs)")
plt.xticks(rotation = 45)
plt.legend(title="")
plt.savefig('Images/north_weekly.png', dpi=300, bbox_inches='tight')
plt.show()


# In[69]:


plt.figure(figsize=(10,8))
sns.barplot(x='week',y='value', hue='variable',data=plot_by_week2_small)
plt.ylabel("Total Number of Cases Per Week")
plt.xticks(rotation = 45)
plt.legend(title="")
plt.savefig('Images/small_weekly.png', dpi=300, bbox_inches='tight')
plt.show()


# In[26]:


new_cases_major = major_affected.copy()
for row in range(1,len(major_affected.index)):
    new_cases_major.iloc[row,0:5] = (major_affected.iloc[row,0:5]-major_affected.iloc[row-1,0:5]).astype(int)

new_cases_north = northern_states.copy()
for row in range(1,len(northern_states.index)):
    new_cases_north.iloc[row,0:4] = (northern_states.iloc[row,0:4]-northern_states.iloc[row-1,0:4]).astype(int)

new_cases_east = eastern_states.copy()
for row in range(1,len(eastern_states.index)):
    new_cases_east.iloc[row,0:4] = (eastern_states.iloc[row,0:4]-eastern_states.iloc[row-1,0:4]).astype(int)

new_cases_small = small_states.copy()
for row in range(1,len(small_states.index)):
    new_cases_small.iloc[row,0:4] = (small_states.iloc[row,0:4]-small_states.iloc[row-1,0:4]).astype(int)


# In[27]:


err_del =[new_cases_major['Delhi'].iloc[0]]
err_mah = [new_cases_major['Maharashtra'].iloc[0]]
err_tn=[new_cases_major['Tamil Nadu'].iloc[0]]
err_ap=[new_cases_major["Andhra Pradesh"].iloc[0]]
err_kr=[new_cases_major['Karnataka'].iloc[0]]


for i in range(1,len(major_affected.index)):
    err_del.append(new_cases_major['Delhi'].iloc[i]-new_cases_major['Delhi'].iloc[i-1])
    err_mah.append(new_cases_major['Maharashtra'].iloc[i]-new_cases_major['Maharashtra'].iloc[i-1])
    err_tn.append(new_cases_major['Tamil Nadu'].iloc[i]-new_cases_major['Tamil Nadu'].iloc[i-1])
    err_ap.append(new_cases_major["Andhra Pradesh"].iloc[i]-new_cases_major["Andhra Pradesh"].iloc[i-1])
    err_kr.append(new_cases_major['Karnataka'].iloc[i]-new_cases_major['Karnataka'].iloc[i-1])

error = pd.DataFrame({"err_del":err_del, "err_mah":err_mah, "err_tn":err_tn, "err_kr":err_kr, "err_ap":err_ap}, index = new_cases_major.index, dtype = np.float32)

error['week'] = new_cases_major['week']
error = error.groupby(['week']).apply(np.std).drop(labels='week',axis=1)

week_avg = new_cases_major.groupby(['week']).sum()
week_avg['week'] = starting_week_dates
week_avg.drop(week_avg.tail(1).index,inplace=True)

for col in week_avg:
    if (col == 'week'):
        continue
    else:
        week_avg[col] = (week_avg[col]/7).astype(int)

week_avg = pd.concat([week_avg, error], axis=1)
week_avg.drop(week_avg.tail(1).index,inplace=True)


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Delhi'],
                     error_y=dict(type='data', array=np.array(week_avg['err_del'])
                                  ,visible = True,thickness =1),
                name='Delhi',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Maharashtra'],
                error_y=dict(type='data', array=np.array(week_avg['err_mah'])
                             ,visible = True, thickness =1),
                name='Maharashtra',
                marker_color='rgb(0,125,150)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Tamil Nadu'],
                error_y=dict(type='data', array=np.array(week_avg['err_tn'])
                             ,visible = True, thickness =1),
                name='Tamil Nadu',
                marker_color='rgb(13, 255, 20)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Andhra Pradesh'],
                error_y=dict(type='data', array=np.array(week_avg['err_ap'])
                             ,visible = True, thickness =1),
                name='Andhra Pradesh',
                marker_color='rgb(0, 200, 70)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Karnataka'],
                error_y=dict(type='data', array=np.array(week_avg['err_kr']) 
                             ,visible = True, thickness =1),
                name='Karnataka',
                marker_color='rgb(255, 10, 10)'
                ))

fig.update_layout(
    title='Distribution of Covid-19 New Cases Across Major Affected States' ,
    
    xaxis=dict(tickfont_size=14,
#                title = 'Week',
               ticktext= week_avg['week'],
               tickvals = list(range(1,new_cases_major.shape[0]))
              ),    
    yaxis=dict(
        title='Average new cases per week',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,# gap between bars of the same location coordinate.
    shapes=[
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="Jul 2020",
            y0=0,
            x1="Apr 2020",
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
#             x=15.5,
#             y=400,
            xref="x",
            yref="y",
            text="",
#             family="sans serif",
            showarrow=False
        )
    ]
)

# fig.update_xaxes(week_avg['week'])
fig.write_image('Images/major_affected_weekly_avg.png',width=800, height=600)
fig.show()


# In[28]:


# North - "Uttar Pradesh", "Punjab",  "Rajasthan", "Haryana"

err_up =[new_cases_north['Uttar Pradesh'].iloc[0]]
err_pun = [new_cases_north['Punjab'].iloc[0]]
err_rj=[new_cases_north['Rajasthan'].iloc[0]]
err_hr=[new_cases_north["Haryana"].iloc[0]]


for i in range(1,len(northern_states.index)):
    err_up.append(new_cases_north['Uttar Pradesh'].iloc[i]-new_cases_north['Uttar Pradesh'].iloc[i-1])
    err_pun.append(new_cases_north['Punjab'].iloc[i]-new_cases_north['Punjab'].iloc[i-1])
    err_rj.append(new_cases_north['Rajasthan'].iloc[i]-new_cases_north['Rajasthan'].iloc[i-1])
    err_hr.append(new_cases_north["Haryana"].iloc[i]-new_cases_north["Haryana"].iloc[i-1])

error = pd.DataFrame({"err_up":err_up, "err_pun":err_pun, "err_rj":err_rj, "err_hr":err_hr}, index = new_cases_north.index, dtype = np.float32)

error['week'] = new_cases_north['week']
error = error.groupby(['week']).apply(np.std).drop(labels='week',axis=1)

week_avg = new_cases_north.groupby(['week']).sum()
week_avg['week'] = starting_week_dates
week_avg.drop(week_avg.tail(1).index,inplace=True)

for col in week_avg:
    if (col == 'week'):
        continue
    else:
        week_avg[col] = (week_avg[col]/7).astype(int)

week_avg = pd.concat([week_avg, error], axis=1)
week_avg.drop(week_avg.tail(1).index,inplace=True)


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Uttar Pradesh'],
                     error_y=dict(type='data', array=np.array(week_avg['err_up'])
                                  ,visible = True,thickness =1),
                name='Uttar Pradesh',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Punjab'],
                error_y=dict(type='data', array=np.array(week_avg['err_pun'])
                             ,visible = True, thickness =1),
                name='Punjab',
                marker_color='rgb(0,125,150)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Rajasthan'],
                error_y=dict(type='data', array=np.array(week_avg['err_rj'])
                             ,visible = True, thickness =1),
                name='Rajasthan',
                marker_color='rgb(13, 255, 20)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Haryana'],
                error_y=dict(type='data', array=np.array(week_avg['err_hr'])
                             ,visible = True, thickness =1),
                name='Haryana',
                marker_color='rgb(0, 200, 70)'
                ))

fig.update_layout(
    title='Distribution of Covid-19 New Cases Across Northern States' ,
    
    xaxis=dict(tickfont_size=14,
#                title = 'Week',
               ticktext= week_avg['week'],
               tickvals = list(range(1,new_cases_north.shape[0]))
              ),    
    yaxis=dict(
        title='Average new cases per week',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,# gap between bars of the same location coordinate.
    shapes=[
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="Jul 2020",
            y0=0,
            x1="Apr 2020",
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
#             x=15.5,
#             y=400,
            xref="x",
            yref="y",
            text="",
#             family="sans serif",
            showarrow=False
        )
    ]
)

# fig.update_xaxes(week_avg['week'])
fig.write_image('Images/north_weekly_avg.png',width=800, height=600)
fig.show()


# In[29]:


# East - "West Bengal", "Jharkhand", "Bihar", "Odisha"

err_wb =[new_cases_east['West Bengal'].iloc[0]]
err_jk = [new_cases_east['Jharkhand'].iloc[0]]
err_bh=[new_cases_east['Bihar'].iloc[0]]
err_od=[new_cases_east["Odisha"].iloc[0]]


for i in range(1,len(eastern_states.index)):
    err_wb.append(new_cases_east['West Bengal'].iloc[i]-new_cases_east['West Bengal'].iloc[i-1])
    err_jk.append(new_cases_east['Jharkhand'].iloc[i]-new_cases_east['Jharkhand'].iloc[i-1])
    err_bh.append(new_cases_east['Bihar'].iloc[i]-new_cases_east['Bihar'].iloc[i-1])
    err_od.append(new_cases_east["Odisha"].iloc[i]-new_cases_east["Odisha"].iloc[i-1])

error = pd.DataFrame({"err_wb":err_wb, "err_jk":err_jk, "err_bh":err_bh, "err_od":err_od}, index = new_cases_east.index, dtype = np.float32)

error['week'] = new_cases_east['week']
error = error.groupby(['week']).apply(np.std).drop(labels='week',axis=1)

week_avg = new_cases_east.groupby(['week']).sum()
week_avg['week'] = starting_week_dates
week_avg.drop(week_avg.tail(1).index,inplace=True)

for col in week_avg:
    if (col == 'week'):
        continue
    else:
        week_avg[col] = (week_avg[col]/7).astype(int)

week_avg = pd.concat([week_avg, error], axis=1)
week_avg.drop(week_avg.tail(1).index,inplace=True)


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['West Bengal'],
                     error_y=dict(type='data', array=np.array(week_avg['err_wb'])
                                  ,visible = True,thickness =1),
                name='West Bengal',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Jharkhand'],
                error_y=dict(type='data', array=np.array(week_avg['err_jk'])
                             ,visible = True, thickness =1),
                name='Jharkhand',
                marker_color='rgb(0,125,150)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Bihar'],
                error_y=dict(type='data', array=np.array(week_avg['err_bh'])
                             ,visible = True, thickness =1),
                name='Bihar',
                marker_color='rgb(13, 255, 20)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Odisha'],
                error_y=dict(type='data', array=np.array(week_avg['err_od'])
                             ,visible = True, thickness =1),
                name='Odisha',
                marker_color='rgb(0, 200, 70)'
                ))

fig.update_layout(
    title='Distribution of Covid-19 New Cases Across Eastern States' ,
    
    xaxis=dict(tickfont_size=14,
#                title = 'Week',
               ticktext= week_avg['week'],
               tickvals = list(range(1,new_cases_east.shape[0]))
              ),    
    yaxis=dict(
        title='Average new cases per week',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,# gap between bars of the same location coordinate.
    shapes=[
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="Jul 2020",
            y0=0,
            x1="Apr 2020",
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
#             x=15.5,
#             y=400,
            xref="x",
            yref="y",
            text="",
#             family="sans serif",
            showarrow=False
        )
    ]
)

# fig.update_xaxes(week_avg['week'])
fig.write_image('Images/east_weekly_avg.png',width=800, height=600)
fig.show()


# In[30]:


# Small - "Chandigarh", "Goa", "Sikkim", "Tripura"

err_ch =[new_cases_small['Chandigarh'].iloc[0]]
err_goa = [new_cases_small['Goa'].iloc[0]]
err_sk=[new_cases_small['Sikkim'].iloc[0]]
err_tp=[new_cases_small["Tripura"].iloc[0]]


for i in range(1,len(eastern_states.index)):
    err_ch.append(new_cases_small['Chandigarh'].iloc[i]-new_cases_small['Chandigarh'].iloc[i-1])
    err_goa.append(new_cases_small['Goa'].iloc[i]-new_cases_small['Goa'].iloc[i-1])
    err_sk.append(new_cases_small['Sikkim'].iloc[i]-new_cases_small['Sikkim'].iloc[i-1])
    err_tp.append(new_cases_small["Tripura"].iloc[i]-new_cases_small["Tripura"].iloc[i-1])

error = pd.DataFrame({"err_ch":err_ch, "err_goa":err_goa, "err_sk":err_sk, "err_tp":err_tp}, index = new_cases_small.index, dtype = np.float32)

error['week'] = new_cases_small['week']
error = error.groupby(['week']).apply(np.std).drop(labels='week',axis=1)

week_avg = new_cases_small.groupby(['week']).sum()
week_avg['week'] = starting_week_dates
week_avg.drop(week_avg.tail(1).index,inplace=True)

for col in week_avg:
    if (col == 'week'):
        continue
    else:
        week_avg[col] = (week_avg[col]/7).astype(int)

week_avg = pd.concat([week_avg, error], axis=1)
week_avg.drop(week_avg.tail(1).index,inplace=True)


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Chandigarh'],
                     error_y=dict(type='data', array=np.array(week_avg['err_ch'])
                                  ,visible = True,thickness =1),
                name='Chandigarh',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Goa'],
                error_y=dict(type='data', array=np.array(week_avg['err_goa'])
                             ,visible = True, thickness =1),
                name='Goa',
                marker_color='rgb(0,125,150)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Sikkim'],
                error_y=dict(type='data', array=np.array(week_avg['err_sk'])
                             ,visible = True, thickness =1),
                name='Sikkim',
                marker_color='rgb(13, 255, 20)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Tripura'],
                error_y=dict(type='data', array=np.array(week_avg['err_tp'])
                             ,visible = True, thickness =1),
                name='Tripura',
                marker_color='rgb(0, 200, 70)'
                ))

fig.update_layout(
    title='Distribution of Covid-19 New Cases Across Small States' ,
    
    xaxis=dict(tickfont_size=14,
#                title = 'Week',
               ticktext= week_avg['week'],
               tickvals = list(range(1,new_cases_small.shape[0]))
              ),    
    yaxis=dict(
        title='Average new cases per week',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,# gap between bars of the same location coordinate.
    shapes=[
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="Jul 2020",
            y0=0,
            x1="Apr 2020",
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
#             x=15.5,
#             y=400,
            xref="x",
            yref="y",
            text="",
#             family="sans serif",
            showarrow=False
        )
    ]
)

# fig.update_xaxes(week_avg['week'])
fig.write_image('Images/small_weekly_avg.png',width=800, height=600)
fig.show()


# In[31]:


error = pd.DataFrame({"err_del":err_del, "err_mah":err_mah, "err_tn":err_tn, "err_rest":err_rest, "err_ap":err_ap}, index = new_cases.index, dtype = np.float32)


# In[32]:


error['week'] = new_cases['week']
error = error.groupby(['week']).apply(np.std).drop(labels='week',axis=1)


# In[33]:


week_avg = new_cases.groupby(['week']).sum()
week_avg['week'] = starting_week_dates
week_avg.drop(week_avg.tail(1).index,inplace=True)
week_avg


# In[34]:


for col in week_avg:
    if (col == 'week'):
        continue
    else:
        week_avg[col] = (week_avg[col]/7).astype(int)


# In[35]:


week_avg = pd.concat([week_avg, error], axis=1)


# In[36]:


week_avg.drop(week_avg.tail(1).index,inplace=True)


# In[37]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Delhi'],
                     error_y=dict(type='data', array=np.array(week_avg['err_del'])
                                  ,visible = True,thickness =1),
                name='Delhi',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Maharashtra'],
                error_y=dict(type='data', array=np.array(week_avg['err_mah'])
                             ,visible = True, thickness =1),
                name='Maharashtra',
                marker_color='rgb(0,125,150)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Tamil Nadu'],
                error_y=dict(type='data', array=np.array(week_avg['err_tn'])
                             ,visible = True, thickness =1),
                name='Tamil Nadu',
                marker_color='rgb(13, 255, 20)'
                ))
fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
                y=week_avg['Andhra Pradesh'],
                error_y=dict(type='data', array=np.array(week_avg['err_ap'])
                             ,visible = True, thickness =1),
                name='Andhra Pradesh',
                marker_color='rgb(0, 200, 70)'
                ))
# fig.add_trace(go.Bar(x=list(range(1,weeks+1)),
#                 y=week_avg['Rest of India'],
#                 error_y=dict(type='data', array=np.array(week_avg['err_rest']) 
#                              ,visible = True, thickness =1),
#                 name='Rest of India',
#                 marker_color='rgb(255, 10, 10)'
#                 ))

fig.update_layout(
    title='Distribution of Covid-19 Cases Across India' ,
    
    xaxis=dict(tickfont_size=14,
#                title = 'Week',
               ticktext= week_avg['week'],
               tickvals = list(range(1,plot_cases.shape[0]))
              ),    
    yaxis=dict(
        title='Average new cases per week',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,# gap between bars of the same location coordinate.
    shapes=[
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="Jul 2020",
            y0=0,
            x1="Apr 2020",
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
#             x=15.5,
#             y=400,
            xref="x",
            yref="y",
            text="",
#             family="sans serif",
            showarrow=False
        )
    ]
)

# fig.update_xaxes(week_avg['week'])
fig.write_image('Images/weekly_avg.png',width=800, height=600)
fig.show()




