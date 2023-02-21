#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
import base64
from IPython.core.display import display, HTML
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering


# In[2]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[3]:


df = pd.read_csv('Data/2023-02-08-DATA624-Assignment4-Data.csv')


# In[4]:


### Clustering
#### 1. KMeans
#### Elbow method for KMeans
inertia = []
for i in range(1,10):
    kmeans = KMeans(
        n_clusters = i, # Number of clusters to find
        init = "k-means++", # How to place the initial cluster centroids,
        max_iter= 1000, # Maximum number of iterations for the algorithm to run
        tol=0.01, # Roughly how much the centroids need to change between iterations to keep going
    ).fit(
        df
    )
    #inertia here is the sum of squared distances of samples to their closest cluster center.
    inertia.append(kmeans.inertia_)


# In[5]:


fig=px.line(y = inertia,x = range(1,10), markers=True)
fig.update_layout(title='<b>Figure 1. Elbow method for KMeans</b>')
fig.update_xaxes(title='Number of clusters')
fig.update_yaxes(title='Inertia')
fig.update_layout(width=800, height=600)

# Show the plot
fig.show()


# In[6]:


#### Silhouette method for KMeans
# by Silhouette method, we can see that the optimal number of clusters are 4 for Kmeans clustering method
sscore = []
# starting with 2 clusters now becuase the silhouette isn't defined for 1
for i in range(2,10):
    kmeans = KMeans(
        n_clusters = i, # Number of clusters to find
        init = "k-means++", # How to place the initial cluster centroids,
        max_iter= 500, # Maximum number of iterations for the algorithm to run
        tol=0.001, # Roughly how much the centroids need to change between iterations to keep going
    ).fit(
        df
    )
    
    silhouette_avg = silhouette_score(df, kmeans.labels_)
    sscore.append(silhouette_avg)
    


# In[7]:


fig1=px.line(y = sscore,x = range(2,10), markers=True)
fig1.update_layout(title='<b>Figure 2. Silhouette method for KMeans</b>')
fig1.update_xaxes(title='Number of clusters')
fig1.update_yaxes(title='Silhouette score')
fig1.update_layout(width=800, height=600)

# Show the plot
fig1.show()


# In[8]:


#### Visualization with t-SNE
# Fitting KMeans using best n_clusters (4)

kmeans = KMeans(
    n_clusters = 4, # Number of clusters to find
    init = "k-means++", # How to place the initial cluster centroids,
    max_iter= 500, # Maximum number of iterations for the algorithm to run
    tol=0.001, # Roughly how much the centroids need to change between iterations to keep going
).fit(
    df
)

inertia.append(kmeans.inertia_)


# In[9]:


# Initialize dataframe for storing labels
kmeansdf = pd.DataFrame(kmeans.labels_)

# Rename column
kmeansdf.rename(columns={0:'KM_labels'}, inplace=True)

kmeansdf


# In[10]:


# Fitting TSNE to the data 
ts = TSNE(
    perplexity=100, # Roughly the "size" of the clusters to look for (original paper
                   # recommends in the 5-50 range, but in general should be less than
                   # then number of points in your dataset
    learning_rate="auto",
    n_iter=2000,
    init='pca',
).fit_transform(df)


# In[11]:


# To (properly) plot using TSNE, I will 
# create one dataframe (km_df) which includes both
# the 2-D TSNE results and the KM labels

km_df = pd.concat([pd.DataFrame(ts),kmeansdf['KM_labels']])

km_df['clusters'] = kmeansdf['KM_labels']


# In[12]:


# TSNE plot to visualize the clusters identified 
# through KMEANS
fig3=px.scatter(
    km_df,
    x=0,
    y=1,
    color='clusters'
)
fig3.update_layout(title='<b>Figure 3. Four clusters identified with help of KMeans</b>')

fig3.update_layout(width=800, height=600)

# Show the plot
fig3.show()


# In[15]:


#### 2. DBscan
#### Silhouette method for DBscan
sscore = []
# starting with 2 clusters now becuase the silhouette isn't defined for 1
for i in np.arange(1, 5.0, 0.2):
    dnscan = DBSCAN(
    min_samples=35,
    eps= i # Roughly how much the centroids need to change between iterations to keep going
    ).fit(
        df
    )
    
    silhouette_avg = silhouette_score(df, dnscan.labels_)
    sscore.append(silhouette_avg)


# In[16]:


fig4=px.line(y = sscore,x =np.arange(1, 5.0, 0.2) , markers=True)
fig4.update_layout(title='<b>Figure 4. Silhouette method for DBscan</b>')
fig4.update_xaxes(title='Epsilon')
fig4.update_yaxes(title='Silhouette score')
fig4.update_layout(width=800, height=600)

# Show the plot
fig4.show()
#optimal eps is 3


# In[17]:


#### Visualization with t-SNE
# Fitting DBscan using best eps (3)
clusters = DBSCAN(
    min_samples=35,
    eps= 3
).fit(
    df
)
# Initialize dataframe for storing labels
dbdf = pd.DataFrame(clusters.labels_)

# Rename column
dbdf.rename(columns={0:'DB_labels'}, inplace=True)


# In[18]:


db_df = pd.concat([pd.DataFrame(ts),dbdf['DB_labels']])

db_df['clusters'] = dbdf['DB_labels']


# In[19]:


fig5=px.scatter(
    db_df,
    x=0,
    y=1,
    color='clusters'
)
fig5.update_layout(title='<b>Figure 5. Five clusters identified with help of DBscan</b>')

fig5.update_layout(width=800, height=600)

# Show the plot
fig5.show()


# In[20]:


#### 3. Agglomerative Clustering
#### Silhouette method for Agglomerative Clustering
sscore = []
# starting with 2 clusters now becuase the silhouette isn't defined for 1
for i in range(2,10):
    agglom = AgglomerativeClustering(
    n_clusters = i, # Number of clusters to find
    affinity='euclidean',
    linkage='ward'
    ).fit(
    df
    )
    
    silhouette_avg = silhouette_score(df, agglom.labels_)
    sscore.append(silhouette_avg)
    


# In[21]:


fig6=px.line(y = sscore,x = range(2,10), markers=True)
fig6.update_layout(title='<b>Figure 6. Silhouette method for Agglomerative Clustering</b>')
fig6.update_xaxes(title='Number of Clusters')
fig6.update_yaxes(title='Silhouette score')
fig6.update_layout(width=800, height=600)

# Show the plot
fig6.show()


# In[22]:


#### Visualization with t-SNE
# Fitting Agglomerative Clustering using best n of clusters 4
clusters_agg = AgglomerativeClustering(
    n_clusters = 4, # Number of clusters to find
    affinity='euclidean',
    linkage='ward'
).fit(
    df
)


# In[23]:


# Initialize dataframe for storing labels
agg_df = pd.DataFrame(clusters_agg.labels_)

# Rename column
agg_df.rename(columns={0:'cluster'}, inplace=True)


# In[24]:


agglom_df = pd.concat([pd.DataFrame(ts),agg_df['cluster']])

agglom_df['clusters'] = agg_df['cluster']


# In[25]:


fig7=px.scatter(
    agglom_df,
    x=0,
    y=1,
    color='clusters'
)
fig7.update_layout(title='<b>Figure 7. Four clusters identified with help of Agglomerative Clustering</b>')

fig7.update_layout(width=800, height=600)

# Show the plot
fig7.show()


# In[50]:


app.layout = html.Div(
    [
        html.H1("Clustering Analysis"),
        """
       In order to choose number of clusters for our dataset, we will use Elbow and Silhouette method based on KMeans, DBscan and Agglomerative Clustering methods.
        """,
        html.Center(
        html.P('Clustering identification with help of KMeans', style={'font-weight': 'bold'}),
        ),
        html.Div(
            [
            dcc.Graph(
                figure=fig,
                style={
                    "width": "80%",
                    "height": "100vh",
                },
                id="OurFirstFigure",
            ), 
        # html.P('This is some text between the figures.'),
            dcc.Graph(
                figure=fig1,
                style={
                    "width": "80%",
                    "height": "100vh",
                },
                id="OurSecondFigure",
            ), 
        ],
            style={
                'display': 'flex',
                'flex-direction': 'row',
                'justify-content': 'center',
                'align-items': 'center',
                'margin': '20px 0'
            }
        ),
        html.Center(
         dcc.Graph(
                figure=fig3,
                style={
                    "width": "100%",
                    "height": "100vh"
                },
                id="OurThirdFigure"
        ),
    ),
    html.Center(
    html.P('Clustering identification with help of DBscan', style={'font-weight': 'bold'}),
        ),
    html.Div(
            [
            dcc.Graph(
                figure=fig4,
                style={
                    "width": "80%",
                    "height": "100vh",
                },
                id="OurFourthFigure",
            ), 
        # html.P('This is some text between the figures.'),
            dcc.Graph(
                figure=fig5,
                style={
                    "width": "80%",
                    "height": "100vh",
                },
                id="OurFifthFigure",
            ), 
        ],
            style={
                'display': 'flex',
                'flex-direction': 'row',
                'justify-content': 'center',
                'align-items': 'center',
                'margin': '20px 0'
            }
        ),
            html.Center(
    html.P('Clustering identification with help of Agglomerative Clustering', style={'font-weight': 'bold'}),
        ),
    html.Div(
            [
            dcc.Graph(
                figure=fig6,
                style={
                    "width": "80%",
                    "height": "100vh",
                },
                id="OurSixthFigure",
            ), 
        # html.P('This is some text between the figures.'),
            dcc.Graph(
                figure=fig7,
                style={
                    "width": "80%",
                    "height": "100vh",
                },
                id="OurSeventhFigure",
            ), 
        ],
            style={
                'display': 'flex',
                'flex-direction': 'row',
                'justify-content': 'center',
                'align-items': 'center',
                'margin': '20px 0'
            }
        ),
        
]
)

if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




