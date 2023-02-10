import sys, os
import pandas as pd # working with data
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import plotly.express as px # visualization
import datetime as dt # time and date
import streamlit as st # streamlit

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from PIL import Image

st.title("Segmentasi Tenant dengan Teknik Clustering K-Means dan Model RFM")
st.subheader("Studi Kasus: Event Semarang Great Sale 2022")

img_1, img_2, img_3 = st.columns(3)
with img_1:
    img_sm_one = Image.open('sm_one.png')
    st.image(img_sm_one, caption='Event Semargres 2022')

with img_2:
    img_elbow_two = Image.open('sm_two.png')
    st.image(img_elbow_two, caption='Event Semargres 2022')

with img_3:
    img_sm_three = Image.open('sm_three.png')
    st.image(img_sm_three, caption='Event Semargres 2022')

col1, col2, col3 = st.columns(3)

with col1:
    # import data kategori
    st.subheader("Tabel Data Kategori")
    kategori = pd.read_csv('kategori.csv', encoding='unicode_escape')
    st.dataframe(kategori)
    st.write('Total dataset kategori : ', len(kategori))

with col2:
    # import data tenant
    st.subheader("Tabel Data Tenant")
    tenant = pd.read_csv('tenant.csv', encoding='unicode_escape')
    st.dataframe(tenant)
    st.write('Total dataset tenant : ', len(tenant))

with col3:
    # import data transaksi
    st.subheader("Tabel Data Transaksi")
    transaksi = pd.read_csv('transaksi.csv', encoding='unicode_escape')
    transaksi["tanggal"] = pd.to_datetime(transaksi["tanggal"])
    st.dataframe(transaksi)
    st.write('Total dataset transaksi : ', len(transaksi))

# data selection
st.header("Data Selection")
data_set = transaksi.merge(tenant, on="id_tenant", how="left")
data_set = data_set[["id_trx", "id_k", "id_tenant", "nama", "id_user", "tanggal", "total"]]
data_set = data_set.merge(kategori, on="id_k", how="left")
data_set = data_set[["id_trx", "id_k", "nama_kat", "id_tenant", "nama", "tanggal", "total"]]
st.dataframe(data_set)
st.write('Total dataset : ', len(data_set))

# data cleaning
st.header("Data Cleaning")
st.write(data_set.isnull().sum())
data_set.dropna(inplace=True)
st.dataframe(data_set)
st.write('Total dataset : ', len(data_set))


# data transformation
st.header("Data Transformation")
start_date = dt.datetime(2022, 7, 20)
end_date = dt.datetime(2022, 8, 20)
st.write('Periode Penelitian : {}  sampai  {}'.format(start_date.strftime("%d-%B-%Y"), end_date.strftime("%d-%B-%Y")))
data_set = data_set.loc[(data_set["tanggal"] >= start_date) & (data_set["tanggal"] <= end_date)]
st.dataframe(data_set)
st.write('Total dataset : ', len(data_set))

# get top 3 category
st.subheader("Top 3 Category")
kategori_dt = tenant.copy()
kategori_dt = kategori_dt[kategori_dt["status"] == 1]
kategori_dt.isnull().sum()
kategori_dt.dropna(inplace=True)

kategori_df = kategori_dt.groupby(["id_k"]).agg({'id_tenant': lambda x : x.count()})
kategori_df.rename(columns = {
                        'id_tenant' : 'total'}, inplace = True)
kategori_df.reset_index(inplace = True)
kategori_df = kategori_df.sort_values(by='total', ascending=False)[:3]
kategori_df = kategori_df.merge(kategori, on="id_k")
st.dataframe(kategori_df)

kategori_1 = kategori_df['id_k'].iloc[0]
kategori_nama_1 = kategori_df['nama_kat'].iloc[0]
kategori_2 = kategori_df['id_k'].iloc[1]
kategori_nama_2 = kategori_df['nama_kat'].iloc[1]
kategori_3 = kategori_df['id_k'].iloc[2]
kategori_nama_3 = kategori_df['nama_kat'].iloc[2]

tab1, tab2, tab3 = st.tabs([kategori_nama_1, kategori_nama_2, kategori_nama_3])

# kategori 1
with tab1:
    dataset_one = data_set[data_set["id_k"] == kategori_1]
    st.dataframe(dataset_one)
    st.write('Total dataset : ', len(dataset_one))

    st.subheader("Top 5 Tenant - Category {}".format(kategori_nama_1))
    trx_category_1 = dataset_one.groupby(["nama"]).agg({'id_tenant': lambda x : x.count()}).sort_values(by="id_tenant", ascending=False)[:5]
    trx_category_1.rename(columns = {
                            'id_tenant' : 'total'}, inplace = True)
    trx_category_1.reset_index(inplace = True)
    st.dataframe(trx_category_1)

    title_kategori_1 = 'Top 5 Tenant Category {} by Frequency'.format(kategori_nama_1)
    st.subheader(title_kategori_1)
    fig = px.bar(x=trx_category_1['nama'], y=trx_category_1['total'])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    col1_1, col1_2 = st.columns(2)

    # RFM ANALYSIS MODEL    
    with col1_1: 
        st.subheader("RFM ANALYSIS MODEL")
        st.text('Recency = tanggal transaksi terakhir')
        st.text('Frequency = jumlah transaksi ')
        st.text('Monetary = total transaksi')
        rfm_dt_one = dataset_one[["id_tenant","tanggal", "total"]].groupby(["id_tenant"]).agg({
                            'tanggal' : lambda x : ((end_date - x.max()).days) + 1, 
                            'id_tenant' : lambda x : x.count(), 
                            'total' : lambda x : sum(x) })

        rfm_dt_one.rename(columns = {
                                'tanggal' : 'Recency', 
                                'id_tenant' : 'Frequency', 
                                'total' : 'Monetary' }, inplace = True)

        st.dataframe(rfm_dt_one)
        st.write('Total dataset : ', len(rfm_dt_one))

    with col1_2:
        # normalize
        st.subheader("Normalisasi Data RFM")
        # scaler
        sc_one = MinMaxScaler((0, 1))
        km_scaled_one = pd.DataFrame(sc_one.fit_transform(rfm_dt_one[["Recency", "Frequency", "Monetary"]]),
                                    index=rfm_dt_one.index, columns=["recency", "frequency", "monetary"])
        km_scaled_one.reset_index(inplace=True)
        st.dataframe(km_scaled_one)

    # K-Means
    st.subheader("Metode Elbow")
    kmeans_one = KMeans(random_state=42)
    elbow_one = KElbowVisualizer(kmeans_one, k=(2, 15))
    elbow_one.fit(km_scaled_one[['recency', 'frequency', 'monetary']])
    elbow_one.show()
    # outpath='elbowplot1.png'
    img_elbow_one = Image.open('elbowplot1.png')
    st.image(img_elbow_one, caption='Elbow Method')
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    #Calculating the silhoutte coefficient
    st.subheader("Silhoutte Coefficient")
    sc_score_one = []
    for n_cluster in range(4, 15):
        kmeans_one = KMeans(n_clusters=n_cluster, random_state=42).fit(km_scaled_one[['recency', 'frequency', 'monetary']])
        label = kmeans_one.labels_
        sil_coeff = silhouette_score(km_scaled_one[['recency', 'frequency', 'monetary']], label, metric="euclidean")
        sc_score_one.append([n_cluster, sil_coeff])
    st.write(pd.DataFrame(sc_score_one))

    fig = px.line(x=pd.DataFrame(sc_score_one)[0], y=pd.DataFrame(sc_score_one)[1])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.header("K-Means Clustering")
    kmeans_one = KMeans(n_clusters=elbow_one.elbow_value_, random_state=42).fit(km_scaled_one[['recency', 'frequency', 'monetary']])
    km_df_one = km_scaled_one
    km_df_one["clusters"] = kmeans_one.labels_ 

    km_dt_one = rfm_dt_one.copy()
    km_dt_one['clusters'] = km_df_one["clusters"].values
    st.dataframe(km_dt_one)

    st.subheader("Plot of Tenant Distribution")
    fig = px.scatter_3d(km_dt_one[["Recency", "Frequency", "Monetary"]], x='Recency', y='Frequency', z='Monetary', width=1000, height=800,
              color=km_dt_one["clusters"])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Tenant Distribution of Cluster")
    col1_3, col1_4 = st.columns(2)
    with col1_3:
        km_cluster_one = km_df_one.groupby("clusters").agg({"id_tenant": "count"})
        km_cluster_one.reset_index(inplace=True)
        km_cluster_one.columns = ['clusters', 'count']
        st.dataframe(km_cluster_one)

    with col1_4:
        fig = px.pie(km_cluster_one, values='count', names='clusters')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Grafik Hasil Clustering Berdasarkan Rata-Rata Nilai RFM")
    cluster_df_one = km_df_one
    fig = px.histogram(cluster_df_one, x='clusters', y=['recency', 'frequency', 'monetary'], barmode='group', histfunc='avg', height=500)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Kesimpulan")
    st.text("Cluster 1 => Platinum")
    st.text("Cluster 3 => Gold")
    st.text("Cluster 5 => Silver")
    st.text("Cluster 0,2,4 => Bronze")


# kategori 2
with tab2:
    dataset_two = data_set[data_set["id_k"] == kategori_2]
    st.dataframe(dataset_two)
    st.write('Total dataset : ', len(dataset_two))

    st.subheader("Top 5 Tenant - Category {}".format(kategori_nama_2))
    trx_category_2 = dataset_two.groupby(["nama"]).agg({'id_tenant': lambda x : x.count()}).sort_values(by="id_tenant", ascending=False)[:5]
    trx_category_2.rename(columns = {
                            'id_tenant' : 'total'}, inplace = True)
    trx_category_2.reset_index(inplace = True)
    st.dataframe(trx_category_2)

    title_kategori_2 = 'Top 5 Tenant Category {} by Frequency'.format(kategori_nama_2)
    st.subheader(title_kategori_2)
    fig = px.bar(x=trx_category_2['nama'], y=trx_category_2['total'])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    col2_1, col2_2 = st.columns(2)

    with col2_1:
        # RFM ANALYSIS MODEL
        st.subheader("RFM ANALYSIS MODEL")
        st.text('Recency = tanggal transaksi terakhir')
        st.text('Frequency = jumlah transaksi ')
        st.text('Monetary = total transaksi')
        rfm_dt_two = dataset_two[["id_tenant","tanggal", "total"]].groupby(["id_tenant"]).agg({
                            'tanggal' : lambda x : ((end_date - x.max()).days) + 1, 
                            'id_tenant' : lambda x : x.count(), 
                            'total' : lambda x : sum(x) })

        rfm_dt_two.rename(columns = {
                                'tanggal' : 'Recency', 
                                'id_tenant' : 'Frequency', 
                                'total' : 'Monetary' }, inplace = True)

        st.dataframe(rfm_dt_two)
        st.write('Total dataset : ', len(rfm_dt_two))

    with col2_2:
        # normalize
        st.subheader("Normalisasi Data RFM")
        # scaler
        sc_two = MinMaxScaler((0, 1))
        km_scaled_two = pd.DataFrame(sc_two.fit_transform(rfm_dt_two[["Recency", "Frequency", "Monetary"]]),
                                    index=rfm_dt_two.index, columns=["recency", "frequency", "monetary"])
        km_scaled_two.reset_index(inplace=True)
        st.dataframe(km_scaled_two)

    # K-Means
    st.subheader("Metode Elbow")
    kmeans_two = KMeans(random_state=42)
    elbow_two = KElbowVisualizer(kmeans_two, k=(2, 12))
    elbow_two.fit(km_scaled_two[['recency', 'frequency', 'monetary']])
    elbow_two.show()
    # outpath='elbowplot2.png'
    img_elbow_two = Image.open('elbowplot2.png')
    st.image(img_elbow_two, caption='Elbow Method')
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    #Calculating the silhoutte coefficient
    st.subheader("Silhoutte Coefficient")
    sc_score_two = []
    for n_cluster in range(4, 9):
        kmeans_two = KMeans(n_clusters=n_cluster, random_state=42).fit(km_scaled_two[['recency', 'frequency', 'monetary']])
        label = kmeans_two.labels_
        sil_coeff = silhouette_score(km_scaled_two[['recency', 'frequency', 'monetary']], label, metric="euclidean")
        sc_score_two.append([n_cluster, sil_coeff])
    st.write(pd.DataFrame(sc_score_two))

    fig = px.line(x=pd.DataFrame(sc_score_two)[0], y=pd.DataFrame(sc_score_two)[1])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.header("K-Means Clustering")
    kmeans_two = KMeans(n_clusters=elbow_two.elbow_value_, random_state=42).fit(km_scaled_two[['recency', 'frequency', 'monetary']])
    km_df_two = km_scaled_two
    km_df_two["clusters"] = kmeans_two.labels_ 

    km_dt_two = rfm_dt_two.copy()
    km_dt_two['clusters'] = km_df_two["clusters"].values
    st.dataframe(km_dt_two)

    st.subheader("Plot of Tenant Distribution")
    fig = px.scatter_3d(km_dt_two[["Recency", "Frequency", "Monetary"]], x='Recency', y='Frequency', z='Monetary', width=1000, height=800,
              color=km_dt_two["clusters"])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Tenant Distribution of Cluster")
    col2_3, col2_4 = st.columns(2)
    with col2_3:
        km_cluster_two = km_df_two.groupby("clusters").agg({"id_tenant": "count"})
        km_cluster_two.reset_index(inplace=True)
        km_cluster_two.columns = ['clusters', 'count']
        st.dataframe(km_cluster_two)

    with col2_4:
        fig = px.pie(km_cluster_two, values='count', names='clusters')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    
    st.subheader("Grafik Hasil Clustering Berdasarkan Rata-Rata Nilai RFM")
    cluster_df_two = km_df_two

    fig = px.histogram(cluster_df_two, x='clusters', y=['recency', 'frequency', 'monetary'], barmode='group', histfunc='avg', height=500)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Kesimpulan")
    st.text("Cluster 1 => Platinum")
    st.text("Cluster 3 => Gold")
    st.text("Cluster 0 => Silver")
    st.text("Cluster 2 => Bronze")

# kategori 3
with tab3:
    dataset_three = data_set[data_set["id_k"] == kategori_3]
    st.dataframe(dataset_three)
    st.write('Total dataset : ', len(dataset_three))

    st.subheader("Top 5 Tenant - Category {}".format(kategori_nama_3))
    trx_category_3 = dataset_three.groupby(["nama"]).agg({'id_tenant': lambda x : x.count()}).sort_values(by="id_tenant", ascending=False)[:5]
    trx_category_3.rename(columns = {
                            'id_tenant' : 'total'}, inplace = True)
    trx_category_3.reset_index(inplace = True)
    st.dataframe(trx_category_3)

    title_kategori_3 = 'Top 5 Tenant Category {} by Frequency'.format(kategori_nama_3)
    st.subheader(title_kategori_3)
    fig = px.bar(x=trx_category_3['nama'], y=trx_category_3['total'])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    col3_1, col3_2 = st.columns(2)
    with col3_1:
        # RFM ANALYSIS MODEL
        st.subheader("RFM ANALYSIS MODEL")
        st.text('Recency = tanggal transaksi terakhir')
        st.text('Frequency = jumlah transaksi ')
        st.text('Monetary = total transaksi')
        rfm_dt_three = dataset_three[["id_tenant","tanggal", "total"]].groupby(["id_tenant"]).agg({
                            'tanggal' : lambda x : ((end_date - x.max()).days) + 1, 
                            'id_tenant' : lambda x : x.count(), 
                            'total' : lambda x : sum(x) })

        rfm_dt_three.rename(columns = {
                                'tanggal' : 'Recency', 
                                'id_tenant' : 'Frequency', 
                                'total' : 'Monetary' }, inplace = True)

        st.dataframe(rfm_dt_three)
        st.write('Total dataset : ', len(rfm_dt_three))

    with col3_2:
        # normalize
        st.subheader("Normalisasi Data RFM")
        # scaler
        sc_three = MinMaxScaler((0, 1))
        km_scaled_three = pd.DataFrame(sc_three.fit_transform(rfm_dt_three[["Recency", "Frequency", "Monetary"]]),
                                    index=rfm_dt_three.index, columns=["recency", "frequency", "monetary"])
        km_scaled_three.reset_index(inplace=True)
        st.dataframe(km_scaled_three)

    # K-Means
    st.subheader("Metode Elbow")
    kmeans_three = KMeans(random_state=42)
    elbow_three = KElbowVisualizer(kmeans_three, k=(3, 10))
    elbow_three.fit(km_scaled_three[['recency', 'frequency', 'monetary']])
    elbow_three.show()
    # outpath='elbowplot3.png'
    img_elbow_three = Image.open('elbowplot3.png')
    st.image(img_elbow_three, caption='Elbow Method')
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    #Calculating the silhoutte coefficient
    st.subheader("Silhoutte Coefficient")
    sc_score_three = []
    for n_cluster in range(3, 10):
        kmeans_three = KMeans(n_clusters=n_cluster, random_state=42).fit(km_scaled_three[['recency', 'frequency', 'monetary']])
        label = kmeans_three.labels_
        sil_coeff = silhouette_score(km_scaled_three[['recency', 'frequency', 'monetary']], label, metric="euclidean")
        sc_score_three.append([n_cluster, sil_coeff])
    st.write(pd.DataFrame(sc_score_three))

    fig = px.line(x=pd.DataFrame(sc_score_three)[0], y=pd.DataFrame(sc_score_three)[1])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.header("K-Means Clustering")
    kmeans_three = KMeans(n_clusters=elbow_three.elbow_value_, random_state=42).fit(km_scaled_three[['recency', 'frequency', 'monetary']])
    km_df_three = km_scaled_three
    km_df_three["clusters"] = kmeans_three.labels_ 

    km_dt_three = rfm_dt_three.copy()
    km_dt_three['clusters'] = km_df_three["clusters"].values
    st.dataframe(km_dt_three)

    st.subheader("Plot of Tenant Distribution")
    fig = px.scatter_3d(km_dt_three[["Recency", "Frequency", "Monetary"]], x='Recency', y='Frequency', z='Monetary', width=1000, height=800,
              color=km_dt_three["clusters"])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Tenant Distribution of Cluster")
    col3_3, col3_4 = st.columns(2)
    with col3_3:
        km_cluster_three = km_df_three.groupby("clusters").agg({"id_tenant": "count"})
        km_cluster_three.reset_index(inplace=True)
        km_cluster_three.columns = ['clusters', 'count']
        st.dataframe(km_cluster_three)

    with col3_4:
        fig = px.pie(km_cluster_three, values='count', names='clusters')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Grafik Hasil Clustering Berdasarkan Rata-Rata Nilai RFM")
    cluster_df_three = km_df_three

    fig = px.histogram(cluster_df_three, x='clusters', y=['recency', 'frequency', 'monetary'], barmode='group', histfunc='avg', height=500)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Kesimpulan")
    st.text("Cluster 1 => Platinum")
    st.text("Cluster 4 => Gold")
    st.text("Cluster 2 => Silver")
    st.text("Cluster 0,3 => Bronze")